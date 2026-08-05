"""Single-GPU TurboVLA training entry for GR3, invoked by Interactive Training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from turbovla.data.gr3_anygrasp import GR3_ANYGRASP_PROFILE_ID, Gr3AnygraspDataset
from turbovla.data.gr3_common import (
    GR3_ACTION_DIM,
    GR3_MODEL_ACTION_DIM,
    GR3_MODEL_STATE_DIM,
    GR3_STATE_DIM,
    Gr3NormalizationStats,
)
from turbovla.data.gr3_dagger import GR3_PROFILE_ID, Gr3DaggerDataset
from turbovla.models import TurboVLAConfig, build_turbovla
from turbovla.models.configuration import (
    ActionHeadConfig,
    InteractionConfig,
    TextEncoderConfig,
    VisionEncoderConfig,
)


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if isinstance(value.get(key), dict):
                return value[key]
    if not isinstance(value, dict):
        raise TypeError("initial checkpoint must contain a state dictionary")
    return value


def _load_compatible(model: torch.nn.Module, path: Path) -> tuple[int, int]:
    source = _checkpoint_state(path)
    cleaned = {}
    for key, value in source.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("model."):
            key = key[6:]
        cleaned[key] = value
    source = cleaned
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items() if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    migrated_projection_keys = {
        "action_head.decoder.action_projection.layers.2.weight",
        "action_head.decoder.action_projection.layers.2.bias",
    }
    for key in migrated_projection_keys:
        if key not in source or key not in target or key in compatible:
            continue
        source_value = source[key]
        target_value = target[key]
        if (
            source_value.ndim == target_value.ndim
            and source_value.shape[0] in {33, GR3_ACTION_DIM}
            and target_value.shape[0] == GR3_MODEL_ACTION_DIM
            and tuple(source_value.shape[1:]) == tuple(target_value.shape[1:])
        ):
            compatible[key] = source_value[:GR3_MODEL_ACTION_DIM].clone()
    migrated_state_projection_keys = {
        "action_head.state_projection.net.0.weight": 0,
        "action_head.state_projection.net.0.bias": 0,
        "action_head.state_projection.net.1.weight": 1,
    }
    for key, axis in migrated_state_projection_keys.items():
        if key not in source or key not in target or key in compatible:
            continue
        source_value = source[key]
        target_value = target[key]
        if (
            source_value.shape[axis] == GR3_STATE_DIM
            and target_value.shape[axis] == GR3_MODEL_STATE_DIM
        ):
            index = [slice(None)] * source_value.ndim
            index[axis] = slice(0, GR3_MODEL_STATE_DIM)
            candidate = source_value[tuple(index)].clone()
            if tuple(candidate.shape) == tuple(target_value.shape):
                compatible[key] = candidate
    if not compatible:
        raise RuntimeError(f"no compatible TurboVLA tensors found in {path}")
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(source) - len(compatible)


def _load_normalization(
    path: Path,
    *,
    expected_dataset_id: str,
) -> tuple[Gr3NormalizationStats, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != expected_dataset_id:
        raise ValueError("normalization dataset_id does not match training manifest")
    stats = Gr3NormalizationStats.from_dict(payload["normalization"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return stats, {"path": str(path.resolve()), "sha256": digest}


def _collate(processor, batch: list[dict[str, Any]]) -> dict[str, Any]:
    pixels = processor(images=[sample["image"] for sample in batch], return_tensors="pt")["pixel_values"]
    return {
        "instructions": [sample["lang"] for sample in batch],
        "pixels": pixels[:, None],
        "state": torch.from_numpy(np.stack([sample["state"] for sample in batch])),
        "action": torch.from_numpy(np.stack([sample["action"] for sample in batch])),
        "mask": torch.from_numpy(np.stack([sample["action_mask"] for sample in batch])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dinov3-path", type=Path, required=True)
    parser.add_argument("--bert-path", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--decode-threads", type=int, default=1)
    parser.add_argument("--batch-cache-size", type=int, default=2)
    parser.add_argument("--preload-batches", action="store_true")
    parser.add_argument("--normalization-json", type=Path)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--action-frequency-hz", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--train-vision-encoder", action="store_true")
    parser.add_argument("--train-text-encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TurboVLA GR3 training requires CUDA")
    if (
        args.max_steps < 1
        or args.batch_size < 1
        or args.gradient_accumulation_steps < 1
        or args.num_workers < 0
        or args.decode_threads < 1
        or args.batch_cache_size < 1
        or args.save_every < 1
    ):
        raise ValueError("training steps and batch sizes must be positive")
    for path, name in ((args.dinov3_path, "DINOv3"), (args.bert_path, "BERT")):
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"{name} config not found: {path / 'config.json'}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to use non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_profile = json.loads(args.dataset_manifest.read_text(encoding="utf-8")).get("profile_id")
    manifest_dataset_id = json.loads(
        args.dataset_manifest.read_text(encoding="utf-8")
    )["dataset_id"]
    normalization = None
    normalization_source = None
    if args.normalization_json is not None:
        normalization, normalization_source = _load_normalization(
            args.normalization_json,
            expected_dataset_id=manifest_dataset_id,
        )
    if manifest_profile == GR3_ANYGRASP_PROFILE_ID:
        if args.num_workers > 0 and args.decode_threads > 1:
            raise ValueError(
                "use either DataLoader workers or in-process decode threads, not both"
            )
        dataset = Gr3AnygraspDataset(
            args.dataset_manifest,
            horizon=args.horizon,
            image_size=args.image_size,
            stats=normalization,
            decode_threads=args.decode_threads,
            batch_cache_size=args.batch_cache_size,
        )
        if args.preload_batches:
            preloaded = dataset.preload_batches()
            print(f"preloaded_batches={preloaded}", flush=True)
    elif manifest_profile == GR3_PROFILE_ID:
        dataset = Gr3DaggerDataset(
            args.dataset_manifest,
            horizon=args.horizon,
            action_frequency_hz=args.action_frequency_hz,
            image_size=args.image_size,
        )
    else:
        raise ValueError(f"unsupported TurboVLA GR3 dataset profile: {manifest_profile}")
    processor = AutoImageProcessor.from_pretrained(args.dinov3_path, local_files_only=True)
    config = TurboVLAConfig(
        text=TextEncoderConfig(
            model_name_or_path=str(args.bert_path),
            frozen=not args.train_text_encoder,
            local_files_only=True,
        ),
        vision=VisionEncoderConfig(
            model_name_or_path=str(args.dinov3_path),
            image_size=args.image_size,
            num_views=1,
            position_embedding="view",
            encode_views_separately=True,
            frozen=not args.train_vision_encoder,
            local_files_only=True,
            compute_precision="bf16_autocast",
        ),
        interaction=InteractionConfig(compute_precision="bf16_autocast", attention_backend="sdpa"),
        action=ActionHeadConfig(
            action_dim=GR3_MODEL_ACTION_DIM,
            state_dim=GR3_MODEL_STATE_DIM,
            horizon=args.horizon,
        ),
    )
    model = build_turbovla(config)
    init_result = None
    if args.init_checkpoint is not None:
        loaded, skipped = _load_compatible(model, args.init_checkpoint)
        init_result = {"path": str(args.init_checkpoint.resolve()), "loaded": loaded, "skipped": skipped}
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=1e-10)
    loader_options: dict[str, Any] = {}
    if args.num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: _collate(processor, batch),
        **loader_options,
    )
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    last_loss = None
    started_at = time.monotonic()
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        pixels = batch["pixels"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        target = batch["action"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        precision = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with precision if device.type == "cuda" else nullcontext():
            prediction = model(batch["instructions"], {"dinov3": pixels}, state)
            elementwise = torch.abs(prediction - target)
            loss = (elementwise * mask.unsqueeze(-1)).sum() / (
                mask.sum().clamp_min(1.0) * GR3_MODEL_ACTION_DIM
            )
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        if step % args.gradient_accumulation_steps == 0 or step == args.max_steps:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        last_loss = float(loss.detach().cpu()) * args.gradient_accumulation_steps
        if step == 1 or step % 10 == 0:
            elapsed_seconds = time.monotonic() - started_at
            print(
                f"step={step} loss={last_loss:.6f} "
                f"elapsed_seconds={elapsed_seconds:.1f} "
                f"steps_per_second={step / max(elapsed_seconds, 1e-6):.4f}",
                flush=True,
            )
        if step % args.save_every == 0 and step != args.max_steps:
            torch.save(model.state_dict(), args.output_dir / f"model_step_{step}.pt")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": config.to_dict(),
        "normalization": dataset.stats.to_dict(),
        "normalization_source": normalization_source,
        "dataset_id": dataset.manifest["dataset_id"],
        "profile_id": dataset.manifest["profile_id"],
        "action_frequency_hz": args.action_frequency_hz,
        "model_action_dim": GR3_MODEL_ACTION_DIM,
        "model_state_dim": GR3_MODEL_STATE_DIM,
        "canonical_action_dim": GR3_ACTION_DIM,
        "last_loss": last_loss,
        "init": init_result,
        "diagnostics": dataset.normalization_diagnostics(),
    }
    torch.save(checkpoint, args.output_dir / "model_final.pt")
    (args.output_dir / "training.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset.manifest["dataset_id"],
                "profile_id": dataset.manifest["profile_id"],
                "samples": len(dataset),
                "max_steps": args.max_steps,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": (
                    args.batch_size * args.gradient_accumulation_steps
                ),
                "sample_exposures": args.batch_size * args.max_steps,
                "approximate_dataset_epochs": (
                    args.batch_size * args.max_steps / len(dataset)
                ),
                "save_every": args.save_every,
                "seed": args.seed,
                "last_loss": last_loss,
                "model_action_dim": GR3_MODEL_ACTION_DIM,
                "model_state_dim": GR3_MODEL_STATE_DIM,
                "canonical_action_dim": GR3_ACTION_DIM,
                "num_workers": args.num_workers,
                "decode_threads": args.decode_threads,
                "batch_cache_size": args.batch_cache_size,
                "preload_batches": args.preload_batches,
                "elapsed_seconds": time.monotonic() - started_at,
                "model_config": config.to_dict(),
                "normalization": dataset.stats.to_dict(),
                "normalization_source": normalization_source,
                "diagnostics": dataset.normalization_diagnostics(),
                "init": init_result,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
