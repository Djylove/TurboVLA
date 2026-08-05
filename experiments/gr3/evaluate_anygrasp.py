"""Offline held-out evaluation for a manifest-bound GR3 AnyGrasp checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor

from turbovla.data.gr3_anygrasp import Gr3AnygraspDataset
from turbovla.data.gr3_common import (
    GR3_ACTION_DIM,
    GR3_MODEL_ACTION_DIM,
    Gr3NormalizationStats,
)
from turbovla.models import TurboVLAConfig, build_turbovla

from .train import _checkpoint_state, _load_compatible


def _collate(processor, batch: list[dict[str, Any]]) -> dict[str, Any]:
    pixels = processor(images=[sample["image"] for sample in batch], return_tensors="pt")["pixel_values"]
    return {
        "instructions": [sample["lang"] for sample in batch],
        "pixels": pixels[:, None],
        "state": torch.from_numpy(np.stack([sample["state"] for sample in batch])),
        "action": torch.from_numpy(np.stack([sample["action"] for sample in batch])),
        "mask": torch.from_numpy(np.stack([sample["action_mask"] for sample in batch])),
        "task_ids": [sample["metadata"]["task_id"] for sample in batch],
    }


def _digest_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stratified_indices(
    dataset: Gr3AnygraspDataset,
    *,
    max_samples: int,
    seed: int,
) -> tuple[list[int], dict[str, int]]:
    """Select samples round-robin across tasks using stable SHA-256 ordering."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        grouped[sample.task_id].append(index)
    task_ids = sorted(grouped, key=lambda task_id: _digest_key(f"{seed}:task:{task_id}"))
    for task_id in task_ids:
        grouped[task_id].sort(key=lambda index: _digest_key(f"{seed}:{task_id}:{index}"))

    selected: list[int] = []
    per_task: dict[str, int] = {task_id: 0 for task_id in sorted(grouped)}
    depth = 0
    while len(selected) < max_samples:
        added = False
        for task_id in task_ids:
            if depth < len(grouped[task_id]):
                selected.append(grouped[task_id][depth])
                per_task[task_id] += 1
                added = True
                if len(selected) == max_samples:
                    break
        if not added:
            break
        depth += 1
    return selected, per_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initialization-checkpoint", type=Path)
    parser.add_argument("--model-state-checkpoint", type=Path)
    parser.add_argument("--dinov3-path", type=Path, required=True)
    parser.add_argument("--bert-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    args = parser.parse_args()
    if args.initialization_checkpoint is not None and args.model_state_checkpoint is not None:
        parser.error("initialization-checkpoint and model-state-checkpoint are mutually exclusive")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("GR3 AnyGrasp evaluation requires a CUDA-compatible device")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = Gr3NormalizationStats.from_dict(payload["normalization"])
    config = TurboVLAConfig.from_mapping(payload["model_config"])
    config.vision.model_name_or_path = str(args.dinov3_path.resolve())
    config.text.model_name_or_path = str(args.bert_path.resolve())
    config.vision.local_files_only = True
    config.text.local_files_only = True
    dataset = Gr3AnygraspDataset(
        args.dataset_manifest,
        horizon=config.action.horizon,
        image_size=config.vision.image_size,
        stats=stats,
        model_action_dim=config.action.action_dim,
        model_state_dim=config.action.state_dim,
    )
    if args.batch_size < 1 or args.max_batches < 1:
        parser.error("batch-size and max-batches must be positive")
    evaluation_indices, per_task_sample_count = _stratified_indices(
        dataset,
        max_samples=args.batch_size * args.max_batches,
        seed=args.sampling_seed,
    )
    processor = AutoImageProcessor.from_pretrained(args.dinov3_path, local_files_only=True)
    loader = DataLoader(
        Subset(dataset, evaluation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: _collate(processor, batch),
    )
    device = torch.device("cuda")
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    model = build_turbovla(config)
    init_result = None
    model_state_checkpoint = None
    if args.model_state_checkpoint is not None:
        model.load_state_dict(_checkpoint_state(args.model_state_checkpoint), strict=True)
        model_state_source = "intermediate_checkpoint"
        model_state_checkpoint = str(args.model_state_checkpoint.resolve())
    elif args.initialization_checkpoint is None:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model_state_source = "trained_checkpoint"
    else:
        loaded, skipped = _load_compatible(model, args.initialization_checkpoint)
        init_result = {
            "path": str(args.initialization_checkpoint.resolve()),
            "loaded": loaded,
            "skipped": skipped,
        }
        model_state_source = "released_shape_compatible_initialization"
    model.to(device, dtype=torch.bfloat16).eval().requires_grad_(False)

    total_absolute_error = 0.0
    total_valid_values = 0.0
    model_action_dim = int(config.action.action_dim)
    if model_action_dim not in {GR3_MODEL_ACTION_DIM, GR3_ACTION_DIM}:
        raise ValueError(f"unsupported GR3 model action dimension: {model_action_dim}")
    axis_absolute_error = torch.zeros(model_action_dim, dtype=torch.float64)
    axis_valid_steps = 0.0
    evaluated_samples = 0
    latencies: list[float] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            pixels = batch["pixels"].to(device, dtype=torch.bfloat16)
            state = batch["state"].to(device, dtype=torch.bfloat16)
            target = batch["action"].to(device)
            mask = batch["mask"].to(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = model(batch["instructions"], {"dinov3": pixels}, state).float()
            torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - started)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("held-out prediction contains NaN or Inf")
            error = (prediction - target).abs() * mask.unsqueeze(-1)
            total_absolute_error += float(error.sum().cpu())
            valid_steps = float(mask.sum().cpu())
            total_valid_values += valid_steps * model_action_dim
            axis_absolute_error += error.sum(dim=(0, 1)).double().cpu()
            axis_valid_steps += valid_steps
            evaluated_samples += int(prediction.shape[0])
    if evaluated_samples < 1 or total_valid_values <= 0:
        raise RuntimeError("held-out evaluation produced no valid samples")

    warm_latencies = latencies[1:] or latencies
    index_digest = hashlib.sha256(
        json.dumps(evaluation_indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    report = {
        "status": "passed",
        "dataset_id": dataset.manifest["dataset_id"],
        "profile_id": dataset.manifest["profile_id"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_dataset_id": payload["dataset_id"],
        "model_state_source": model_state_source,
        "model_state_checkpoint": model_state_checkpoint,
        "model_seed": args.model_seed,
        "initialization": init_result,
        "evaluated_samples": evaluated_samples,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "sampling_strategy": "task_round_robin_sha256",
        "sampling_seed": args.sampling_seed,
        "evaluation_index_sha256": index_digest,
        "evaluation_task_count": sum(count > 0 for count in per_task_sample_count.values()),
        "per_task_sample_count": per_task_sample_count,
        "normalized_l1": total_absolute_error / total_valid_values,
        "per_axis_normalized_l1": (axis_absolute_error / axis_valid_steps).tolist(),
        "model_action_dim": model_action_dim,
        "canonical_action_dim": GR3_ACTION_DIM,
        "mean_inference_latency_ms": 1000.0 * sum(latencies) / len(latencies),
        "p95_inference_latency_ms": 1000.0 * float(np.quantile(latencies, 0.95)),
        "cold_start_inference_latency_ms": 1000.0 * latencies[0],
        "warm_mean_inference_latency_ms": 1000.0 * sum(warm_latencies) / len(warm_latencies),
        "warm_p95_inference_latency_ms": 1000.0 * float(np.quantile(warm_latencies, 0.95)),
        "output_shape": [config.action.horizon, config.action.action_dim],
        "normalization_source": "training_checkpoint",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
