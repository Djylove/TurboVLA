"""Interactive Training bound single-GPU RoboTwin baseline training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROFILE_ID = "turbovla.robotwin_clean_v1"
SUPPORTED_TASKS = {"beat_block_hammer"}


def _bound_dataset(manifest_path: Path) -> tuple[Path, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("profile_id") != PROFILE_ID:
        raise ValueError(f"expected dataset profile {PROFILE_ID}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise ValueError("local RoboTwin baseline requires exactly one task")
    episode = episodes[0]
    if not episode.get("train_eligible_after_filters"):
        raise ValueError("RoboTwin task is not training eligible")
    task_name = str(episode.get("task_id", ""))
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported local RoboTwin task: {task_name}")
    task_path = Path(episode["path"]).expanduser().resolve()
    if task_path.name != task_name or task_path.parent.name != "Clean":
        raise ValueError("RoboTwin task must be stored below <root>/Clean/<task>")
    return task_path.parent.parent, task_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dinov3-path", type=Path, required=True)
    parser.add_argument("--bert-path", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("steps, batch size, and learning rate must be positive")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite output: {args.output_dir}")
    for path, label in (
        (args.dinov3_path / "config.json", "DINOv3"),
        (args.bert_path / "config.json", "BERT"),
        (args.pretrained_checkpoint, "TurboVLA checkpoint"),
    ):
        if path.is_symlink() or not path.is_file():
            parser.error(f"missing regular {label} asset: {path}")

    data_root, task_name = _bound_dataset(args.dataset_manifest.resolve())
    os.environ.update(
        {
            "ROBOTWIN_DATA_ROOT": str(data_root),
            "DINOV3_MODEL_PATH": str(args.dinov3_path.resolve()),
            "BERT_MODEL_PATH": str(args.bert_path.resolve()),
            # Kept resolvable by the upstream config; initialization itself is
            # disabled because the released full TurboVLA checkpoint is loaded.
            "TURBOVLA_INIT_CKPT": str(args.pretrained_checkpoint.resolve()),
            "WANDB_MODE": "disabled",
        }
    )
    mixture = f"robotwin_clean_{task_name}"
    config = Path(__file__).with_name("configs") / "clean50.yaml"
    sys.argv = [
        "interactive_train",
        "--config_yaml",
        str(config),
        f"--run_root_dir={args.output_dir.parent.resolve()}",
        f"--run_id={args.output_dir.name}",
        f"--seed={args.seed}",
        "--trainer.use_deepspeed=false",
        f"--trainer.max_train_steps={args.max_steps}",
        "--trainer.num_warmup_steps=0",
        f"--trainer.save_interval={args.max_steps}",
        "--trainer.eval_interval=1000000",
        "--trainer.logging_frequency=1",
        "--trainer.ema_device=cpu",
        f"--trainer.learning_rate.base={args.learning_rate}",
        f"--trainer.learning_rate.text_encoder={args.learning_rate}",
        f"--trainer.learning_rate.vision_encoder={args.learning_rate}",
        f"--trainer.learning_rate.vision_language_interaction={args.learning_rate}",
        f"--trainer.learning_rate.vision_projection={args.learning_rate}",
        f"--trainer.learning_rate.action_head={args.learning_rate}",
        f"--trainer.pretrained_checkpoint={args.pretrained_checkpoint.resolve()}",
        "--framework.initialization.load_pretrained=false",
        "--framework.text.attn_implementation=null",
        "--framework.vision.attn_implementation=null",
        "--framework.vision.freeze_vision_encoder=true",
        f"--datasets.vla_data.data_root_dir={data_root}",
        f"--datasets.vla_data.data_mix={mixture}",
        f"--datasets.vla_data.per_device_batch_size={args.batch_size}",
        "--datasets.vla_data.num_workers=0",
        "--datasets.vla_data.persistent_workers=false",
        "--datasets.vla_data.pin_memory=false",
    ]
    from starVLA.training.train_robotwin_clean_act_pi05_recipe import main as train

    train()
    final_model = args.output_dir / "final_model" / "pytorch_model.pt"
    if not final_model.is_file():
        raise RuntimeError(f"RoboTwin training did not produce {final_model}")


if __name__ == "__main__":
    main()
