"""CPU-only GR3 dataset preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from turbovla.data.gr3_anygrasp import GR3_ANYGRASP_PROFILE_ID, Gr3AnygraspDataset
from turbovla.data.gr3_dagger import GR3_PROFILE_ID, Gr3DaggerDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--action-frequency-hz", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile_id = json.loads(args.dataset_manifest.read_text(encoding="utf-8")).get("profile_id")
    if profile_id == GR3_ANYGRASP_PROFILE_ID:
        dataset = Gr3AnygraspDataset(
            args.dataset_manifest,
            horizon=args.horizon,
            image_size=args.image_size,
        )
    elif profile_id == GR3_PROFILE_ID:
        dataset = Gr3DaggerDataset(
            args.dataset_manifest,
            horizon=args.horizon,
            action_frequency_hz=args.action_frequency_hz,
            image_size=args.image_size,
        )
    else:
        raise ValueError(f"unsupported GR3 dataset profile: {profile_id}")
    sample = dataset[0]
    result = {
        "dataset_id": dataset.manifest["dataset_id"],
        "profile_id": dataset.manifest["profile_id"],
        "manifest_bindings": len(dataset.manifest["episodes"]),
        "samples": len(dataset),
        "image_shape": list(sample["image"].shape),
        "state_shape": list(sample["state"].shape),
        "action_shape": list(sample["action"].shape),
        "valid_action_steps_first_sample": int(sample["action_mask"].sum()),
        "diagnostics": dataset.normalization_diagnostics(),
        "normalization": dataset.stats.to_dict(),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        if args.output.exists():
            parser.error(f"refusing to overwrite output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
