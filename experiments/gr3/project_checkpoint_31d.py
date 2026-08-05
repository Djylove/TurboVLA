"""Project a legacy GR3 33D state/action checkpoint to the 31-joint contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from turbovla.data.gr3_common import GR3_MODEL_ACTION_DIM, GR3_MODEL_STATE_DIM


STATE_SLICES = {
    "action_head.state_projection.net.0.weight": (0, 33),
    "action_head.state_projection.net.0.bias": (0, 33),
    "action_head.state_projection.net.1.weight": (1, 33),
}
ACTION_SLICES = {
    "action_head.decoder.action_projection.layers.2.weight": (0, 33),
    "action_head.decoder.action_projection.layers.2.bias": (0, 33),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slice_axis(value: torch.Tensor, axis: int, source_dim: int, target_dim: int) -> torch.Tensor:
    if value.shape[axis] != source_dim:
        raise ValueError(
            f"expected source dimension {source_dim} on axis {axis}, got {tuple(value.shape)}"
        )
    index = [slice(None)] * value.ndim
    index[axis] = slice(0, target_dim)
    return value[tuple(index)].clone().contiguous()


def project_payload(payload: dict) -> dict:
    config = json.loads(json.dumps(payload["model_config"]))
    action_config = config["action"]
    if int(action_config["state_dim"]) != 33 or int(action_config["action_dim"]) != 33:
        raise ValueError("source checkpoint must use the legacy GR3 33D/33D head")

    state = dict(payload["model_state_dict"])
    for key, (axis, source_dim) in STATE_SLICES.items():
        state[key] = _slice_axis(
            state[key], axis, source_dim, GR3_MODEL_STATE_DIM
        )
    for key, (axis, source_dim) in ACTION_SLICES.items():
        state[key] = _slice_axis(
            state[key], axis, source_dim, GR3_MODEL_ACTION_DIM
        )

    action_config["state_dim"] = GR3_MODEL_STATE_DIM
    action_config["action_dim"] = GR3_MODEL_ACTION_DIM
    result = dict(payload)
    result["model_state_dict"] = state
    result["model_config"] = config
    result["model_state_dim"] = GR3_MODEL_STATE_DIM
    result["model_action_dim"] = GR3_MODEL_ACTION_DIM
    result["projection"] = {
        "type": "gr3_joint31_from_legacy33",
        "source_state_dim": 33,
        "source_action_dim": 33,
        "target_state_dim": GR3_MODEL_STATE_DIM,
        "target_action_dim": GR3_MODEL_ACTION_DIM,
        "dropped_state_axes": ["base_height", "base_pitch"],
        "dropped_action_axes": ["vel_height", "vel_pitch"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        parser.error(f"source checkpoint does not exist: {source}")
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    projected = project_payload(payload)
    projected["projection"]["source_checkpoint"] = str(source)
    projected["projection"]["source_sha256"] = _sha256(source)
    torch.save(projected, output)
    report = {
        **projected["projection"],
        "output_checkpoint": str(output),
        "output_sha256": _sha256(output),
    }
    output.with_name("projection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
