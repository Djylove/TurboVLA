"""GR3-only reader for XPolicyLab DAgger v2 dataset manifests.

This module intentionally does not define a generic DAgger format. It consumes
only the ``xpolicylab.gr3_dagger_v2`` profile and produces the one-camera,
33-state, 37-action samples used by the TurboVLA GR3 head.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .gr3_common import (
    GR3_ACTION_DIM,
    GR3_STATE_DIM,
    Gr3NormalizationStats,
    prepare_gr3_rgb,
)

GR3_PROFILE_ID = "xpolicylab.gr3_dagger_v2"
GR3_SOURCE_SCHEMA = "gr3_dagger_v2"
DEFAULT_ALIGNMENT_LIMIT_MS = 20.0


def _nearest_index(timestamps: Sequence[int], target: int) -> tuple[int, float]:
    if not timestamps:
        raise ValueError("cannot align against an empty timestamp stream")
    index = bisect.bisect_left(timestamps, target)
    candidates = [item for item in (index - 1, index) if 0 <= item < len(timestamps)]
    nearest = min(candidates, key=lambda item: abs(int(timestamps[item]) - target))
    return nearest, abs(int(timestamps[nearest]) - target) / 1e6


def _named_vector(value: Any, expected_names: Sequence[str], *, stream: str) -> np.ndarray:
    if not isinstance(value, list):
        raise TypeError(f"{stream} row must be a named-vector list")
    names = [str(item["name"]) for item in value]
    if names != list(expected_names):
        raise ValueError(f"{stream} vector order does not match schema.json")
    vector = np.asarray([item["value"] for item in value], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError(f"{stream} contains NaN or Inf")
    return vector


def load_gr3_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("GR3 dataset manifest cannot be a symlink")
    source = source.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "xpolicy_dataset.v1":
        raise ValueError("unsupported XPolicyLab dataset manifest version")
    if payload.get("profile_id") != GR3_PROFILE_ID:
        raise ValueError(f"TurboVLA GR3 requires profile_id={GR3_PROFILE_ID}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("GR3 dataset manifest requires episodes")
    eligible = [episode for episode in episodes if episode.get("train_eligible_after_filters") is True]
    if not eligible:
        raise ValueError("GR3 dataset manifest has no eligible episodes")
    for episode in eligible:
        if episode.get("profile_id") != GR3_PROFILE_ID:
            raise ValueError("episode profile does not match GR3 dataset profile")
        if episode.get("source_schema") != GR3_SOURCE_SCHEMA:
            raise ValueError("episode source schema is not gr3_dagger_v2")
    return {**payload, "episodes": eligible}


class _Episode:
    def __init__(self, episode: dict[str, Any]) -> None:
        import pyarrow.parquet as pq

        self.path = Path(episode["path"]).expanduser().resolve()
        schema = json.loads((self.path / "schema.json").read_text(encoding="utf-8"))
        if (
            schema.get("schema_version") != GR3_SOURCE_SCHEMA
            or schema.get("state_dim") != GR3_STATE_DIM
            or schema.get("action_dim") != GR3_ACTION_DIM
        ):
            raise ValueError(f"unsupported GR3 schema: {self.path}")
        self.state_names = tuple(schema["state_order"])
        self.action_names = tuple(schema["action_order"])
        self.width = int(schema["camera_top"]["width"])
        self.height = int(schema["camera_top"]["height"])
        if schema["camera_top"].get("encoding") != "bgr8":
            raise ValueError("TurboVLA GR3 requires raw bgr8 camera frames")

        camera_name = "observation.images.camera_top"
        state_name = "observation.state_robot"
        action_name = "action.expert_safe_robot"
        self.camera = pq.read_table(
            self.path / f"{camera_name}.parquet",
            columns=["source_timestamp_ns", camera_name],
        )
        state = pq.read_table(
            self.path / f"{state_name}.parquet",
            columns=["source_timestamp_ns", state_name],
        )
        action = pq.read_table(
            self.path / f"{action_name}.parquet",
            columns=["source_timestamp_ns", action_name],
        )
        self.camera_ts = [int(value.as_py()) for value in self.camera["source_timestamp_ns"]]
        self.state_ts = [int(value.as_py()) for value in state["source_timestamp_ns"]]
        self.action_ts = [int(value.as_py()) for value in action["source_timestamp_ns"]]
        self.states = np.stack(
            [_named_vector(value.as_py(), self.state_names, stream=state_name) for value in state[state_name]]
        )
        self.actions = np.stack(
            [_named_vector(value.as_py(), self.action_names, stream=action_name) for value in action[action_name]]
        )
        self.instruction = str(episode.get("task_instruction", ""))

    def rgb(self, camera_index: int, image_size: int) -> np.ndarray:
        name = "observation.images.camera_top"
        bgr = np.asarray(self.camera[name][camera_index].as_py(), dtype=np.uint8).reshape(self.height, self.width, 3)
        return prepare_gr3_rgb(bgr, image_size, input_bgr=True)


@dataclass(frozen=True)
class _Sample:
    episode: int
    camera: int
    state: int
    actions: tuple[int, ...]
    mask: tuple[float, ...]


class Gr3DaggerDataset:
    """Timestamp-aligned GR3 samples resampled at the policy action frequency."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        horizon: int = 50,
        action_frequency_hz: float = 30.0,
        image_size: int = 224,
        alignment_limit_ms: float = DEFAULT_ALIGNMENT_LIMIT_MS,
        stats: Gr3NormalizationStats | None = None,
    ) -> None:
        if horizon < 1 or action_frequency_hz <= 0 or image_size < 16:
            raise ValueError("horizon/frequency/image_size must be positive")
        self.manifest = load_gr3_manifest(manifest_path)
        self.horizon = int(horizon)
        self.action_frequency_hz = float(action_frequency_hz)
        self.image_size = int(image_size)
        self.alignment_limit_ms = float(alignment_limit_ms)
        self.episodes = [_Episode(episode) for episode in self.manifest["episodes"]]
        self.samples: list[_Sample] = []
        period_ns = int(round(1e9 / self.action_frequency_hz))
        for episode_index, episode in enumerate(self.episodes):
            for camera_index, timestamp in enumerate(episode.camera_ts):
                if timestamp > episode.state_ts[-1]:
                    continue
                state_index, state_error = _nearest_index(episode.state_ts, timestamp)
                if state_error > self.alignment_limit_ms:
                    continue
                action_indices: list[int] = []
                mask: list[float] = []
                for step in range(self.horizon):
                    action_index, action_error = _nearest_index(episode.action_ts, timestamp + step * period_ns)
                    valid = action_error <= self.alignment_limit_ms
                    action_indices.append(action_index)
                    mask.append(float(valid))
                if mask[0] == 1.0:
                    self.samples.append(
                        _Sample(
                            episode=episode_index,
                            camera=camera_index,
                            state=state_index,
                            actions=tuple(action_indices),
                            mask=tuple(mask),
                        )
                    )
        if not self.samples:
            raise ValueError("GR3 dataset contains no aligned training samples")
        self.stats = stats or self.compute_stats()

    def compute_stats(self) -> Gr3NormalizationStats:
        states = np.stack([self.episodes[sample.episode].states[sample.state] for sample in self.samples])
        actions = np.concatenate(
            [
                self.episodes[sample.episode].actions[
                    [index for index, valid in zip(sample.actions, sample.mask) if valid]
                ]
                for sample in self.samples
            ],
            axis=0,
        )
        state_std = np.maximum(states.std(axis=0), 1e-4)
        action_low = np.quantile(actions, 0.01, axis=0).astype(np.float32)
        action_high = np.quantile(actions, 0.99, axis=0).astype(np.float32)
        stagnant = action_high - action_low < 1e-5
        action_low[stagnant] -= 1e-3
        action_high[stagnant] += 1e-3
        return Gr3NormalizationStats(
            state_mean=states.mean(axis=0),
            state_std=state_std,
            action_low=action_low,
            action_high=action_high,
        )

    def normalization_diagnostics(self) -> dict[str, list[str]]:
        state_names = self.episodes[0].state_names
        action_names = self.episodes[0].action_names
        return {
            "near_constant_state_axes": [
                name for name, std in zip(state_names, self.stats.state_std, strict=True) if std <= 1.01e-4
            ],
            "near_constant_action_axes": [
                name
                for name, low, high in zip(
                    action_names,
                    self.stats.action_low,
                    self.stats.action_high,
                    strict=True,
                )
                if high - low <= 2.01e-3
            ],
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        episode = self.episodes[sample.episode]
        actions = episode.actions[list(sample.actions)]
        return {
            "image": episode.rgb(sample.camera, self.image_size),
            "lang": episode.instruction,
            "state": self.stats.normalize_state(episode.states[sample.state]),
            "action": self.stats.normalize_action(actions),
            "action_mask": np.asarray(sample.mask, dtype=np.float32),
            "metadata": {
                "episode": str(episode.path),
                "camera_index": sample.camera,
                "source_timestamp_ns": episode.camera_ts[sample.camera],
            },
        }
