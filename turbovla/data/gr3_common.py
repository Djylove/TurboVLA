"""Shared numerical and image contracts for GR3 policy datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

GR3_STATE_DIM = 33
GR3_ACTION_DIM = 37
GR3_MODEL_STATE_DIM = 31
GR3_MODEL_ACTION_DIM = 31
GR3_HAND_ACTION_INDICES = tuple(range(19, 31))
GR3_LEFT_FINGER_ACTION_INDICES = (19, 20, 21, 22)


def canonicalize_gr3_action(value: np.ndarray) -> np.ndarray:
    """Expand a learned 31-joint chunk to the external canonical 37D contract."""

    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] == GR3_ACTION_DIM:
        return array.copy()
    if array.shape[-1] != GR3_MODEL_ACTION_DIM:
        raise ValueError("GR3 learned action must end in 31 dimensions")
    canonical = np.zeros((*array.shape[:-1], GR3_ACTION_DIM), dtype=np.float32)
    canonical[..., :GR3_MODEL_ACTION_DIM] = array
    return canonical


def prepare_gr3_rgb(image: np.ndarray, image_size: int, *, input_bgr: bool = False) -> np.ndarray:
    """Center-crop and resize a GR3 top-camera frame."""

    color = np.asarray(image)
    if color.ndim != 3 or color.shape[-1] != 3 or color.dtype != np.uint8:
        raise ValueError(f"GR3 image must be uint8 HxWx3, got {color.dtype} {color.shape}")
    if input_bgr:
        color = np.ascontiguousarray(color[..., ::-1])
    height, width = color.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    cropped = color[y0 : y0 + side, x0 : x0 + side]
    resized = Image.fromarray(cropped).resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.uint8, copy=True)


@dataclass(frozen=True)
class Gr3NormalizationStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray
    state_std_floor: float = 0.0
    state_clip_z: float | None = None

    def __post_init__(self) -> None:
        expected = {
            "state_mean": (self.state_mean, GR3_STATE_DIM),
            "state_std": (self.state_std, GR3_STATE_DIM),
            "action_low": (self.action_low, GR3_ACTION_DIM),
            "action_high": (self.action_high, GR3_ACTION_DIM),
        }
        for name, (value, dim) in expected.items():
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (dim,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite {dim}D vector")
            object.__setattr__(self, name, array)
        if np.any(self.state_std <= 0):
            raise ValueError("state_std must be positive")
        if np.any(self.action_high <= self.action_low):
            raise ValueError("action bounds must be strictly increasing")
        if not np.isfinite(self.state_std_floor) or self.state_std_floor < 0:
            raise ValueError("state_std_floor must be finite and non-negative")
        if self.state_clip_z is not None and (
            not np.isfinite(self.state_clip_z) or self.state_clip_z <= 0
        ):
            raise ValueError("state_clip_z must be finite and positive when set")

    def normalize_state(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.shape[-1] not in {GR3_MODEL_STATE_DIM, GR3_STATE_DIM}:
            raise ValueError(
                "GR3 state must end in the 31D model or 33D canonical dimension"
            )
        dim = value.shape[-1]
        scale = np.maximum(self.state_std[:dim], self.state_std_floor)
        normalized = (value - self.state_mean[:dim]) / scale
        if self.state_clip_z is not None:
            normalized = np.clip(normalized, -self.state_clip_z, self.state_clip_z)
        return normalized.astype(np.float32)

    def normalize_action(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.shape[-1] not in {GR3_MODEL_ACTION_DIM, GR3_ACTION_DIM}:
            raise ValueError(
                "GR3 action must end in the 31D model or 37D canonical dimension"
            )
        dim = value.shape[-1]
        scaled = (
            2.0
            * (value - self.action_low[:dim])
            / (self.action_high[:dim] - self.action_low[:dim])
            - 1.0
        )
        return np.clip(scaled, -1.0, 1.0).astype(np.float32)

    def denormalize_action(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.shape[-1] not in {GR3_MODEL_ACTION_DIM, GR3_ACTION_DIM}:
            raise ValueError(
                "GR3 action must end in the 31D model or 37D canonical dimension"
            )
        dim = value.shape[-1]
        clipped = np.clip(value, -1.0, 1.0)
        return (
            0.5
            * (clipped + 1.0)
            * (self.action_high[:dim] - self.action_low[:dim])
            + self.action_low[:dim]
        ).astype(np.float32)

    def with_state_robustness(
        self,
        *,
        std_floor: float,
        clip_z: float | None,
    ) -> "Gr3NormalizationStats":
        """Return the same dataset statistics with a robust state transform."""

        return Gr3NormalizationStats(
            state_mean=self.state_mean,
            state_std=self.state_std,
            action_low=self.action_low,
            action_high=self.action_high,
            state_std_floor=std_floor,
            state_clip_z=clip_z,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "state_std_floor": self.state_std_floor,
            "state_clip_z": self.state_clip_z,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Gr3NormalizationStats":
        return cls(
            state_mean=np.asarray(value["state_mean"], dtype=np.float32),
            state_std=np.asarray(value["state_std"], dtype=np.float32),
            action_low=np.asarray(value["action_low"], dtype=np.float32),
            action_high=np.asarray(value["action_high"], dtype=np.float32),
            state_std_floor=float(value.get("state_std_floor", 0.0)),
            state_clip_z=(
                None
                if value.get("state_clip_z") is None
                else float(value["state_clip_z"])
            ),
        )


def gr3_active_hand_action_indices(
    stats: Gr3NormalizationStats,
    *,
    min_range: float = 0.05,
    model_action_dim: int = GR3_MODEL_ACTION_DIM,
) -> tuple[int, ...]:
    """Select hand axes that contain a learnable action range in this dataset."""

    if min_range < 0 or model_action_dim not in {GR3_MODEL_ACTION_DIM, GR3_ACTION_DIM}:
        raise ValueError("invalid hand range threshold or model action dimension")
    action_range = stats.action_high - stats.action_low
    return tuple(
        index
        for index in GR3_HAND_ACTION_INDICES
        if index < model_action_dim and float(action_range[index]) >= min_range
    )


def gr3_left_hand_closed(
    actions: np.ndarray,
    *,
    threshold: float = -0.5,
) -> np.ndarray:
    """Classify GR3 left-finger closure from raw joint-space actions."""

    values = np.asarray(actions, dtype=np.float32)
    if values.shape[-1] <= max(GR3_LEFT_FINGER_ACTION_INDICES):
        raise ValueError("GR3 action does not contain the left finger axes")
    aperture = values[..., list(GR3_LEFT_FINGER_ACTION_INDICES)].mean(axis=-1)
    return aperture < threshold
