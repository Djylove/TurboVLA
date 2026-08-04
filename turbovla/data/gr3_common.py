"""Shared numerical and image contracts for GR3 policy datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

GR3_STATE_DIM = 33
GR3_ACTION_DIM = 37


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

    def normalize_state(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.state_mean) / self.state_std).astype(np.float32)

    def normalize_action(self, value: np.ndarray) -> np.ndarray:
        scaled = 2.0 * (value - self.action_low) / (self.action_high - self.action_low) - 1.0
        return np.clip(scaled, -1.0, 1.0).astype(np.float32)

    def denormalize_action(self, value: np.ndarray) -> np.ndarray:
        clipped = np.clip(value, -1.0, 1.0)
        return (0.5 * (clipped + 1.0) * (self.action_high - self.action_low) + self.action_low).astype(
            np.float32
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Gr3NormalizationStats":
        return cls(**{name: np.asarray(value[name], dtype=np.float32) for name in cls.__dataclass_fields__})
