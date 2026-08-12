"""Inference wrapper for TurboVLA checkpoints trained on the GR3 profile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor

from turbovla.data.gr3_anygrasp import GR3_ANYGRASP_PROFILE_ID
from turbovla.data.gr3_common import (
    GR3_ACTION_DIM,
    GR3_MODEL_ACTION_DIM,
    GR3_MODEL_STATE_DIM,
    Gr3NormalizationStats,
    canonicalize_gr3_action,
    prepare_gr3_rgb,
)
from turbovla.data.gr3_dagger import GR3_PROFILE_ID
from turbovla.models import TurboVLAConfig, build_turbovla


def _load_gr3_checkpoint(path: Path) -> tuple[dict, Path | None]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload, None
    if not (
        isinstance(payload, dict)
        and payload
        and all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items())
    ):
        raise ValueError(f"unsupported TurboVLA GR3 checkpoint payload: {path}")

    metadata_path = path.with_name("model_final.pt")
    if metadata_path == path or not metadata_path.is_file():
        raise ValueError(
            "raw TurboVLA GR3 weights require model_final.pt in the same directory"
        )
    metadata = torch.load(metadata_path, map_location="cpu")
    if not isinstance(metadata, dict) or "model_state_dict" not in metadata:
        raise ValueError(f"invalid TurboVLA GR3 metadata checkpoint: {metadata_path}")
    metadata["model_state_dict"] = payload
    return metadata, metadata_path


class TurboVLAGr3Policy:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        dinov3_path: str | Path | None = None,
        bert_path: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        payload, self.metadata_checkpoint_path = _load_gr3_checkpoint(
            self.checkpoint_path
        )
        if payload.get("profile_id") not in {GR3_PROFILE_ID, GR3_ANYGRASP_PROFILE_ID}:
            raise ValueError("checkpoint is not a TurboVLA GR3 profile")
        config = TurboVLAConfig.from_mapping(payload["model_config"])
        if (
            config.action.state_dim != GR3_MODEL_STATE_DIM
            or config.action.action_dim != GR3_MODEL_ACTION_DIM
        ):
            raise ValueError("checkpoint does not use the GR3 31D joint contract")
        if config.vision.num_views != 1:
            raise ValueError("TurboVLA GR3 checkpoint must use one camera view")
        if dinov3_path is not None:
            config.vision.model_name_or_path = str(Path(dinov3_path).expanduser().resolve())
        if bert_path is not None:
            config.text.model_name_or_path = str(Path(bert_path).expanduser().resolve())
        config.vision.local_files_only = True
        config.text.local_files_only = True
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TurboVLA GR3 inference requested CUDA but it is unavailable")
        self.model = build_turbovla(config)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.processor = AutoImageProcessor.from_pretrained(config.vision.model_name_or_path, local_files_only=True)
        self.stats = Gr3NormalizationStats.from_dict(payload["normalization"])
        self.action_frequency_hz = float(payload.get("action_frequency_hz", 30.0))
        self.image_size = int(config.vision.image_size)

    @torch.inference_mode()
    def predict(self, image_rgb: np.ndarray, instruction: str, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (GR3_MODEL_STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError("TurboVLA GR3 state must be a finite 31D joint vector")
        image = prepare_gr3_rgb(image_rgb, self.image_size)
        pixels = self.processor(images=[image], return_tensors="pt")["pixel_values"]
        pixels = pixels[:, None].to(self.device, dtype=torch.bfloat16)
        normalized_state = torch.from_numpy(self.stats.normalize_state(state))[None].to(
            self.device, dtype=torch.bfloat16
        )
        prediction = self.model([str(instruction)], {"dinov3": pixels}, normalized_state)[0]
        predicted_values = self.stats.denormalize_action(
            prediction.float().cpu().numpy()
        )
        values = canonicalize_gr3_action(predicted_values)
        if values.ndim != 2 or values.shape[1] != GR3_ACTION_DIM:
            raise ValueError(f"TurboVLA GR3 produced invalid action shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("TurboVLA GR3 produced NaN or Inf")
        return values
