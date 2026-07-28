from dataclasses import dataclass, field
import os
import time
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.ACT_ActionHeader import ACTActionHead, get_action_model as get_act_action_model
from starVLA.model.modules.groundingdino_vla import (
    DINOv3Backbone,
    GroundingDINOFeatureEnhancer,
    GroundingDINOTextEncoder,
    VisionProjector,
    cuda_autocast,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class GroundingDINODiTDefaultConfig:
    name: str = "GroundingDINODiT"

    text: dict = field(
        default_factory=lambda: {
            "bert_path": "/path/to/bert-base-uncased",
            "max_text_len": 256,
            "sub_sentence_present": True,
            "local_files_only": True,
            "freeze_text_encoder": True,
            "attn_implementation": "flash_attention_2",
        }
    )

    dinov3: dict = field(
        default_factory=lambda: {
            "model_path": "/path/to/dinov3",
            "image_size": 224,
            "num_views": 2,
            "local_files_only": True,
            "freeze_vision_encoder": True,
            "attn_implementation": "flash_attention_2",
            "vision_pos_init_std": 0.01,
            "vision_pos_scale_init": 0.01,
            "vision_dropout": 0.1,
        }
    )

    fusion: dict = field(
        default_factory=lambda: {
            "hidden_dim": 256,
            "nheads": 8,
            "dim_feedforward": 2048,
            "enhancer_inner_dim": 1024,
            "num_layers": 6,
            "text_dropout": 0.0,
            "fusion_dropout": 0.0,
            "fusion_droppath": 0.1,
        }
    )

    groundingdino: dict = field(
        default_factory=lambda: {
            "pretrained_ckpt": "/path/to/groundingdino_swint_ogc.pth",
            "load_pretrained": True,
            "load_bert": True,
            "load_text_proj": True,
            "load_feature_enhancer": True,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "act",
            "action_dim": 14,
            "state_dim": 14,
            "action_horizon": 50,
            "act_hidden_dim": 256,
            "act_nheads": 8,
            "act_num_layers": 3,
            "act_dim_feedforward": 2048,
            "act_dropout": 0.1,
            "act_state_tokens": 2,
            "act_state_hidden_dim": 256,
            "act_mlp_hidden_dim": 512,
            "act_loss_type": "l1",
        }
    )


@FRAMEWORK_REGISTRY.register("GroundingDINODiT")
class Grounding_DINO_DiT(baseframework):
    """
    DINOv3 + GroundingDINO BERT/feature-enhancer + state-conditioned ACT.

    The framework consumes the standard StarVLA LeRobot batch schema:
      - image: list of three PIL images or arrays
      - lang: instruction string
      - state: optional [1, state_dim] proprioception
      - action: [T, action_dim] normalized action chunk
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self._enable_torch_flash_sdp()
        self.config = merge_framework_config(GroundingDINODiTDefaultConfig, config)

        fw = self.config.framework
        hidden_dim = int(fw.fusion.hidden_dim)
        self.num_views = int(fw.dinov3.num_views)

        self.text_encoder = GroundingDINOTextEncoder(
            bert_path=fw.text.bert_path,
            hidden_dim=hidden_dim,
            max_text_len=fw.text.max_text_len,
            sub_sentence_present=fw.text.sub_sentence_present,
            local_files_only=fw.text.local_files_only,
            freeze_text_encoder=fw.text.freeze_text_encoder,
            attn_implementation=fw.text.get("attn_implementation", "flash_attention_2"),
        )

        self.dinov3 = DINOv3Backbone(
            model_path=fw.dinov3.model_path,
            local_files_only=fw.dinov3.local_files_only,
            image_size=fw.dinov3.image_size,
            freeze_vision_encoder=fw.dinov3.freeze_vision_encoder,
            attn_implementation=fw.dinov3.get("attn_implementation", "flash_attention_2"),
        )
        self.vision_proj = VisionProjector(
            in_dim=self.dinov3.hidden_size,
            out_dim=hidden_dim,
            hidden_dim=max(hidden_dim * 4, self.dinov3.hidden_size // 2),
            dropout=fw.dinov3.vision_dropout,
        )

        self.vision_pos_embed = torch.nn.Parameter(
            torch.zeros(1, self.num_views, self.dinov3.num_patches, hidden_dim)
        )
        torch.nn.init.trunc_normal_(self.vision_pos_embed, std=float(fw.dinov3.vision_pos_init_std))
        self.vision_pos_scale = torch.nn.Parameter(
            torch.ones(1, self.num_views, 1, 1) * float(fw.dinov3.vision_pos_scale_init)
        )
        self.view_embed = torch.nn.Parameter(torch.zeros(1, self.num_views, 1, hidden_dim))
        torch.nn.init.trunc_normal_(self.view_embed, std=0.02)

        self.feature_enhancer = GroundingDINOFeatureEnhancer(
            num_layers=fw.fusion.num_layers,
            d_model=hidden_dim,
            nheads=fw.fusion.nheads,
            enhancer_inner_dim=fw.fusion.enhancer_inner_dim,
            text_dropout=fw.fusion.text_dropout,
            fusion_dropout=fw.fusion.fusion_dropout,
            fusion_droppath=fw.fusion.fusion_droppath,
        )

        if fw.groundingdino.load_pretrained:
            self._load_groundingdino_pretrained(
                ckpt_path=fw.groundingdino.pretrained_ckpt,
                load_bert=fw.groundingdino.load_bert,
                load_text_proj=fw.groundingdino.load_text_proj,
                load_feature_enhancer=fw.groundingdino.load_feature_enhancer,
            )

        self.action_model = self._build_action_model()
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _build_action_model(self) -> ACTActionHead:
        action_model_type = str(self.config.framework.action_model.get("action_model_type", "act")).lower()
        if action_model_type != "act":
            raise ValueError(
                "This open-source subset only supports the ACT action head; "
                f"got action_model_type={action_model_type!r}."
            )
        return get_act_action_model(config=self.config)

    def _is_act_action_model(self) -> bool:
        return isinstance(self.action_model, ACTActionHead)

    @staticmethod
    def _rank0_print(message: str):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(message)

    @staticmethod
    def _enable_torch_flash_sdp():
        if not hasattr(torch.backends, "cuda"):
            return
        disable_flash_attn = os.getenv("STARVLA_DISABLE_FLASH_ATTN", "0").strip().lower() in {"1", "true", "yes", "on"}
        backend_flags = (
            ("enable_flash_sdp", not disable_flash_attn),
            ("enable_mem_efficient_sdp", not disable_flash_attn),
            ("enable_math_sdp", True),
        )
        for name, enabled in backend_flags:
            fn = getattr(torch.backends.cuda, name, None)
            if fn is not None:
                try:
                    fn(enabled)
                except Exception:
                    pass

    @staticmethod
    def _normalize_gdino_state_dict(checkpoint):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        return {
            (key[len("module.") :] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }

    @staticmethod
    def _load_compatible(module: torch.nn.Module, incoming: dict, module_name: str):
        current = module.state_dict()
        compatible = {}
        skipped = []
        for key, value in incoming.items():
            if key in current and tuple(current[key].shape) == tuple(value.shape):
                compatible[key] = value
            else:
                expected = tuple(current[key].shape) if key in current else None
                skipped.append((key, tuple(value.shape), expected))

        missing, unexpected = module.load_state_dict(compatible, strict=False)
        Grounding_DINO_DiT._rank0_print(
            f"[GroundingDINODiT] loaded {len(compatible)} tensors into {module_name}; "
            f"skipped={len(skipped)}, missing_after_partial={len(missing)}, unexpected={len(unexpected)}"
        )
        if skipped:
            preview = ", ".join(f"{key}:{shape}->{expected}" for key, shape, expected in skipped[:5])
            Grounding_DINO_DiT._rank0_print(f"[GroundingDINODiT] skipped {module_name} preview: {preview}")

    def _load_groundingdino_pretrained(
        self,
        ckpt_path: str,
        load_bert: bool = True,
        load_text_proj: bool = True,
        load_feature_enhancer: bool = True,
    ):
        if not ckpt_path:
            self._rank0_print("[GroundingDINODiT] groundingdino.pretrained_ckpt is empty; skip loading.")
            return

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = self._normalize_gdino_state_dict(checkpoint)
        self._rank0_print(f"[GroundingDINODiT] loading GroundingDINO pretrained weights from {ckpt_path}")

        if load_bert:
            bert_state = {
                key[len("bert.") :]: value
                for key, value in state_dict.items()
                if key.startswith("bert.")
            }
            self._load_compatible(self.text_encoder.bert, bert_state, "text_encoder.bert")

        if load_text_proj:
            text_proj_state = {
                key.replace("feat_map.", ""): value
                for key, value in state_dict.items()
                if key.startswith("feat_map.")
            }
            self._load_compatible(self.text_encoder.text_proj, text_proj_state, "text_encoder.text_proj")

        if load_feature_enhancer:
            prefix = "transformer.encoder."
            enhancer_state = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
                and (key[len(prefix) :].startswith("fusion_layers.") or key[len(prefix) :].startswith("text_layers."))
            }
            self._load_compatible(self.feature_enhancer, enhancer_state, "feature_enhancer")

    @staticmethod
    def _as_view_list(images, num_views: int) -> list[Image.Image]:
        images = to_pil_preserve(images)
        if isinstance(images, Image.Image):
            views = [images]
        elif isinstance(images, (list, tuple)):
            views = list(images)
        else:
            raise TypeError(f"Unsupported image container: {type(images)}")

        if len(views) == 0:
            raise ValueError("Each example must contain at least one image view.")
        if len(views) < num_views:
            views = views + [views[-1]] * (num_views - len(views))
        return views[:num_views]

    def _state_tensor(self, examples: List[dict], device, dtype):
        if "state" not in examples[0] or examples[0]["state"] is None:
            return None
        state = torch.as_tensor(np.array([example["state"] for example in examples]), device=device, dtype=dtype)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        if state.dim() != 3:
            raise ValueError(f"state should be [B, D] or [B, T, D], got {tuple(state.shape)}")
        return state

    def get_action_condition(self, examples: List[dict]):
        if type(examples) is not list:
            examples = [examples]

        device = next(self.parameters()).device
        batch_views = [self._as_view_list(example["image"], self.num_views) for example in examples]
        flat_images = [view for views in batch_views for view in views]
        instructions = [example["lang"] for example in examples]

        with cuda_autocast(torch.bfloat16):
            text_tokens, text_key_padding_mask, text_self_attention_masks = self.text_encoder(
                instructions=instructions,
                device=device,
            )

            pixel_values = self.dinov3.prepare_pixel_values(flat_images, device=device)
            if self.config.framework.dinov3.freeze_vision_encoder:
                with torch.no_grad():
                    dino_tokens = self.dinov3(pixel_values)
            else:
                dino_tokens = self.dinov3(pixel_values)

            batch_size = len(examples)
            dino_tokens = self.vision_proj(dino_tokens)
            dino_tokens = dino_tokens.view(batch_size, self.num_views, self.dinov3.num_patches, -1)
            pos = self.vision_pos_embed.to(device=device, dtype=dino_tokens.dtype)
            scale = self.vision_pos_scale.to(device=device, dtype=dino_tokens.dtype)
            view_embed = self.view_embed.to(device=device, dtype=dino_tokens.dtype)
            visual_tokens = dino_tokens + scale * pos + view_embed
            visual_tokens = visual_tokens.flatten(1, 2)

            visual_tokens, text_tokens = self.feature_enhancer(
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_key_padding_mask=text_key_padding_mask,
                text_self_attention_masks=text_self_attention_masks,
            )
            condition = torch.cat([visual_tokens, text_tokens], dim=1)

        state = self._state_tensor(examples, device=device, dtype=condition.dtype)
        return condition, state

    @staticmethod
    def _sync_latency_device(device):
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _time_latency_segment(timings: dict, name: str, device, fn):
        Grounding_DINO_DiT._sync_latency_device(device)
        start = time.perf_counter()
        output = fn()
        Grounding_DINO_DiT._sync_latency_device(device)
        timings[name] = (time.perf_counter() - start) * 1000.0
        return output

    def _get_action_condition_with_latency(self, examples: List[dict]):
        if type(examples) is not list:
            examples = [examples]

        timings = {}
        device = next(self.parameters()).device
        condition_start = time.perf_counter()

        batch_views = self._time_latency_segment(
            timings,
            "input_image_view_normalize",
            device,
            lambda: [self._as_view_list(example["image"], self.num_views) for example in examples],
        )
        flat_images = [view for views in batch_views for view in views]
        instructions = [example["lang"] for example in examples]

        with cuda_autocast(torch.bfloat16):
            text_tokens, text_key_padding_mask, text_self_attention_masks = self._time_latency_segment(
                timings,
                "text_encoder_tokenize_bert_proj",
                device,
                lambda: self.text_encoder(
                    instructions=instructions,
                    device=device,
                ),
            )

            pixel_values = self._time_latency_segment(
                timings,
                "image_preprocess_processor_h2d",
                device,
                lambda: self.dinov3.prepare_pixel_values(flat_images, device=device),
            )

            def _dinov3_forward():
                if self.config.framework.dinov3.freeze_vision_encoder:
                    with torch.no_grad():
                        return self.dinov3(pixel_values)
                return self.dinov3(pixel_values)

            dino_tokens = self._time_latency_segment(timings, "dinov3_forward", device, _dinov3_forward)

            def _vision_projector_pos():
                batch_size = len(examples)
                projected = self.vision_proj(dino_tokens)
                projected = projected.view(batch_size, self.num_views, self.dinov3.num_patches, -1)
                pos = self.vision_pos_embed.to(device=device, dtype=projected.dtype)
                scale = self.vision_pos_scale.to(device=device, dtype=projected.dtype)
                view_embed = self.view_embed.to(device=device, dtype=projected.dtype)
                return (projected + scale * pos + view_embed).flatten(1, 2)

            visual_tokens = self._time_latency_segment(
                timings,
                "vision_projector_pos_embed",
                device,
                _vision_projector_pos,
            )

            visual_tokens, text_tokens = self._time_latency_segment(
                timings,
                "groundingdino_feature_enhancer",
                device,
                lambda: self.feature_enhancer(
                    visual_tokens=visual_tokens,
                    text_tokens=text_tokens,
                    text_key_padding_mask=text_key_padding_mask,
                    text_self_attention_masks=text_self_attention_masks,
                ),
            )
            condition = self._time_latency_segment(
                timings,
                "condition_concat",
                device,
                lambda: torch.cat([visual_tokens, text_tokens], dim=1),
            )

        state = self._time_latency_segment(
            timings,
            "state_tensor",
            device,
            lambda: self._state_tensor(examples, device=device, dtype=condition.dtype),
        )
        self._sync_latency_device(device)
        timings["condition_total"] = (time.perf_counter() - condition_start) * 1000.0
        return condition, state, timings

    def forward(self, examples: List[dict] = None, **kwargs):
        condition, state = self.get_action_condition(examples)

        with cuda_autocast(enabled=False):
            action_dtype = self.action_model.dtype
            condition = condition.to(dtype=action_dtype)
            state = state.to(dtype=action_dtype) if state is not None else None
            actions = torch.as_tensor(
                np.array([example["action"] for example in examples]),
                device=condition.device,
                dtype=action_dtype,
            )
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_steps = 1 if self._is_act_action_model() else int(
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
            )
            action_loss = self.action_model(
                condition.repeat(repeated_steps, 1, 1),
                actions_target.repeat(repeated_steps, 1, 1),
                state.repeat(repeated_steps, 1, 1) if state is not None else None,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs):
        profile_latency = bool(kwargs.pop("profile_latency", False))
        if type(examples) is not list:
            examples = [examples]
        total_start = time.perf_counter()
        if profile_latency:
            condition, state, latency_ms = self._get_action_condition_with_latency(examples)
        else:
            condition, state = self.get_action_condition(examples)
            latency_ms = {}

        def _action_head_predict():
            with cuda_autocast(enabled=False):
                action_dtype = self.action_model.dtype
                return self.action_model.predict_action(
                    condition.to(dtype=action_dtype),
                    state.to(dtype=action_dtype) if state is not None else None,
                )

        device = condition.device
        if profile_latency:
            pred_actions = self._time_latency_segment(latency_ms, "action_head_predict", device, _action_head_predict)
        else:
            pred_actions = _action_head_predict()

        output = {"normalized_actions": pred_actions.detach().float().cpu().numpy()}
        if profile_latency:
            self._sync_latency_device(device)
            latency_ms["predict_action_total"] = (time.perf_counter() - total_start) * 1000.0
            output["latency_ms"] = latency_ms
        return output
