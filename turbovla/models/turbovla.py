import torch
from torch import nn
from transformers import AutoModel

from .components.fusion import BiAttentionBlock
from .components.transformer import TransformerEncoderLayer
from .components.utils import MLP, _get_clones


class ACTActionDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nheads,
        action_dim,
        chunk_size=8,
        num_layers=3,
        dim_feedforward=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.action_queries = nn.Embedding(self.chunk_size, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = MLP(hidden_dim, 512, action_dim, 3)

    def forward(self, memory):
        batch_size = memory.shape[0]
        tgt = self.action_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = self.decoder(tgt=tgt, memory=memory)
        pred_actions = torch.tanh(self.action_head(hidden_states))
        return pred_actions, hidden_states


class VisionProjector(nn.Module):
    def __init__(self, in_dim=1024, out_dim=768, hidden_dim=1536, dropout=0.1):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm_out(self.skip(x) + self.mlp(self.norm_in(x)))


class StateProjector(nn.Module):
    def __init__(self, state_dim=8, hidden_dim=768, num_tokens=2, proj_hidden=256, dropout=0.1):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, self.num_tokens * self.hidden_dim),
        )
        self.pos = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim) * 0.02)
        self.out_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, state):
        x = self.net(state)
        x = x.view(state.shape[0], self.num_tokens, self.hidden_dim)
        x = self.out_norm(x + self.pos)
        return x


class GroundingDINOFeatureEnhancer(nn.Module):
    def __init__(
        self,
        num_layers=4,
        d_model=768,
        nheads=12,
        enhancer_inner_dim=1024,
        text_dropout=0.0,
        fusion_dropout=0.0,
        fusion_droppath=0.1,
    ):
        super().__init__()
        text_nheads = max(1, nheads // 2)
        fusion_nheads = max(1, nheads // 2)

        text_enhance_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=text_nheads,
            dim_feedforward=enhancer_inner_dim,
            dropout=text_dropout,
        )
        feature_fusion_layer = BiAttentionBlock(
            v_dim=d_model,
            l_dim=d_model,
            embed_dim=enhancer_inner_dim,
            num_heads=fusion_nheads,
            dropout=fusion_dropout,
            drop_path=fusion_droppath,
        )

        self.text_layers = _get_clones(text_enhance_layer, num_layers)
        self.fusion_layers = _get_clones(feature_fusion_layer, num_layers)

    def forward(
        self,
        visual_tokens,
        text_tokens,
        text_key_padding_mask,
        text_self_attention_masks=None,
    ):
        for fusion_layer, text_layer in zip(self.fusion_layers, self.text_layers):
            visual_tokens, text_tokens = fusion_layer(
                v=visual_tokens,
                l=text_tokens,
                attention_mask_v=None,
                attention_mask_l=text_key_padding_mask,
            )

            src_mask = None
            if text_self_attention_masks is not None:
                src_mask = ~text_self_attention_masks

            text_tokens = text_layer(
                src=text_tokens.transpose(0, 1),
                src_mask=src_mask,
                src_key_padding_mask=text_key_padding_mask,
                pos=None,
            ).transpose(0, 1)

        return visual_tokens, text_tokens


class GroundingDINOVLA(nn.Module):
    def __init__(
        self,
        LOCAL_DINOV3_PATH,
        action_dim=7,
        chunk_size=8,
        text_hidden_dim=768,
        hidden_dim=256,
        nheads=8,
        dim_feedforward=2048,
        enhancer_inner_dim=1024,
        max_text_len=256,
        vla_feature_enhancer_layers=6,
        state_dim=8,
        num_state_tokens=2,
        text_dropout=0.0,
        fusion_dropout=0.0,
        fusion_droppath=0.1,
        freeze_vision_encoder=False,
        local_files_only=True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.text_hidden_dim = int(text_hidden_dim)
        self.max_text_len = int(max_text_len)
        self.freeze_vision_encoder = bool(freeze_vision_encoder)

        self.text_proj = nn.Linear(self.text_hidden_dim, self.hidden_dim, bias=True)
        nn.init.constant_(self.text_proj.bias.data, 0)
        nn.init.xavier_uniform_(self.text_proj.weight.data)

        self.dinov3 = AutoModel.from_pretrained(
            LOCAL_DINOV3_PATH,
            local_files_only=bool(local_files_only),
        )

        self.dinov3_prefix_tokens = self._get_num_prefix_tokens(self.dinov3.config, default_value=5)
        dino_hidden = self._get_hidden_size(self.dinov3.config)
        self.vision_proj = VisionProjector(
            in_dim=dino_hidden,
            out_dim=self.hidden_dim,
            hidden_dim=max(self.hidden_dim * 4, dino_hidden // 2),
            dropout=0.1,
        )

        self.view_embed = nn.Parameter(torch.zeros(1, 2, self.hidden_dim))
        nn.init.trunc_normal_(self.view_embed, std=0.02)

        self.feature_enhancer = GroundingDINOFeatureEnhancer(
            num_layers=vla_feature_enhancer_layers,
            d_model=self.hidden_dim,
            nheads=nheads,
            enhancer_inner_dim=enhancer_inner_dim,
            text_dropout=text_dropout,
            fusion_dropout=fusion_dropout,
            fusion_droppath=fusion_droppath,
        )

        self.state_proj = StateProjector(
            state_dim=state_dim,
            hidden_dim=self.hidden_dim,
            num_tokens=num_state_tokens,
            proj_hidden=256,
            dropout=0.1,
        )

        self.action_policy = ACTActionDecoder(
            hidden_dim=self.hidden_dim,
            nheads=nheads,
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
        )

        if self.freeze_vision_encoder:
            for param in self.dinov3.parameters():
                param.requires_grad_(False)

    @staticmethod
    def _get_hidden_size(config):
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
            return int(config.vision_config.hidden_size)
        raise AttributeError("Cannot infer hidden size from vision config")

    @staticmethod
    def _get_patch_size(config):
        if hasattr(config, "patch_size"):
            return int(config.patch_size)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "patch_size"):
            return int(config.vision_config.patch_size)
        raise AttributeError("Cannot infer patch size from vision config")

    @staticmethod
    def _get_num_prefix_tokens(config, default_value=0):
        if hasattr(config, "num_register_tokens"):
            return int(getattr(config, "num_register_tokens")) + 1
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "num_register_tokens"):
            return int(getattr(config.vision_config, "num_register_tokens")) + 1
        return int(default_value)

    @staticmethod
    def _extract_patch_tokens(outputs, expected_n, backbone_name, allowed_prefix_tokens=(0,)):
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None and len(hidden_states) >= 2:
            tokens = hidden_states[-1]
        else:
            tokens = outputs.last_hidden_state

        for num_prefix in allowed_prefix_tokens:
            if tokens.shape[1] == expected_n + num_prefix:
                return tokens[:, num_prefix:, :] if num_prefix > 0 else tokens

        allowed_shapes = [expected_n + int(num_prefix) for num_prefix in allowed_prefix_tokens]
        raise RuntimeError(
            f"{backbone_name} produced {tokens.shape[1]} tokens, expected patch tokens {expected_n} "
            f"with allowed total token counts {allowed_shapes}."
        )

    def _add_view_embed(self, tokens, view_idx):
        return tokens + self.view_embed[:, view_idx : view_idx + 1, :]

    def _encode_text(self, cached_text, device):
        if not isinstance(cached_text, dict):
            raise ValueError("cached_text must be a dict with last_hidden_state and attention_mask tensors")
        if "last_hidden_state" not in cached_text or "attention_mask" not in cached_text:
            raise ValueError("cached_text must contain last_hidden_state and attention_mask")

        last_hidden_state = cached_text["last_hidden_state"]
        attention_mask = cached_text["attention_mask"]
        if last_hidden_state.ndim != 3:
            raise ValueError(f"last_hidden_state should be [B, L, H], got {last_hidden_state.shape}")
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask should be [B, L], got {attention_mask.shape}")
        if tuple(last_hidden_state.shape[:2]) != tuple(attention_mask.shape):
            raise ValueError(
                f"text cache shape mismatch: last_hidden_state={last_hidden_state.shape}, "
                f"attention_mask={attention_mask.shape}"
            )
        if last_hidden_state.shape[-1] != self.text_hidden_dim:
            raise ValueError(
                f"text cache hidden dim {last_hidden_state.shape[-1]} does not match model text_hidden_dim "
                f"{self.text_hidden_dim}"
            )

        last_hidden_state = last_hidden_state[:, : self.max_text_len, :]
        attention_mask = attention_mask[:, : self.max_text_len]
        text_self_attention_masks = cached_text.get("text_self_attention_masks")
        if text_self_attention_masks is not None:
            if text_self_attention_masks.ndim != 3:
                raise ValueError(
                    f"text_self_attention_masks should be [B, L, L], got {text_self_attention_masks.shape}"
                )
            if text_self_attention_masks.shape[0] != last_hidden_state.shape[0]:
                raise ValueError("text_self_attention_masks batch size does not match last_hidden_state")
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
        else:
            token_mask_for_self = attention_mask.bool()
            text_self_attention_masks = token_mask_for_self[:, :, None] & token_mask_for_self[:, None, :]

        last_hidden_state = last_hidden_state.to(
            device=device,
            dtype=self.text_proj.weight.dtype,
            non_blocking=True,
        )
        text_token_mask = attention_mask.to(device=device, dtype=torch.bool, non_blocking=True)
        text_self_attention_masks = text_self_attention_masks.to(
            device=device,
            dtype=torch.bool,
            non_blocking=True,
        )

        encoded_text = self.text_proj(last_hidden_state)
        text_key_padding_mask = ~text_token_mask
        return encoded_text, text_key_padding_mask, text_self_attention_masks

    def _normalize_samples(self, samples):
        if isinstance(samples, dict):
            if "dinov3" in samples:
                dino_samples = samples["dinov3"]
            elif "dinov2" in samples:
                dino_samples = samples["dinov2"]
            else:
                raise ValueError("samples dict must contain 'dinov3' (or legacy key 'dinov2')")
        else:
            dino_samples = samples

        if dino_samples.ndim == 6:
            dino_samples = dino_samples[:, -1, ...]

        if dino_samples.ndim != 5:
            raise ValueError(
                "samples should be [B, 2, 3, H, W] or [B, T, 2, 3, H, W] for dinov3 inputs"
            )
        if dino_samples.shape[1] != 2:
            raise ValueError("current model expects 2 views for dinov3 inputs")

        return dino_samples

    def _encode_one_view(self, dino_pixel_values, view_idx):
        dino_patch = self._get_patch_size(self.dinov3.config)
        dino_h, dino_w = dino_pixel_values.shape[-2:]

        if dino_h % dino_patch != 0 or dino_w % dino_patch != 0:
            raise ValueError(
                f"DINOv3 input size {(dino_h, dino_w)} must be divisible by patch size {dino_patch}."
            )

        dino_expected_n = (dino_h // dino_patch) * (dino_w // dino_patch)

        if self.freeze_vision_encoder:
            with torch.no_grad():
                dino_outputs = self.dinov3(pixel_values=dino_pixel_values, output_hidden_states=True)
        else:
            dino_outputs = self.dinov3(pixel_values=dino_pixel_values, output_hidden_states=True)

        dino_tokens = self._extract_patch_tokens(
            dino_outputs,
            dino_expected_n,
            "DINOv3",
            allowed_prefix_tokens=(self.dinov3_prefix_tokens,),
        )

        dino_tokens = self.vision_proj(dino_tokens)
        return self._add_view_embed(dino_tokens, view_idx=view_idx)

    def forward(self, cached_text, samples, state):
        dino_samples = self._normalize_samples(samples)
        batch_size = dino_samples.shape[0]

        if state.ndim == 3:
            state = state[:, -1, ...]
        if state.ndim != 2:
            raise ValueError(f"state should be [B, D] or [B, T, D], got {state.shape}")

        device = dino_samples.device
        encoded_text, text_key_padding_mask, text_self_attention_masks = self._encode_text(
            cached_text,
            device=device,
        )
        if encoded_text.shape[0] != batch_size:
            raise ValueError(
                f"cached text batch size {encoded_text.shape[0]} does not match sample batch size {batch_size}"
            )

        feat_img1 = self._encode_one_view(dino_samples[:, 0, ...], view_idx=0)
        feat_img2 = self._encode_one_view(dino_samples[:, 1, ...], view_idx=1)

        visual_tokens = torch.cat([feat_img1, feat_img2], dim=1)
        visual_tokens, text_tokens = self.feature_enhancer(
            visual_tokens=visual_tokens,
            text_tokens=encoded_text,
            text_key_padding_mask=text_key_padding_mask,
            text_self_attention_masks=text_self_attention_masks,
        )

        state_tokens = self.state_proj(state)
        act_memory = torch.cat([visual_tokens, text_tokens, state_tokens], dim=1)
        pred_actions, _ = self.action_policy(act_memory)
        return pred_actions


def build_groundingdino_vla(args):
    model = GroundingDINOVLA(
        LOCAL_DINOV3_PATH=args.LOCAL_DINOV3_PATH,
        action_dim=getattr(args, "action_dim", 7),
        chunk_size=getattr(args, "chunk_size", 8),
        text_hidden_dim=getattr(args, "text_hidden_dim", 768),
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        dim_feedforward=args.dim_feedforward,
        enhancer_inner_dim=getattr(args, "enhancer_inner_dim", 1024),
        max_text_len=getattr(args, "max_text_len", 256),
        vla_feature_enhancer_layers=getattr(args, "vla_feature_enhancer_layers", 6),
        state_dim=getattr(args, "state_dim", 8),
        num_state_tokens=getattr(args, "num_state_tokens", 2),
        text_dropout=getattr(args, "text_dropout", 0.0),
        fusion_dropout=getattr(args, "fusion_dropout", 0.0),
        fusion_droppath=getattr(args, "fusion_droppath", 0.1),
        freeze_vision_encoder=getattr(args, "freeze_vision_encoder", False),
        local_files_only=getattr(args, "local_files_only", True),
    )
    return model


build_groundingdino = build_groundingdino_vla
build_turbovla = build_groundingdino_vla
TurboVLA = GroundingDINOVLA
