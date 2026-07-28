from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

try:
    from timm.layers import DropPath
except Exception:
    from timm.models.layers import DropPath


def cuda_autocast(dtype=torch.bfloat16, enabled=True):
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=dtype, enabled=enabled)
    return nullcontext()


def _get_clones(module: nn.Module, num_layers: int) -> nn.ModuleList:
    import copy

    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _auto_model_from_pretrained(model_path: str, local_files_only: bool = True, attn_implementation: str | None = None):
    attempts = []
    if attn_implementation:
        attempts.append(attn_implementation)
        if attn_implementation == "flash_attention_2":
            attempts.append("sdpa")
    attempts.append(None)

    seen = set()
    last_error = None
    for impl in attempts:
        if impl in seen:
            continue
        seen.add(impl)
        kwargs = {"local_files_only": local_files_only}
        if impl:
            kwargs["attn_implementation"] = impl
        try:
            model = AutoModel.from_pretrained(model_path, **kwargs)
            return model, impl or "default"
        except Exception as exc:
            last_error = exc
            if impl:
                print(
                    f"[GroundingDINODiT] AutoModel attn_implementation={impl!r} failed for "
                    f"{model_path}; trying fallback. Error: {type(exc).__name__}: {exc}"
                )

    raise last_error


def generate_masks_with_special_tokens(tokenized, special_tokens_list: Iterable[int]):
    input_ids = tokenized["input_ids"]
    batch_size, num_tokens = input_ids.shape
    device = input_ids.device

    attention_mask = torch.eye(num_tokens, device=device, dtype=torch.bool).unsqueeze(0).repeat(batch_size, 1, 1)
    position_ids = torch.zeros((batch_size, num_tokens), device=device, dtype=torch.long)

    for row in range(batch_size):
        special_mask = torch.zeros(num_tokens, device=device, dtype=torch.bool)
        for special_token in special_tokens_list:
            if special_token is not None and int(special_token) >= 0:
                special_mask |= input_ids[row] == int(special_token)

        special_cols = torch.nonzero(special_mask, as_tuple=False).flatten().tolist()
        previous_col = 0
        for col in special_cols:
            if col == 0 or col == num_tokens - 1:
                attention_mask[row, col, col] = True
                position_ids[row, col] = 0
            else:
                start = previous_col + 1
                end = col + 1
                attention_mask[row, start:end, start:end] = True
                position_ids[row, start:end] = torch.arange(0, end - start, device=device)
            previous_col = col

    return attention_mask, position_ids


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.nhead = int(nhead)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if src_mask is not None and src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
            src_mask = src_mask.repeat(self.nhead, 1, 1)

        q = k = src if pos is None else src + pos
        src2 = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim: int, l_dim: int, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.v_dim = int(v_dim)
        self.l_dim = int(l_dim)
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}.")

        self.scale = self.head_dim**-0.5
        self.dropout = float(dropout)
        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)
        self.use_sdpa = hasattr(F, "scaled_dot_product_attention")
        self._reset_parameters()

    def _reset_parameters(self):
        for module in [
            self.v_proj,
            self.l_proj,
            self.values_v_proj,
            self.values_l_proj,
            self.out_v_proj,
            self.out_l_proj,
        ]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @staticmethod
    def _padding_mask_to_sdpa(mask: torch.Tensor | None, dtype: torch.dtype, target_len: int):
        if mask is None:
            return None
        additive_mask = torch.zeros(
            (mask.shape[0], 1, target_len, mask.shape[1]),
            device=mask.device,
            dtype=dtype,
        )
        return additive_mask.masked_fill(mask[:, None, None, :], float("-inf"))

    def _forward_sdpa(self, v, l, attention_mask_v=None, attention_mask_l=None):
        batch_size, tgt_len, _ = v.size()
        src_len = l.size(1)

        query_v = self._shape(self.v_proj(v), tgt_len, batch_size)
        key_l = self._shape(self.l_proj(l), src_len, batch_size)
        value_v = self._shape(self.values_v_proj(v), tgt_len, batch_size)
        value_l = self._shape(self.values_l_proj(l), src_len, batch_size)

        dropout_p = self.dropout if self.training else 0.0
        attn_mask_l = self._padding_mask_to_sdpa(attention_mask_l, query_v.dtype, tgt_len)
        attn_output_v = F.scaled_dot_product_attention(
            query_v,
            key_l,
            value_l,
            attn_mask=attn_mask_l,
            dropout_p=dropout_p,
        )

        attn_mask_v = self._padding_mask_to_sdpa(attention_mask_v, key_l.dtype, src_len)
        attn_output_l = F.scaled_dot_product_attention(
            key_l,
            query_v,
            value_v,
            attn_mask=attn_mask_v,
            dropout_p=dropout_p,
        )

        attn_output_v = attn_output_v.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)
        attn_output_l = attn_output_l.transpose(1, 2).reshape(batch_size, src_len, self.embed_dim)
        return self.out_v_proj(attn_output_v), self.out_l_proj(attn_output_l)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        if self.use_sdpa:
            return self._forward_sdpa(v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l)

        batch_size, tgt_len, _ = v.size()
        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, batch_size)
        value_v_states = self._shape(self.values_v_proj(v), -1, batch_size)
        value_l_states = self._shape(self.values_l_proj(l), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = attn_weights - attn_weights.max()
        attn_weights = torch.clamp(attn_weights, min=-50000, max=50000)

        attn_weights_l = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_l - torch.max(attn_weights_l, dim=-1, keepdim=True)[0]
        attn_weights_l = torch.clamp(attn_weights_l, min=-50000, max=50000)

        if attention_mask_v is not None:
            attention_mask_v = attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_weights_l = attn_weights_l.masked_fill(attention_mask_v, float("-inf"))
        if attention_mask_l is not None:
            attention_mask_l = attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_weights = attn_weights.masked_fill(attention_mask_l, float("-inf"))

        attn_probs_v = F.dropout(attn_weights.softmax(dim=-1), p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l.softmax(dim=-1), p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        attn_output_v = attn_output_v.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)
        attn_output_l = attn_output_l.view(batch_size, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2).reshape(batch_size, src_len, self.embed_dim)
        return self.out_v_proj(attn_output_v), self.out_l_proj(attn_output_l)


class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim: int,
        l_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        init_values: float = 1e-4,
    ):
        super().__init__()
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        v_norm = self.layer_norm_v(v)
        l_norm = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v_norm, l_norm, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l)
        return v + self.drop_path(self.gamma_v * delta_v), l + self.drop_path(self.gamma_l * delta_l)


class GroundingDINOFeatureEnhancer(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        d_model: int = 768,
        nheads: int = 12,
        enhancer_inner_dim: int = 1024,
        text_dropout: float = 0.0,
        fusion_dropout: float = 0.0,
        fusion_droppath: float = 0.1,
    ):
        super().__init__()
        text_nheads = max(1, int(nheads) // 2)
        fusion_nheads = max(1, int(nheads) // 2)
        text_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=text_nheads,
            dim_feedforward=enhancer_inner_dim,
            dropout=text_dropout,
        )
        fusion_layer = BiAttentionBlock(
            v_dim=d_model,
            l_dim=d_model,
            embed_dim=enhancer_inner_dim,
            num_heads=fusion_nheads,
            dropout=fusion_dropout,
            drop_path=fusion_droppath,
        )
        self.text_layers = _get_clones(text_layer, num_layers)
        self.fusion_layers = _get_clones(fusion_layer, num_layers)

    def forward(self, visual_tokens, text_tokens, text_key_padding_mask, text_self_attention_masks=None):
        text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)
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
                # `src_mask` already isolates padding positions to self-attend.
                # Passing key_padding_mask as well can make padding query rows all-masked,
                # producing NaNs that poison gradients before the rows are zeroed below.
                src_key_padding_mask=None,
                pos=None,
            ).transpose(0, 1)
            text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)
        return visual_tokens, text_tokens


class VisionProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(hidden_dim or max(out_dim * 4, in_dim // 2))
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


class GroundingDINOTextEncoder(nn.Module):
    def __init__(
        self,
        bert_path: str,
        hidden_dim: int = 768,
        max_text_len: int = 256,
        sub_sentence_present: bool = True,
        local_files_only: bool = True,
        freeze_text_encoder: bool = False,
        attn_implementation: str | None = "flash_attention_2",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=local_files_only)
        self.bert, self.attn_implementation = _auto_model_from_pretrained(
            bert_path,
            local_files_only=local_files_only,
            attn_implementation=attn_implementation,
        )
        if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
            for param in self.bert.pooler.parameters():
                param.requires_grad_(False)

        self.max_text_len = int(max_text_len)
        self.sub_sentence_present = bool(sub_sentence_present)
        self.freeze_text_encoder = bool(freeze_text_encoder)
        self.text_proj = nn.Linear(self.bert.config.hidden_size, hidden_dim, bias=True)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.constant_(self.text_proj.bias, 0.0)
        self.special_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        if self.freeze_text_encoder:
            for param in self.bert.parameters():
                param.requires_grad_(False)

    def forward(self, instructions, device):
        tokenized = self.tokenizer(
            instructions,
            padding="longest",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        ).to(device)
        text_self_attention_masks, position_ids = generate_masks_with_special_tokens(tokenized, self.special_tokens)

        if self.sub_sentence_present:
            bert_inputs = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            bert_inputs["attention_mask"] = text_self_attention_masks
            bert_inputs["position_ids"] = position_ids
        else:
            bert_inputs = tokenized

        if self.freeze_text_encoder:
            with torch.no_grad():
                bert_output = self.bert(**bert_inputs)
        else:
            bert_output = self.bert(**bert_inputs)

        encoded_text = self.text_proj(bert_output.last_hidden_state)
        text_token_mask = tokenized.attention_mask.bool()
        text_key_padding_mask = ~text_token_mask
        encoded_text = encoded_text.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)
        return encoded_text, text_key_padding_mask, text_self_attention_masks


class DINOv3Backbone(nn.Module):
    def __init__(
        self,
        model_path: str,
        local_files_only: bool = True,
        image_size: int | None = None,
        freeze_vision_encoder: bool = False,
        attn_implementation: str | None = "flash_attention_2",
    ):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=local_files_only)
        self.model, self.attn_implementation = _auto_model_from_pretrained(
            model_path,
            local_files_only=local_files_only,
            attn_implementation=attn_implementation,
        )
        self.hidden_size = self._get_hidden_size(self.model.config)
        self.patch_size = self._get_patch_size(self.model.config)
        self.prefix_tokens = self._get_num_prefix_tokens(self.model.config, default_value=5)
        self.image_size = int(image_size or self._infer_processor_size(self.processor) or getattr(self.model.config, "image_size"))
        self.num_patches = (self.image_size // self.patch_size) ** 2

        if hasattr(self.processor, "size"):
            self.processor.size = {"height": self.image_size, "width": self.image_size}

        # DINOv3 keeps a mask token for masked-image pretraining, but StarVLA only
        # uses dense patch embeddings from fully observed images. Leaving this
        # parameter trainable causes DDP unused-parameter failures during full
        # finetuning because it never participates in the forward graph.
        embeddings = getattr(self.model, "embeddings", None)
        mask_token = getattr(embeddings, "mask_token", None)
        if mask_token is not None:
            mask_token.requires_grad_(False)

        if freeze_vision_encoder:
            for param in self.model.parameters():
                param.requires_grad_(False)

    @staticmethod
    def _infer_processor_size(processor):
        size = getattr(processor, "size", None)
        if isinstance(size, dict):
            return size.get("height") or size.get("shortest_edge")
        return None

    @staticmethod
    def _get_hidden_size(config):
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
            return int(config.vision_config.hidden_size)
        raise AttributeError("Cannot infer hidden size from DINOv3 config.")

    @staticmethod
    def _get_patch_size(config):
        if hasattr(config, "patch_size"):
            return int(config.patch_size)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "patch_size"):
            return int(config.vision_config.patch_size)
        raise AttributeError("Cannot infer patch size from DINOv3 config.")

    @staticmethod
    def _get_num_prefix_tokens(config, default_value=0):
        if hasattr(config, "num_register_tokens"):
            return int(getattr(config, "num_register_tokens")) + 1
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "num_register_tokens"):
            return int(getattr(config.vision_config, "num_register_tokens")) + 1
        return int(default_value)

    def prepare_pixel_values(self, images, device):
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return pixel_values.to(device=device)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        tokens = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        expected_with_prefix = self.num_patches + self.prefix_tokens
        if tokens.shape[1] == expected_with_prefix:
            return tokens[:, self.prefix_tokens :, :]
        if tokens.shape[1] == self.num_patches:
            return tokens
        raise RuntimeError(
            f"DINOv3 produced {tokens.shape[1]} tokens, expected {self.num_patches} patch tokens "
            f"or {expected_with_prefix} tokens including prefix."
        )
