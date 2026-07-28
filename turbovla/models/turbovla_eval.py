from pathlib import Path

import torch

from .turbovla import GroundingDINOVLA


class EvalCachedTextBank:
    def __init__(self, payload, path):
        if not isinstance(payload, dict):
            raise ValueError(f"text cache at {path} must be a dict")
        for key in ("instructions", "last_hidden_state", "attention_mask"):
            if key not in payload:
                raise ValueError(f"text cache at {path} missing required key: {key}")

        self.path = str(path)
        self.instructions = [str(item) for item in payload["instructions"]]
        raw_mapping = payload.get("instruction_to_index")
        if raw_mapping is None:
            raw_mapping = {instruction: idx for idx, instruction in enumerate(self.instructions)}
        self.instruction_to_index = {str(key): int(value) for key, value in raw_mapping.items()}
        self.stripped_to_index = {}
        for instruction, idx in self.instruction_to_index.items():
            self.stripped_to_index.setdefault(instruction.strip(), idx)

        self.last_hidden_state = payload["last_hidden_state"].detach().cpu().contiguous()
        self.attention_mask = payload["attention_mask"].detach().cpu().bool().contiguous()
        self.text_self_attention_masks = payload.get("text_self_attention_masks")
        if self.text_self_attention_masks is not None:
            self.text_self_attention_masks = self.text_self_attention_masks.detach().cpu().bool().contiguous()

        if self.last_hidden_state.ndim != 3:
            raise ValueError(f"last_hidden_state should be [N, L, H], got {self.last_hidden_state.shape}")
        if self.attention_mask.ndim != 2:
            raise ValueError(f"attention_mask should be [N, L], got {self.attention_mask.shape}")
        if tuple(self.last_hidden_state.shape[:2]) != tuple(self.attention_mask.shape):
            raise ValueError(
                f"text cache shape mismatch: last_hidden_state={self.last_hidden_state.shape}, "
                f"attention_mask={self.attention_mask.shape}"
            )

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"text cache not found: {path}")
        return cls(torch.load(path, map_location="cpu"), path)

    @property
    def text_hidden_dim(self):
        return int(self.last_hidden_state.shape[-1])

    def lookup(self, instructions):
        indices = []
        for instruction in instructions:
            instruction = str(instruction)
            idx = self.instruction_to_index.get(instruction)
            if idx is None:
                idx = self.stripped_to_index.get(instruction.strip())
            if idx is None:
                known = "\n".join(f"- {item}" for item in self.instructions[:20])
                raise KeyError(
                    f"instruction not found in text cache: {instruction!r}\nKnown cached instructions:\n{known}"
                )
            indices.append(idx)

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        result = {
            "last_hidden_state": self.last_hidden_state.index_select(0, idx_tensor),
            "attention_mask": self.attention_mask.index_select(0, idx_tensor),
        }
        if self.text_self_attention_masks is not None:
            result["text_self_attention_masks"] = self.text_self_attention_masks.index_select(0, idx_tensor)
        return result


class GroundingDINOVLATextCacheEval(GroundingDINOVLA):
    def __init__(self, *args, text_cache_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_text_cache = EvalCachedTextBank.load(text_cache_path)
        if self.eval_text_cache.text_hidden_dim != self.text_hidden_dim:
            raise ValueError(
                f"text cache hidden dim {self.eval_text_cache.text_hidden_dim} does not match "
                f"model text_hidden_dim {self.text_hidden_dim}"
            )

    def forward(self, cached_text_or_instructions, samples, state):
        if isinstance(cached_text_or_instructions, dict):
            cached_text = cached_text_or_instructions
        else:
            cached_text = self.eval_text_cache.lookup(cached_text_or_instructions)
        return super().forward(cached_text, samples, state)


def _resolve_text_cache_path(args):
    repo_root = Path(__file__).resolve().parents[3]
    candidates = []
    for name in ("text_cache_path", "eval_text_cache_path", "text_encoder_type"):
        value = getattr(args, name, None)
        if value:
            candidates.append(Path(str(value)))
    candidates.append(repo_root / "data" / "libero" / "libero_10_bert_text_cache.pt")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
        if not candidate.is_absolute():
            repo_candidate = repo_root / candidate
            if repo_candidate.is_file():
                return str(repo_candidate)

    tried = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"could not resolve eval text cache path; tried: {tried}")


def build_groundingdino_vla(args):
    return GroundingDINOVLATextCacheEval(
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
        text_cache_path=_resolve_text_cache_path(args),
    )


build_groundingdino = build_groundingdino_vla
build_turbovla_eval = build_groundingdino_vla
TurboVLAEval = GroundingDINOVLATextCacheEval
