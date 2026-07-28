from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers should be >= 1, got {num_layers}")
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class StateProjector(nn.Module):
    def __init__(
        self,
        state_dim=8,
        hidden_dim=256,
        num_tokens=2,
        proj_hidden=256,
        dropout=0.1,
    ):
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
        if state.ndim == 3:
            state = state[:, -1, :]
        if state.ndim != 2:
            raise ValueError(f"state should be [B, D] or [B, T, D], got {tuple(state.shape)}")
        state_tokens = self.net(state).view(state.shape[0], self.num_tokens, self.hidden_dim)
        return self.out_norm(state_tokens + self.pos.to(dtype=state_tokens.dtype, device=state_tokens.device))


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
        mlp_hidden_dim=512,
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
        self.action_head = MLP(hidden_dim, mlp_hidden_dim, action_dim, num_layers=3)

    def forward(self, memory):
        batch_size = memory.shape[0]
        tgt = self.action_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = self.decoder(tgt=tgt, memory=memory)
        pred_actions = torch.tanh(self.action_head(hidden_states))
        return pred_actions, hidden_states


@dataclass
class ACTActionHeadConfig(PretrainedConfig):
    action_dim: int = field(default=7)
    state_dim: int = field(default=8)
    action_horizon: int = field(default=8)
    hidden_size: int = field(default=256)
    nheads: int = field(default=8)
    num_layers: int = field(default=3)
    dim_feedforward: int = field(default=3072)
    dropout: float = field(default=0.1)
    state_tokens: int = field(default=2)
    state_hidden_dim: int = field(default=256)
    mlp_hidden_dim: int = field(default=512)
    loss_type: str = field(default="l1")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class ACTActionHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config

        diffusion_cfg = config.get("diffusion_model_cfg", {}) or {}
        condition_dim = int(
            config.get(
                "act_hidden_dim",
                diffusion_cfg.get("cross_attention_dim", config.get("action_hidden_dim", config.get("hidden_size", 256))),
            )
        )
        nheads = int(config.get("act_nheads", full_config.framework.get("fusion", {}).get("nheads", 8)))
        if condition_dim % nheads != 0:
            raise ValueError(f"ACT hidden dim {condition_dim} must be divisible by nheads {nheads}.")

        self.config = ACTActionHeadConfig(
            action_dim=int(config.action_dim),
            state_dim=int(config.get("state_dim", 0) or 0),
            action_horizon=int(config.action_horizon),
            hidden_size=condition_dim,
            nheads=nheads,
            num_layers=int(config.get("act_num_layers", config.get("num_layers", 3))),
            dim_feedforward=int(config.get("act_dim_feedforward", 3072)),
            dropout=float(config.get("act_dropout", config.get("dropout", 0.1))),
            state_tokens=int(config.get("act_state_tokens", config.get("num_state_tokens", 2))),
            state_hidden_dim=int(config.get("act_state_hidden_dim", condition_dim)),
            mlp_hidden_dim=int(config.get("act_mlp_hidden_dim", 512)),
            loss_type=str(config.get("act_loss_type", "l1")).lower(),
        )

        self.action_dim = self.config.action_dim
        self.action_horizon = self.config.action_horizon

        self.state_proj = (
            StateProjector(
                state_dim=self.config.state_dim,
                hidden_dim=self.config.hidden_size,
                num_tokens=self.config.state_tokens,
                proj_hidden=self.config.state_hidden_dim,
                dropout=self.config.dropout,
            )
            if self.config.state_dim > 0
            else None
        )
        self.action_policy = ACTActionDecoder(
            hidden_dim=self.config.hidden_size,
            nheads=self.config.nheads,
            action_dim=self.config.action_dim,
            chunk_size=self.config.action_horizon,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            mlp_hidden_dim=self.config.mlp_hidden_dim,
        )

    def _build_memory(self, vl_embs, state=None):
        if self.state_proj is None or state is None:
            return vl_embs
        state_tokens = self.state_proj(state.to(dtype=vl_embs.dtype, device=vl_embs.device))
        return torch.cat([vl_embs, state_tokens], dim=1)

    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None):
        del encoder_attention_mask
        memory = self._build_memory(vl_embs, state)
        pred_actions, _ = self.action_policy(memory)
        actions = actions[:, -self.action_horizon :, :].to(dtype=pred_actions.dtype, device=pred_actions.device)
        if self.config.loss_type == "mse":
            return F.mse_loss(pred_actions, actions)
        if self.config.loss_type in ("smooth_l1", "huber"):
            return F.smooth_l1_loss(pred_actions, actions)
        return F.l1_loss(pred_actions, actions)

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        memory = self._build_memory(vl_embs, state)
        pred_actions, _ = self.action_policy(memory)
        return pred_actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    return ACTActionHead(full_config=config)
