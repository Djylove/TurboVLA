from pathlib import Path

import torch
from torch import nn

from experiments.gr3.train import _load_compatible


class _Projection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 31)]
        )


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_head = nn.Module()
        self.action_head.state_projection = nn.Module()
        self.action_head.state_projection.net = nn.Sequential(
            nn.LayerNorm(31), nn.Linear(31, 4)
        )
        self.action_head.decoder = nn.Module()
        self.action_head.decoder.action_projection = _Projection()


def test_33d_state_and_37d_action_are_sliced_into_31d_head(tmp_path: Path) -> None:
    model = _Model()
    source = model.state_dict()
    weight_key = "action_head.decoder.action_projection.layers.2.weight"
    bias_key = "action_head.decoder.action_projection.layers.2.bias"
    source[weight_key] = torch.arange(37 * 4).reshape(37, 4).float()
    source[bias_key] = torch.arange(37).float()
    state_weight_key = "action_head.state_projection.net.0.weight"
    state_bias_key = "action_head.state_projection.net.0.bias"
    state_linear_key = "action_head.state_projection.net.1.weight"
    source[state_weight_key] = torch.arange(33).float()
    source[state_bias_key] = torch.arange(33).float() + 100
    source[state_linear_key] = torch.arange(4 * 33).reshape(4, 33).float()
    checkpoint = tmp_path / "old-37d.pt"
    torch.save(source, checkpoint)

    loaded, skipped = _load_compatible(model, checkpoint)

    assert loaded == len(model.state_dict())
    assert skipped == 0
    torch.testing.assert_close(
        model.state_dict()[weight_key], source[weight_key][:31]
    )
    torch.testing.assert_close(model.state_dict()[bias_key], source[bias_key][:31])
    torch.testing.assert_close(
        model.state_dict()[state_weight_key], source[state_weight_key][:31]
    )
    torch.testing.assert_close(
        model.state_dict()[state_bias_key], source[state_bias_key][:31]
    )
    torch.testing.assert_close(
        model.state_dict()[state_linear_key], source[state_linear_key][:, :31]
    )
