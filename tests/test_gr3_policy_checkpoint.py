from pathlib import Path

import pytest
import torch

from turbovla.evaluation.gr3_policy import _load_gr3_checkpoint


def test_loads_complete_training_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "model_final.pt"
    payload = {"model_state_dict": {"weight": torch.tensor([1.0])}, "step": 10}
    torch.save(payload, checkpoint)

    loaded, metadata_path = _load_gr3_checkpoint(checkpoint)

    assert loaded["step"] == 10
    assert metadata_path is None


def test_combines_raw_step_weights_with_final_metadata(tmp_path: Path):
    metadata_path = tmp_path / "model_final.pt"
    step_path = tmp_path / "model_step_10000.pt"
    torch.save(
        {"model_state_dict": {"weight": torch.tensor([1.0])}, "step": 15000},
        metadata_path,
    )
    torch.save({"weight": torch.tensor([2.0])}, step_path)

    loaded, source = _load_gr3_checkpoint(step_path)

    assert loaded["step"] == 15000
    assert torch.equal(loaded["model_state_dict"]["weight"], torch.tensor([2.0]))
    assert source == metadata_path


def test_rejects_raw_weights_without_final_metadata(tmp_path: Path):
    checkpoint = tmp_path / "model_step_10000.pt"
    torch.save({"weight": torch.tensor([2.0])}, checkpoint)

    with pytest.raises(ValueError, match="require model_final.pt"):
        _load_gr3_checkpoint(checkpoint)
