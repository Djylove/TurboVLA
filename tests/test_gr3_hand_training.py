import numpy as np
import pytest
import torch

from experiments.gr3.evaluate_anygrasp import _hand_metric_sums
from experiments.gr3.train import _action_axis_weights, _weighted_masked_l1
from turbovla.data.gr3_common import Gr3NormalizationStats


def _stats() -> Gr3NormalizationStats:
    action_low = np.full(37, -0.001, dtype=np.float32)
    action_high = np.full(37, 0.001, dtype=np.float32)
    action_low[19:24] = -1.0
    action_high[19:24] = 1.0
    return Gr3NormalizationStats(
        state_mean=np.zeros(33),
        state_std=np.ones(33),
        action_low=action_low,
        action_high=action_high,
    )


def test_hand_axis_weights_only_select_data_active_hand_joints():
    weights, active_axes = _action_axis_weights(
        _stats(),
        hand_loss_weight=5.0,
        hand_axis_range_threshold=0.05,
    )

    assert active_axes == (19, 20, 21, 22, 23)
    assert weights[19:24].tolist() == [5.0] * 5
    assert weights[24:].tolist() == [1.0] * 7


def test_weighted_masked_l1_emphasizes_active_hand_axis():
    weights, _ = _action_axis_weights(
        _stats(),
        hand_loss_weight=5.0,
        hand_axis_range_threshold=0.05,
    )
    prediction = torch.zeros(1, 2, 31)
    target = torch.zeros_like(prediction)
    target[0, 0, 19] = 1.0
    target[0, 1, 0] = 100.0
    mask = torch.tensor([[1.0, 0.0]])

    loss = _weighted_masked_l1(prediction, target, mask, weights)

    assert float(loss) == pytest.approx(5.0 / weights.sum().item())


def test_hand_metrics_report_missed_closure_in_raw_joint_space():
    stats = _stats()
    target_raw = np.zeros((1, 2, 31), dtype=np.float32)
    target_raw[0, 0, 19:23] = -1.0
    prediction_raw = np.zeros_like(target_raw)
    target = stats.normalize_action(target_raw)
    prediction = stats.normalize_action(prediction_raw)

    metrics = _hand_metric_sums(
        prediction,
        target,
        np.ones((1, 2), dtype=np.float32),
        stats,
        active_hand_axes=(19, 20, 21, 22, 23),
    )

    assert metrics["false_negative"] == 1.0
    assert metrics["true_negative"] == 1.0
    assert metrics["true_positive"] == 0.0
    assert metrics["finger_aperture_absolute_error"] == pytest.approx(1.0)
