import numpy as np

from experiments.robotwin.evaluation.adaptive_ensemble import AdaptiveEnsembler


def test_aligned_actions_exposes_oldest_prediction_for_current_step():
    ensembler = AdaptiveEnsembler(pred_action_horizon=3)

    ensembler.ensemble_action(np.array([[10.0], [11.0], [12.0]]))
    np.testing.assert_array_equal(ensembler.aligned_actions(), [[10.0]])

    ensembler.ensemble_action(np.array([[20.0], [21.0], [22.0]]))
    np.testing.assert_array_equal(ensembler.aligned_actions(), [[11.0], [20.0]])

    ensembler.ensemble_action(np.array([[30.0], [31.0], [32.0]]))
    np.testing.assert_array_equal(
        ensembler.aligned_actions(),
        [[12.0], [21.0], [30.0]],
    )
