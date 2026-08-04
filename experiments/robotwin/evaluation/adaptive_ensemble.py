"""
adaptive_ensemble.py

"""

from collections import deque

import numpy as np


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def aligned_actions(self):
        num_actions = len(self.action_history)
        if num_actions == 0:
            raise RuntimeError("cannot align an empty action history")
        cur_action = self.action_history[-1]
        if cur_action.ndim == 1:
            return np.stack(self.action_history)
        return np.stack(
            [
                pred_actions[index]
                for index, pred_actions in zip(
                    range(num_actions - 1, -1, -1),
                    self.action_history,
                )
            ]
        )

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        curr_act_preds = self.aligned_actions()
        num_actions = len(curr_act_preds)

        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions - 1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()

        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action
