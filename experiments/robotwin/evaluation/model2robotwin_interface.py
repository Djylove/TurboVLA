import json
import os
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import cv2 as cv
import numpy as np
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

try:
    from .adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    from adaptive_ensemble import AdaptiveEnsembler
from starVLA.model.tools import read_mode_config


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        binary_action_source: str = "ensemble",
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
        binary_threshold: float = 0.49,
        exec_horizon: Optional[int] = None,
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = bool(action_ensemble)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        if binary_action_source not in {"ensemble", "latest", "oldest"}:
            raise ValueError(
                "binary_action_source must be ensemble, latest, or oldest, "
                f"got {binary_action_source!r}"
            )
        self.binary_action_source = binary_action_source
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode
        self.binary_threshold = binary_threshold

        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(
            self.unnorm_key, policy_ckpt_path=policy_ckpt_path, action_mode=action_mode
        )
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        if self.action_ensemble:
            if self.action_ensemble_horizon is None:
                self.action_ensemble_horizon = self.action_chunk_size
            self.action_ensemble_horizon = max(
                1,
                min(int(self.action_ensemble_horizon), self.action_chunk_size),
            )
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon,
                self.adaptive_ensemble_alpha,
            )
        else:
            self.action_ensembler = None
        if exec_horizon is None:
            self.exec_horizon = self.action_chunk_size
        else:
            self.exec_horizon = max(1, min(int(exec_horizon), self.action_chunk_size))
        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.model_uses_state = self.get_state_dim(policy_ckpt_path=policy_ckpt_path) > 0
        self.raw_actions = None
        self.last_normalized_actions = None
        trace_path = os.environ.get("TURBOVLA_ACTION_TRACE_JSONL")
        self.action_trace_path = Path(trace_path).expanduser().resolve() if trace_path else None
        if self.action_trace_path is not None:
            if self.action_trace_path.exists():
                raise FileExistsError(
                    f"Refusing to replace action trace: {self.action_trace_path}"
                )
            self.action_trace_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            "*** action_chunk_size: "
            f"{self.action_chunk_size}, temporal_ensemble: {self.action_ensemble}, "
            f"ensemble_history: {self.action_ensemble_horizon if self.action_ensemble else 0}, "
            f"query_interval: {1 if self.action_ensemble else self.exec_horizon}, "
            f"binary_action_source: {self.binary_action_source} ***"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        self.last_normalized_actions = None
        # Reset state tracking for delta/rel modes
        self.initial_state = None
        self.prev_action = None

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        state = example.get("state", None)
        state_for_action_mode = None
        if state is not None:
            state = np.asarray(state)
            state_for_action_mode = self._robotwin_state_to_model_order(state)
            if self.model_uses_state:
                example["state"] = self.normalize_state(
                    state_for_action_mode,
                    self.state_norm_stats,
                    normalization_mode=self.normalization_mode,
                    binary_threshold=self.binary_threshold,
                ).reshape(1, -1)

        # Store initial state for delta/rel modes
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state_for_action_mode).copy()

        task_description = example.get("lang", None)
        images = example["image"]

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)
                # Re-store initial state after reset if in delta/rel mode
                if self.action_mode in ["delta", "rel"] and state is not None:
                    self.initial_state = np.array(state_for_action_mode).copy()

        images = [self._resize_image(image) for image in images]
        example["image"] = images
        example_copy = example.copy()
        if not self.model_uses_state:
            example_copy.pop("state", None)
        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        # Temporal ensemble needs a fresh, overlapping chunk at every environment step.
        queried_policy = (
            self.action_ensemble or step % self.exec_horizon == 0 or self.raw_actions is None
        )
        if queried_policy:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")

            normalized_actions = normalized_actions[0]
            self.last_normalized_actions = np.asarray(normalized_actions)
            # Unnormalize to get delta/rel values
            raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
                normalization_mode=self.normalization_mode,
                binary_threshold=self.binary_threshold,
            )

            # Convert delta/rel to absolute actions
            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state_for_action_mode)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions

        if self.action_ensemble:
            current_action = self.action_ensembler.ensemble_action(self.raw_actions)
            continuous_mask = np.asarray(
                self.action_norm_stats.get(
                    "mask",
                    np.ones_like(current_action, dtype=bool),
                ),
                dtype=bool,
            )
            if continuous_mask.shape != current_action.shape:
                raise ValueError(
                    "Action normalization mask shape does not match the ensembled action: "
                    f"{continuous_mask.shape} != {current_action.shape}"
                )
            if self.binary_action_source == "latest":
                binary_action = self.raw_actions[0]
            elif self.binary_action_source == "oldest":
                binary_action = self.action_ensembler.aligned_actions()[0]
            else:
                binary_action = current_action
            current_action = np.where(
                continuous_mask,
                current_action,
                (binary_action >= 0.5).astype(current_action.dtype),
            )
        else:
            action_idx = step % self.exec_horizon
            if action_idx >= len(self.raw_actions):
                action_idx = len(self.raw_actions) - 1
            current_action = self.raw_actions[action_idx]

        # Update prev_action for delta mode (for cross-chunk continuity)
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        action_model_order = np.asarray(current_action).copy()
        action_robotwin_order = action_model_order[
            [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
        ]
        self._trace_action(
            step=step,
            queried_policy=queried_policy,
            state_model_order=state_for_action_mode,
            action_model_order=action_model_order,
            action_robotwin_order=action_robotwin_order,
        )
        return action_robotwin_order

    def _trace_action(
        self,
        *,
        step: int,
        queried_policy: bool,
        state_model_order: np.ndarray | None,
        action_model_order: np.ndarray,
        action_robotwin_order: np.ndarray,
    ) -> None:
        if self.action_trace_path is None:
            return
        if self.last_normalized_actions is None or self.raw_actions is None:
            raise RuntimeError("action trace requested before the first policy response")
        normalized = np.asarray(self.last_normalized_actions, dtype=np.float64)
        raw_actions = np.asarray(self.raw_actions, dtype=np.float64)
        continuous_mask = np.asarray(
            self.action_norm_stats.get(
                "mask",
                np.ones(action_model_order.shape, dtype=bool),
            ),
            dtype=bool,
        )
        state_array = (
            None
            if state_model_order is None
            else np.asarray(state_model_order, dtype=np.float64)
        )
        action_state_l2 = None
        if state_array is not None:
            action_state_l2 = float(
                np.linalg.norm(action_model_order[continuous_mask] - state_array[continuous_mask])
            )
        record = {
            "schema_version": "turbovla.robotwin_action_trace.v1",
            "step": int(step),
            "task_description": self.task_description,
            "queried_policy": bool(queried_policy),
            "normalized_saturation_fraction": float(np.mean(np.abs(normalized) >= 0.999)),
            "normalized_continuous_saturation_fraction": float(
                np.mean(np.abs(normalized[:, continuous_mask]) >= 0.999)
            ),
            "normalized_chunk_min": normalized.min(axis=0).tolist(),
            "normalized_chunk_max": normalized.max(axis=0).tolist(),
            "normalized_first_model_order": normalized[0].tolist(),
            "raw_chunk_min_model_order": raw_actions.min(axis=0).tolist(),
            "raw_chunk_max_model_order": raw_actions.max(axis=0).tolist(),
            "selected_action_model_order": action_model_order.tolist(),
            "selected_action_robotwin_order": action_robotwin_order.tolist(),
            "state_model_order": None if state_array is None else state_array.tolist(),
            "action_state_l2": action_state_l2,
        }
        with self.action_trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")

    @staticmethod
    def _robotwin_state_to_model_order(state: np.ndarray) -> np.ndarray:
        # RoboTwin provides [left_joints, left_gripper, right_joints, right_gripper].
        # Training uses [left_joints, right_joints, left_gripper, right_gripper].
        return state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]

    @staticmethod
    def normalize_state(
        state: dict[str, np.ndarray],
        state_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
        binary_threshold: float = 0.49,
    ) -> dict[str, np.ndarray]:
        """
        Normalize the state
        """
        continuous_mask = [True, True, True, True, True, True, True, True, True, True, True, True, False, False]
        continuous_mask = np.array(continuous_mask, dtype=bool)
        state_high, state_low = ModelClient._get_normalization_bounds(
            state_norm_stats, normalization_mode=normalization_mode
        )
        valid_mask = continuous_mask & (state_high != state_low)
        normalized_state = np.where(
            valid_mask,
            (state - state_low) / (state_high - state_low) * 2 - 1,
            state,
        )
        normalized_state = np.where(
            ~continuous_mask,
            (normalized_state > binary_threshold).astype(normalized_state.dtype),
            normalized_state,
        )
        return normalized_state

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
        binary_threshold: float = 0.49,
    ) -> np.ndarray:
        action_high, action_low = ModelClient._get_normalization_bounds(
            action_norm_stats, normalization_mode=normalization_mode
        )
        mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))
        normalized_actions = np.clip(normalized_actions, -1, 1)

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            (normalized_actions > binary_threshold).astype(normalized_actions.dtype),
        )

        return actions

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """
        Convert delta actions to absolute actions.

        Training: delta[0] = a[0] - s[0], delta[t] = a[t] - a[t-1]
        Deployment: a[0] = delta[0] + base, a[t] = delta[t] + a[t-1]

        Where base is:
        - First chunk: initial_state (s_0)
        - Subsequent chunks: prev_action (last action from previous chunk)
        """
        abs_actions = np.zeros_like(delta_actions)
        mask = self.action_norm_stats.get("mask", np.ones(delta_actions.shape[-1], dtype=bool))

        # Determine base action
        base = self.prev_action if self.prev_action is not None else self.initial_state

        for i in range(len(delta_actions)):
            abs_actions[i] = np.where(mask, delta_actions[i] + base, delta_actions[i])
            base = abs_actions[i]

        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        """
        Convert relative actions to absolute actions.

        Training: rel[t] = a[t] - s[0]
        Deployment: a[t] = rel[t] + s[0]
        """
        abs_actions = np.zeros_like(rel_actions)
        mask = self.action_norm_stats.get("mask", np.ones(rel_actions.shape[-1], dtype=bool))

        for i in range(len(rel_actions)):
            abs_actions[i] = np.where(mask, rel_actions[i] + self.initial_state, rel_actions[i])

        return abs_actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path, action_mode: str = "abs") -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)

        stats = norm_stats[unnorm_key]

        # Support two formats:
        # New format: {"robotwin": {"abs": {...}, "delta": {...}, "rel": {...}}}
        # Old format: {"robotwin": {"action": {...}, "state": {...}}}

        if action_mode in stats:
            # New format: directly use the corresponding mode stats
            mode_stats = stats[action_mode]
            return mode_stats.get("action", mode_stats)
        if "action" in stats:
            # Old format: only supports abs mode
            if action_mode != "abs":
                raise ValueError(
                    f"Statistics key `{unnorm_key}` only provides `abs` action stats, "
                    f"but action_mode=`{action_mode}` was requested."
                )
            return stats["action"]
        raise ValueError(
            f"Invalid statistics file format for key `{unnorm_key}`. "
            f"Available top-level keys: {sorted(stats.keys())}"
        )

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        framework_cfg = model_config["framework"]
        action_cfg = framework_cfg.get("action", framework_cfg.get("action_model", {}))
        if "horizon" in action_cfg:
            return int(action_cfg["horizon"])
        if "action_horizon" in action_cfg:
            return int(action_cfg["action_horizon"])
        return int(action_cfg["future_action_window_size"]) + 1

    @staticmethod
    def get_state_dim(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        framework_cfg = model_config["framework"]
        action_cfg = framework_cfg.get("action", framework_cfg.get("action_model", {}))
        return int(action_cfg.get("state_dim", 0) or 0)

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        available_keys = sorted(norm_stats.keys())
        if unnorm_key is None:
            if len(available_keys) == 1:
                return available_keys[0]
            raise ValueError(
                "`unnorm_key` must be provided when multiple normalization statistics are available. "
                f"Available keys: {available_keys}"
            )

        if unnorm_key not in norm_stats:
            raise KeyError(
                f"Unknown `unnorm_key`: `{unnorm_key}`. Available keys: {available_keys}"
            )

        return unnorm_key

    @staticmethod
    def _get_normalization_bounds(
        norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> tuple[np.ndarray, np.ndarray]:
        if normalization_mode == "q99":
            if "q01" not in norm_stats or "q99" not in norm_stats:
                raise KeyError(
                    "Normalization mode `q99` requires statistics keys `q01` and `q99`."
                )
            return np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
        if normalization_mode == "min_max":
            if "min" not in norm_stats or "max" not in norm_stats:
                raise KeyError(
                    "Normalization mode `min_max` requires statistics keys `min` and `max`."
                )
            return np.array(norm_stats["max"]), np.array(norm_stats["min"])
        raise ValueError(
            f"Unsupported normalization_mode: {normalization_mode}. Expected one of ['min_max', 'q99']."
        )


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get(
        "action_normalization_mode",
        usr_args.get("normalization_mode", "min_max"),
    )
    binary_threshold = float(usr_args.get("binary_threshold", 0.49))
    exec_horizon = usr_args.get("exec_horizon", None)
    exec_horizon = None if exec_horizon is None else int(exec_horizon)
    action_ensemble = bool(usr_args.get("action_ensemble", False))
    action_ensemble_override = os.environ.get("TURBOVLA_ACTION_ENSEMBLE")
    if action_ensemble_override is not None:
        normalized_override = action_ensemble_override.strip().lower()
        if normalized_override not in {"true", "false"}:
            raise ValueError("TURBOVLA_ACTION_ENSEMBLE must be true or false")
        action_ensemble = normalized_override == "true"
    exec_horizon_override = os.environ.get("TURBOVLA_EXEC_HORIZON")
    if exec_horizon_override is not None:
        exec_horizon = int(exec_horizon_override)
        if exec_horizon < 1:
            raise ValueError("TURBOVLA_EXEC_HORIZON must be positive")
    action_ensemble_horizon = usr_args.get("action_ensemble_horizon", None)
    action_ensemble_horizon = (
        None if action_ensemble_horizon is None else int(action_ensemble_horizon)
    )
    adaptive_ensemble_alpha = float(usr_args.get("adaptive_ensemble_alpha", 0.1))
    binary_action_source = str(usr_args.get("binary_action_source", "ensemble"))
    binary_action_source = os.environ.get(
        "TURBOVLA_BINARY_ACTION_SOURCE",
        binary_action_source,
    )

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
        binary_threshold=binary_threshold,
        exec_horizon=exec_horizon,
        action_ensemble=action_ensemble,
        action_ensemble_horizon=action_ensemble_horizon,
        adaptive_ensemble_alpha=adaptive_ensemble_alpha,
        binary_action_source=binary_action_source,
    )


def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,  # Required for delta/rel action modes
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action)
