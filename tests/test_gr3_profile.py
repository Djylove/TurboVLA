import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from turbovla.data.gr3_dagger import (
    Gr3NormalizationStats,
    load_gr3_manifest,
    prepare_gr3_rgb,
)
from turbovla.data.gr3_common import (
    GR3_MODEL_ACTION_DIM,
    GR3_MODEL_STATE_DIM,
    canonicalize_gr3_action,
)


class Gr3ProfileTest(unittest.TestCase):
    def test_image_preparation_is_explicit_rgb_center_crop(self):
        image = np.zeros((40, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 7
        prepared = prepare_gr3_rgb(image, 24)
        self.assertEqual(prepared.shape, (24, 24, 3))
        self.assertEqual(prepared.dtype, np.uint8)
        self.assertTrue(np.all(prepared[:, :, 0] == 7))

    def test_normalization_round_trip(self):
        stats = Gr3NormalizationStats(
            state_mean=np.zeros(33),
            state_std=np.ones(33),
            action_low=np.full(37, -2.0),
            action_high=np.full(37, 2.0),
        )
        action = np.linspace(-2.0, 2.0, 37, dtype=np.float32)
        restored = stats.denormalize_action(stats.normalize_action(action))
        np.testing.assert_allclose(restored, action, atol=1e-6)
        loaded = Gr3NormalizationStats.from_dict(stats.to_dict())
        np.testing.assert_array_equal(loaded.state_std, stats.state_std)

    def test_model_state_normalization_uses_only_joint_axes(self):
        stats = Gr3NormalizationStats(
            state_mean=np.arange(33, dtype=np.float32),
            state_std=np.ones(33),
            action_low=np.full(37, -1.0),
            action_high=np.full(37, 1.0),
        )
        state = np.arange(GR3_MODEL_STATE_DIM, dtype=np.float32)
        np.testing.assert_array_equal(stats.normalize_state(state), 0.0)

    def test_robust_state_normalization_floors_scale_and_clips_ood_values(self):
        stats = Gr3NormalizationStats(
            state_mean=np.zeros(33),
            state_std=np.full(33, 0.001),
            action_low=np.full(37, -1.0),
            action_high=np.full(37, 1.0),
        ).with_state_robustness(std_floor=0.1, clip_z=1.5)

        normalized = stats.normalize_state(np.full(31, 0.2, dtype=np.float32))

        np.testing.assert_allclose(normalized, 1.5)
        restored = Gr3NormalizationStats.from_dict(stats.to_dict())
        self.assertEqual(restored.state_std_floor, 0.1)
        self.assertEqual(restored.state_clip_z, 1.5)

    def test_old_normalization_payload_keeps_backward_compatible_state_transform(self):
        stats = Gr3NormalizationStats.from_dict(
            {
                "state_mean": [0.0] * 33,
                "state_std": [1.0] * 33,
                "action_low": [-1.0] * 37,
                "action_high": [1.0] * 37,
            }
        )

        self.assertEqual(stats.state_std_floor, 0.0)
        self.assertIsNone(stats.state_clip_z)

    def test_model_action_normalization_uses_first_31_canonical_axes(self):
        stats = Gr3NormalizationStats(
            state_mean=np.zeros(33),
            state_std=np.ones(33),
            action_low=np.arange(37, dtype=np.float32) - 10.0,
            action_high=np.arange(37, dtype=np.float32) + 10.0,
        )
        action = np.arange(GR3_MODEL_ACTION_DIM, dtype=np.float32)[None]
        restored = stats.denormalize_action(stats.normalize_action(action))
        np.testing.assert_allclose(restored, action, atol=1e-6)

    def test_model_action_expands_to_canonical_with_zero_planar_base(self):
        learned = np.arange(2 * GR3_MODEL_ACTION_DIM, dtype=np.float32).reshape(
            2, GR3_MODEL_ACTION_DIM
        )
        canonical = canonicalize_gr3_action(learned)
        self.assertEqual(canonical.shape, (2, 37))
        np.testing.assert_array_equal(canonical[:, :31], learned)
        np.testing.assert_array_equal(canonical[:, 31:], 0.0)

    def test_manifest_rejects_non_gr3_profile(self):
        payload = {
            "schema_version": "xpolicy_dataset.v1",
            "profile_id": "vendor.other_robot",
            "episodes": [{"train_eligible_after_filters": True}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "GR3 requires profile_id"):
                load_gr3_manifest(path)


if __name__ == "__main__":
    unittest.main()
