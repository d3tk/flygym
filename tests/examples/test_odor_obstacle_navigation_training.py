from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

from flygym_demo.odor_obstacle_navigation.ppo import resolve_minibatch_size
from flygym_demo.odor_obstacle_navigation.training_callbacks import (
    SHARED_ROLLOUT_DIAGNOSTIC_DEFAULTS,
    aggregate_rollout_diagnostics,
)


class ResolveMinibatchSizeTests(unittest.TestCase):
    def test_keeps_size_when_it_divides_rollout_buffer(self):
        self.assertEqual(
            resolve_minibatch_size(256, rollout_steps=256, num_envs=4),
            256,
        )

    def test_picks_largest_divisor_at_or_below_requested(self):
        self.assertEqual(
            resolve_minibatch_size(1200, rollout_steps=200, num_envs=842),
            842,
        )


class AggregateRolloutDiagnosticsTests(unittest.TestCase):
    def test_extrinsic_rollout_diagnostics_use_total_reward_and_skip_curiosity(self):
        metrics = aggregate_rollout_diagnostics(
            [
                {
                    "reward_terms": {
                        "progress": 1.0,
                        "time": -0.01,
                        "success": 0.0,
                        "total": 0.99,
                    },
                    "policy_action_label": "low_level",
                    "low_level_action": {
                        "left_force": 1.0,
                        "right_force": 1.0,
                        "groom": False,
                    },
                    "switched_action": False,
                    "forward_speed": 1.5,
                    "yaw_rate": 0.0,
                    "failure_reason": None,
                }
            ]
        )
        self.assertEqual(set(SHARED_ROLLOUT_DIAGNOSTIC_DEFAULTS), set(metrics))
        self.assertAlmostEqual(1.0, metrics["reward/progress"])
        self.assertAlmostEqual(-0.01, metrics["reward/time_penalty"])
        self.assertEqual(0.0, metrics["reward/success_bonus"])
        self.assertAlmostEqual(0.99, metrics["reward/extrinsic"])
        self.assertAlmostEqual(1.0, metrics["options/forward_fraction"])
        self.assertNotIn("reward/intrinsic", metrics)
        self.assertNotIn("curiosity/loss", metrics)

    def test_continuous_rollout_diagnostics_classify_low_level_options(self):
        metrics = aggregate_rollout_diagnostics(
            [
                {
                    "reward_terms": {"total": 0.0},
                    "policy_action_label": "low_level",
                    "low_level_action": {
                        "left_force": 0.8,
                        "right_force": 1.2,
                        "groom": False,
                    },
                    "switched_action": False,
                },
                {
                    "reward_terms": {"total": 0.0},
                    "policy_action_label": "low_level",
                    "low_level_action": {
                        "left_force": 1.2,
                        "right_force": 0.8,
                        "groom": False,
                    },
                    "switched_action": False,
                },
                {
                    "reward_terms": {"total": 0.0},
                    "policy_action_label": "low_level",
                    "low_level_action": {
                        "left_force": 1.0,
                        "right_force": 1.0,
                        "groom": False,
                    },
                    "switched_action": False,
                },
                {
                    "reward_terms": {"total": 0.0},
                    "policy_action_label": "low_level",
                    "low_level_action": {
                        "left_force": 0.0,
                        "right_force": 0.0,
                        "groom": True,
                    },
                    "switched_action": False,
                },
            ]
        )
        self.assertAlmostEqual(0.25, metrics["options/left_fraction"])
        self.assertAlmostEqual(0.25, metrics["options/right_fraction"])
        self.assertAlmostEqual(0.25, metrics["options/forward_fraction"])
        self.assertAlmostEqual(0.25, metrics["options/groom_fraction"])

    def test_success_bonus_is_logged_as_external_reward_part(self):
        metrics = aggregate_rollout_diagnostics(
            [
                {
                    "reward_terms": {
                        "progress": 0.5,
                        "time": -0.01,
                        "success": 10.0,
                        "total": 10.49,
                    },
                    "policy_action_label": "low_level",
                    "switched_action": False,
                    "forward_speed": 0.0,
                    "yaw_rate": 0.0,
                    "failure_reason": None,
                }
            ]
        )
        self.assertAlmostEqual(10.0, metrics["reward/success_bonus"])
        self.assertAlmostEqual(10.49, metrics["reward/extrinsic"])

    def test_empty_rollout_diagnostics_keep_logging_schema(self):
        metrics = aggregate_rollout_diagnostics([])
        self.assertEqual(SHARED_ROLLOUT_DIAGNOSTIC_DEFAULTS, metrics)


class TrainingLayoutPresetTests(unittest.TestCase):
    def test_v1_layout_preset_builds_v1_style_task_config(self):
        module = _load_training_script()
        args = module.parse_args(["--randomize-layout", "--layout-preset", "v1"])

        config = module._make_task_config(args)

        self.assertTrue(config.randomize_layout)
        self.assertEqual((5, 2), config.obstacle_positions.shape)
        self.assertEqual((3, 5), config.obstacle_count_range)
        np.testing.assert_allclose(config.inactive_obstacle_position, [1000.0, 0.0])
        self.assertEqual(10.0, config.layout_min_target_spawn_distance)
        self.assertEqual(3.0, config.layout_min_obstacle_spawn_distance)
        self.assertEqual(4.0, config.layout_min_obstacle_target_distance)
        self.assertEqual(4.0, config.layout_min_obstacle_obstacle_distance)
        self.assertEqual(0.0, config.spawn_xy_jitter)
        self.assertEqual(0.0, config.spawn_yaw_jitter)

    def test_layout_cli_overrides_v1_preset_defaults(self):
        module = _load_training_script()
        args = module.parse_args(
            [
                "--layout-preset",
                "v1",
                "--spawn-xy-jitter",
                "1.25",
                "--spawn-yaw-jitter",
                "0.5",
                "--obstacle-count-range",
                "4",
                "4",
            ]
        )

        config = module._make_task_config(args)

        self.assertEqual(1.25, config.spawn_xy_jitter)
        self.assertEqual(0.5, config.spawn_yaw_jitter)
        self.assertEqual((4, 4), config.obstacle_count_range)

    def test_launch_script_passes_v1_preset_when_randomizing_layout(self):
        launch_path = (
            Path(__file__).resolve().parents[2]
            / "scripts/flygym/launch_odor_obstacle_navigation_training.sh"
        )
        launch_text = launch_path.read_text(encoding="utf-8")

        self.assertIn("RANDOMIZE_LAYOUT", launch_text)
        self.assertIn('RANDOMIZE_LAYOUT="${RANDOMIZE_LAYOUT:-1}"', launch_text)
        self.assertIn("--randomize-layout", launch_text)
        self.assertIn("--layout-preset", launch_text)
        self.assertIn("LAYOUT_PRESET=\"${LAYOUT_PRESET:-v1}\"", launch_text)


def _load_training_script():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts/flygym/train_odor_obstacle_navigation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "train_odor_obstacle_navigation_test", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
