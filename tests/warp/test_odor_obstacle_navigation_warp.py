from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
wp = pytest.importorskip("warp")

pytestmark = pytest.mark.skipif(
    not any(device.is_cuda for device in wp.get_devices()),
    reason="Warp CUDA device is not available.",
)

from flygym_demo.odor_obstacle_navigation.task import (  # noqa: E402
    NavigationTaskConfig,
    _odor_by_side,
    _odor_intensity_from_sources,
)
from flygym_demo.odor_obstacle_navigation.warp_runner import (  # noqa: E402
    WarpNavigationTaskRunner,
)


def _fast_config(**kwargs) -> NavigationTaskConfig:
    defaults = {
        "decision_steps": 1,
        "episode_time": 2e-4,
        "enable_vision": True,
        "enable_olfaction": False,
        "render": False,
        "warmup_seconds": 0.0,
        "obstacle_positions": np.array([[4.0, 0.0]], dtype=float),
    }
    defaults.update(kwargs)
    return NavigationTaskConfig(**defaults)


def _random_layout_config(**kwargs) -> NavigationTaskConfig:
    defaults = {
        "enable_vision": False,
        "enable_olfaction": False,
        "randomize_layout": True,
        "spawn_xy_jitter": 0.0,
        "spawn_yaw_jitter": 0.0,
        "obstacle_positions": np.zeros((5, 2), dtype=float),
        "obstacle_count_range": (3, 5),
        "layout_min_target_spawn_distance": 10.0,
        "layout_min_obstacle_spawn_distance": 3.0,
        "layout_min_obstacle_target_distance": 4.0,
        "layout_min_obstacle_obstacle_distance": 4.0,
    }
    defaults.update(kwargs)
    return _fast_config(**defaults)


def _assert_vision_feature_ranges(features: dict) -> None:
    assert features["com_per_eye"].shape == (2, 2)
    assert features["size_per_eye"].shape == (2,)
    assert features["center_deviation"].shape == (2,)
    assert features["object_mask_counts"].shape == (2,)
    assert np.all(features["size_per_eye"] >= 0.0)
    assert np.all(features["size_per_eye"] <= 1.0)
    assert np.all(features["center_deviation"] >= 0.0)
    assert np.all(features["center_deviation"] <= 1.0)
    assert np.all(features["object_mask_counts"] >= 0)
    assert -1.0 <= features["object_azimuth"] <= 1.0
    assert 0.0 <= features["object_size"] <= 1.0
    assert 0.0 <= features["object_proximity_proxy"] <= 1.0
    assert features["avoidance_direction"] in (-1.0, 1.0)
    assert 0.0 <= features["collision_risk"] <= 1.0
    assert 0.0 <= features["looming_signal"] <= 1.0


def test_warp_runner_supports_gpu_vision_observations():
    runner = WarpNavigationTaskRunner(
        _fast_config(enable_vision=True, ccd_iterations=321),
        num_envs=2,
    )
    try:
        assert runner.sim.mj_model.opt.ccd_iterations == 321
        obs, infos = runner.reset(seed=123)
        assert obs.shape == (2, 8)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        for info in infos:
            _assert_vision_feature_ranges(info["processed_features"])

        obs, rewards, terminated, truncated, infos = runner.step(
            np.zeros((2, 2), dtype=np.float32)
        )
        assert obs.shape == (2, 8)
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
        assert np.all(np.isfinite(obs))
        for info in infos:
            _assert_vision_feature_ranges(info["processed_features"])
    finally:
        runner.close()


def test_warp_runner_no_vision_keeps_vision_fields_zero():
    runner = WarpNavigationTaskRunner(_fast_config(enable_vision=False), num_envs=2)
    try:
        obs, infos = runner.reset(seed=456)
        assert obs.shape == (2, 8)
        np.testing.assert_allclose(obs[:, 2:], 0.0)
        for info in infos:
            features = info["processed_features"]
            np.testing.assert_array_equal(features["size_per_eye"], np.zeros(2))
            np.testing.assert_array_equal(
                features["center_deviation"], np.zeros(2)
            )
            np.testing.assert_array_equal(
                features["object_mask_counts"], np.zeros(2, dtype=np.int32)
            )
            assert features["object_size"] == 0.0
            assert features["collision_risk"] == 0.0
    finally:
        runner.close()


def test_warp_runner_randomized_layout_resets_are_seeded():
    runner = WarpNavigationTaskRunner(_random_layout_config(), num_envs=2)
    try:
        _, infos_a = runner.reset(seed=789)
        _, infos_b = runner.reset(seed=789)

        for info_a, info_b in zip(infos_a, infos_b, strict=True):
            np.testing.assert_allclose(
                info_a["target_position"],
                info_b["target_position"],
            )
            np.testing.assert_allclose(
                info_a["obstacle_positions"],
                info_b["obstacle_positions"],
            )
            assert info_a["active_obstacle_count"] == info_b["active_obstacle_count"]
            np.testing.assert_allclose(
                info_a["active_obstacle_positions"],
                info_b["active_obstacle_positions"],
            )
    finally:
        runner.close()


def test_warp_runner_reset_at_only_changes_selected_layout_rows():
    runner = WarpNavigationTaskRunner(_random_layout_config(), num_envs=3)
    try:
        runner.reset(seed=100)
        odor_before = runner.current_odor_source.copy()
        obstacles_before = runner.current_obstacle_positions.copy()
        active_before = runner.current_active_obstacle_counts.copy()

        runner.reset_at([1], seeds=[999])

        np.testing.assert_allclose(runner.current_odor_source[0], odor_before[0])
        np.testing.assert_allclose(runner.current_odor_source[2], odor_before[2])
        np.testing.assert_allclose(
            runner.current_obstacle_positions[0],
            obstacles_before[0],
        )
        np.testing.assert_allclose(
            runner.current_obstacle_positions[2],
            obstacles_before[2],
        )
        assert runner.current_active_obstacle_counts[0] == active_before[0]
        assert runner.current_active_obstacle_counts[2] == active_before[2]
        assert not np.allclose(runner.current_odor_source[1], odor_before[1])
    finally:
        runner.close()


def test_warp_runner_mocap_positions_match_per_env_layouts():
    runner = WarpNavigationTaskRunner(_random_layout_config(), num_envs=2)
    try:
        runner.reset(seed=200)
        mocap_pos = runner.sim.mjw_data.mocap_pos.numpy()
        obstacle_z = runner.config.obstacle_height / 2

        for env_idx in range(runner.num_envs):
            for source_idx, mocap_id in enumerate(runner._odor_mocap_ids):
                np.testing.assert_allclose(
                    mocap_pos[env_idx, mocap_id],
                    runner.current_odor_source[env_idx, source_idx],
                    atol=1e-5,
                )
            for obstacle_idx, mocap_id in enumerate(runner._obstacle_mocap_ids):
                expected = np.array(
                    [
                        runner.current_obstacle_positions[env_idx, obstacle_idx, 0],
                        runner.current_obstacle_positions[env_idx, obstacle_idx, 1],
                        obstacle_z,
                    ],
                    dtype=np.float32,
                )
                np.testing.assert_allclose(
                    mocap_pos[env_idx, mocap_id],
                    expected,
                    atol=1e-5,
                )
    finally:
        runner.close()


def test_warp_runner_odor_observations_use_each_env_target():
    runner = WarpNavigationTaskRunner(
        _random_layout_config(enable_olfaction=True),
        num_envs=2,
    )
    try:
        obs, infos = runner.reset(seed=300)
        site_positions = runner.sim.mjw_data.site_xpos.numpy()

        for env_idx, info in enumerate(infos):
            sensor_positions = site_positions[
                env_idx,
                runner._olfaction_site_ids,
                :,
            ]
            expected = _odor_by_side(
                _odor_intensity_from_sources(
                    sensor_positions,
                    runner.current_odor_source[env_idx],
                    runner.config.peak_odor_intensity,
                ),
                runner.config.odor_palps_antennae_weights,
            )
            np.testing.assert_allclose(obs[env_idx, :2], expected, rtol=1e-5)
            np.testing.assert_allclose(
                [
                    info["processed_features"]["odor_left"],
                    info["processed_features"]["odor_right"],
                ],
                expected,
                rtol=1e-5,
            )
    finally:
        runner.close()
