from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
wp = pytest.importorskip("warp")

pytestmark = pytest.mark.skipif(
    not any(device.is_cuda for device in wp.get_devices()),
    reason="Warp CUDA device is not available.",
)

from flygym_demo.odor_obstacle_navigation.task import NavigationTaskConfig  # noqa: E402
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
