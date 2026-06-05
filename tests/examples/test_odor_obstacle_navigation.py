from __future__ import annotations

import importlib.util
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import mujoco as mj

from flygym.anatomy import ContactBodiesPreset
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly
from flygym_demo.odor_obstacle_navigation.task import (
    ACTION_PRESETS,
    NavigationTaskConfig,
    NavigationTaskFactory,
    OdorObstacleNavigationTask,
    _odor_intensity_from_sources,
    resolve_navigation_action,
)
from flygym_demo.odor_obstacle_navigation.world import ObstacleOdorWorld
from flygym_demo.odor_obstacle_navigation.runners import (
    SubprocTaskRunner,
    SubprocWorkerError,
    SyncTaskRunner,
)


def _fast_config(**kwargs) -> NavigationTaskConfig:
    defaults = {
        "decision_steps": 1,
        "episode_time": 2e-4,
        "enable_vision": False,
        "warmup_seconds": 0.0,
        "obstacle_positions": np.zeros((0, 2), dtype=float),
    }
    defaults.update(kwargs)
    return NavigationTaskConfig(**defaults)


def _assert_layout_clearances(info: dict, config: NavigationTaskConfig) -> None:
    spawn_xy = config.spawn_position[:2]
    target_xy = info["target_position"][:2]
    active_obstacles = info["active_obstacle_positions"]

    assert (
        np.linalg.norm(target_xy - spawn_xy)
        >= config.layout_min_target_spawn_distance
    )
    if active_obstacles.size == 0:
        return

    assert np.all(
        np.linalg.norm(active_obstacles - spawn_xy, axis=1)
        >= config.layout_min_obstacle_spawn_distance
    )
    assert np.all(
        np.linalg.norm(active_obstacles - target_xy, axis=1)
        >= config.layout_min_obstacle_target_distance
    )
    for obstacle_idx in range(active_obstacles.shape[0] - 1):
        assert np.all(
            np.linalg.norm(
                active_obstacles[obstacle_idx + 1 :]
                - active_obstacles[obstacle_idx],
                axis=1,
            )
            >= config.layout_min_obstacle_obstacle_distance
        )


def test_obstacle_world_adds_contact_pairs_for_obstacles():
    world = ObstacleOdorWorld(obstacle_positions=np.array([[4.0, 0.0]], dtype=float))
    fly = make_locomotion_fly(name="obstacle_contact_fly")
    world.add_fly(
        fly,
        [0, 0, 0.8],
        Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False,
    )
    model, _ = world.compile()

    obstacle_geom_id = mj.mj_name2id(
        model,
        mj.mjtObj.mjOBJ_GEOM,
        world.obstacle_geoms[0].full_identifier,
    )
    paired_geom_ids = set(model.pair_geom1) | set(model.pair_geom2)
    assert obstacle_geom_id in paired_geom_ids


def test_task_reset_returns_flat_observation():
    task = OdorObstacleNavigationTask(
        _fast_config(enable_vision=True, enable_olfaction=True)
    )
    try:
        obs, info = task.reset(seed=0)
        assert obs.shape == (8,)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        assert "processed_features" in info
    finally:
        task.close()


def test_render_task_colorizes_fly_and_adds_navigation_cameras():
    task = OdorObstacleNavigationTask(
        _fast_config(render=True, camera_res=(32, 32), output_fps=10)
    )
    try:
        assert task.renderer is not None
        assert task.camera.name == "navigation_birdeye"
        assert [camera.name for camera in task.cameras] == [
            "navigation_birdeye",
            "navigation_follow",
        ]
        assert task.fly.mjcf_root.asset.find("material", "headthorax") is not None
        assert any(geom.material is not None for geom in task.fly.mjcf_root.find_all("geom"))
    finally:
        task.close()


def test_task_passes_sb3_env_check():
    from stable_baselines3.common.env_checker import check_env

    task = OdorObstacleNavigationTask(_fast_config())
    try:
        check_env(task, warn=False, skip_render_check=True)
    finally:
        task.close()


def test_progress_reward_sign():
    task = OdorObstacleNavigationTask(_fast_config(enable_olfaction=False))
    try:
        task._prev_distance_to_target = 5.0
        reward_closer, terms_closer = task._compute_reward(4.0, success=False)
        reward_farther, terms_farther = task._compute_reward(6.0, success=False)

        assert reward_closer > 0
        assert terms_closer["progress"] > 0
        assert terms_closer["time"] == -task.config.time_penalty
        assert reward_farther < 0
        assert terms_farther["progress"] < 0
        assert terms_farther["time"] == -task.config.time_penalty
    finally:
        task.close()


def test_success_terminates_episode():
    task = OdorObstacleNavigationTask(
        _fast_config(
            enable_olfaction=False,
            target_radius=100.0,
        )
    )
    try:
        task.reset(seed=1)
        _, reward, terminated, truncated, info = task.step(np.array([0.0, 0.0]))

        assert terminated
        assert not truncated
        assert info["success"]
        assert info["reward_terms"]["success"] == task.config.success_bonus
        assert info["reward_terms"]["time"] == -task.config.time_penalty
        assert reward == info["reward_terms"]["total"]
    finally:
        task.close()


def test_discrete_actions_resolve_to_descending_drive_presets():
    np.testing.assert_allclose(
        resolve_navigation_action(0, action_space="discrete"),
        ACTION_PRESETS["forward"],
    )
    np.testing.assert_allclose(
        resolve_navigation_action(np.array(1), action_space="discrete"),
        ACTION_PRESETS["left"],
    )
    np.testing.assert_allclose(
        resolve_navigation_action(np.array([2]), action_space="discrete"),
        ACTION_PRESETS["right"],
    )


def test_continuous_actions_are_positive_and_norm_limited():
    action = resolve_navigation_action(
        np.array([-1.0, 3.0], dtype=np.float32),
        action_space="continuous",
        action_low=np.array([1e-6, 1e-6], dtype=np.float32),
        action_high=np.array([2.0, 2.0], dtype=np.float32),
        action_max_norm=2.0,
    )

    assert np.all(action > 0)
    assert np.linalg.norm(action) <= 2.0 + 1e-6

    action = resolve_navigation_action(
        np.array([2.0, 2.0], dtype=np.float32),
        action_space="continuous",
        action_low=np.array([1e-6, 1e-6], dtype=np.float32),
        action_high=np.array([2.0, 2.0], dtype=np.float32),
        action_max_norm=2.0,
    )
    np.testing.assert_allclose(np.linalg.norm(action), 2.0, rtol=1e-6)


def test_time_limit_truncates_episode():
    task = OdorObstacleNavigationTask(
        _fast_config(
            enable_olfaction=False,
            episode_time=1e-4,
            target_radius=0.01,
        )
    )
    try:
        task.reset(seed=2)
        _, _, terminated, truncated, info = task.step(np.array([0.0, 0.0]))

        assert not terminated
        assert truncated
        assert info["truncated"]
    finally:
        task.close()


def test_seeded_layouts_are_deterministic():
    config = _fast_config(
        enable_olfaction=False,
        randomize_layout=True,
        obstacle_positions=np.array([[7.5, 0.0], [12.5, 5.0]], dtype=float),
    )
    task = OdorObstacleNavigationTask(config)
    try:
        _, info_a = task.reset(seed=42)
        _, info_b = task.reset(seed=42)

        np.testing.assert_allclose(info_a["fly_position"], info_b["fly_position"])
        np.testing.assert_allclose(info_a["target_position"], info_b["target_position"])
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
        task.close()


def test_default_randomized_layout_keeps_all_obstacle_slots_active():
    config = _fast_config(
        enable_olfaction=False,
        randomize_layout=True,
        obstacle_positions=np.array([[7.5, 0.0], [12.5, 5.0]], dtype=float),
    )
    task = OdorObstacleNavigationTask(config)
    try:
        _, info = task.reset(seed=11)

        assert info["active_obstacle_count"] == 2
        np.testing.assert_allclose(
            info["active_obstacle_positions"],
            info["obstacle_positions"],
        )
    finally:
        task.close()


def test_count_range_layout_parks_inactive_obstacle_slots():
    inactive_position = np.array([1000.0, 0.0], dtype=float)
    config = _fast_config(
        enable_olfaction=False,
        randomize_layout=True,
        spawn_xy_jitter=0.0,
        spawn_yaw_jitter=0.0,
        obstacle_positions=np.zeros((5, 2), dtype=float),
        obstacle_count_range=(3, 3),
        inactive_obstacle_position=inactive_position,
        layout_min_target_spawn_distance=10.0,
        layout_min_obstacle_spawn_distance=3.0,
        layout_min_obstacle_target_distance=4.0,
        layout_min_obstacle_obstacle_distance=4.0,
    )
    task = OdorObstacleNavigationTask(config)
    try:
        _, info = task.reset(seed=12)

        assert info["active_obstacle_count"] == 3
        assert info["active_obstacle_positions"].shape == (3, 2)
        np.testing.assert_allclose(
            info["obstacle_positions"][3:],
            np.repeat(inactive_position[np.newaxis, :], 2, axis=0),
        )
        _assert_layout_clearances(info, config)
    finally:
        task.close()


def test_count_range_layout_samples_within_inclusive_range():
    config = _fast_config(
        enable_olfaction=False,
        randomize_layout=True,
        spawn_xy_jitter=0.0,
        spawn_yaw_jitter=0.0,
        obstacle_positions=np.zeros((5, 2), dtype=float),
        obstacle_count_range=(3, 5),
        layout_min_target_spawn_distance=10.0,
        layout_min_obstacle_spawn_distance=3.0,
        layout_min_obstacle_target_distance=4.0,
        layout_min_obstacle_obstacle_distance=4.0,
    )
    task = OdorObstacleNavigationTask(config)
    try:
        for seed in range(20, 26):
            _, info = task.reset(seed=seed)

            assert 3 <= info["active_obstacle_count"] <= 5
            assert (
                info["active_obstacle_positions"].shape[0]
                == info["active_obstacle_count"]
            )
            _assert_layout_clearances(info, config)
    finally:
        task.close()


def test_odor_intensity_helper_matches_obstacle_odor_world():
    odor_source = np.array([[25.0, -2.0, 2.0]], dtype=float)
    peak = np.array([[1.5]], dtype=float)
    sensor_positions = np.array(
        [
            [0.0, 0.0, 0.8],
            [1.0, 0.0, 0.8],
            [0.0, 1.0, 0.8],
            [1.0, 1.0, 0.8],
        ],
        dtype=float,
    )
    world = ObstacleOdorWorld(
        odor_source=odor_source,
        peak_odor_intensity=peak,
        obstacle_positions=np.zeros((0, 2), dtype=float),
    )

    np.testing.assert_allclose(
        _odor_intensity_from_sources(sensor_positions, odor_source, peak),
        world.get_olfaction(sensor_positions),
    )


def test_short_real_rollout_runs():
    task = OdorObstacleNavigationTask(_fast_config())
    try:
        obs, _ = task.reset(seed=3)
        for _ in range(2):
            obs, _, terminated, truncated, _ = task.step(np.array([1.0, 1.0]))
            assert np.all(np.isfinite(obs))
            if terminated or truncated:
                break
    finally:
        task.close()


def test_short_real_discrete_rollout_runs():
    task = OdorObstacleNavigationTask(_fast_config(action_space="discrete"))
    try:
        obs, _ = task.reset(seed=4)
        for _ in range(2):
            obs, _, terminated, truncated, _ = task.step(0)
            assert np.all(np.isfinite(obs))
            if terminated or truncated:
                break
    finally:
        task.close()


def test_sync_task_runner_batches_steps_and_resets():
    runner = SyncTaskRunner(StubNavigationTask, num_envs=2)
    try:
        obs, infos = runner.reset(seed=10)
        assert obs.shape == (2, 8)
        assert [info["seed"] for info in infos] == [10, 11]

        obs, rewards, terminated, truncated, infos = runner.step(
            np.zeros((2, 2), dtype=np.float32)
        )
        assert obs.shape == (2, 8)
        assert rewards.shape == (2,)
        assert not np.any(terminated)
        assert not np.any(truncated)
        assert [info["step_count"] for info in infos] == [1, 1]

        reset_obs, reset_infos = runner.reset_at([1], seeds=[20])
        assert reset_obs.shape == (1, 8)
        assert reset_infos[0]["seed"] == 20
    finally:
        runner.close()


def test_subproc_task_runner_fails_fast_when_worker_is_killed():
    runner = SubprocTaskRunner(
        DieOnResetNavigationTask,
        num_envs=1,
        mp_start_method="spawn",
    )
    try:
        started_at = time.perf_counter()
        with pytest.raises(SubprocWorkerError, match="Worker 0"):
            runner.reset(seed=0)
        assert time.perf_counter() - started_at < 2.0
    finally:
        runner.close()


def test_subproc_task_runner_batches_steps_and_resets():
    runner = SubprocTaskRunner(
        StubNavigationTask,
        num_envs=2,
        mp_start_method="forkserver",
    )
    try:
        obs, infos = runner.reset(seed=30)
        assert obs.shape == (2, 8)
        assert [info["seed"] for info in infos] == [30, 31]

        _, rewards, terminated, truncated, infos = runner.step(
            np.zeros((2, 2), dtype=np.float32)
        )
        assert rewards.shape == (2,)
        assert not np.any(terminated)
        assert not np.any(truncated)
        assert [info["step_count"] for info in infos] == [1, 1]

        reset_obs, reset_infos = runner.reset_at([0], seeds=[40])
        assert reset_obs.shape == (1, 8)
        assert reset_infos[0]["seed"] == 40
    finally:
        runner.close()


@pytest.mark.parametrize(
    ("env_runner", "mp_start_method"),
    [("sync", "spawn"), ("subproc", "forkserver")],
)
def test_navigation_vec_env_auto_resets_done_envs(
    env_runner: str,
    mp_start_method: str,
):
    from flygym_demo.odor_obstacle_navigation.ppo import NavigationVecEnv

    env = NavigationVecEnv(
        StubNavigationTask,
        num_envs=2,
        env_runner=env_runner,
        mp_start_method=mp_start_method,
    )
    try:
        obs = env.reset()
        assert obs.shape == (2, 8)
        for _ in range(3):
            obs, rewards, dones, infos = env.step(
                np.zeros((2, 2), dtype=np.float32)
            )

        assert rewards.shape == (2,)
        assert np.all(dones)
        assert obs.shape == (2, 8)
        assert "terminal_observation" in infos[0]
        assert infos[0]["TimeLimit.truncated"]
        np.testing.assert_allclose(obs, 0.0)
    finally:
        env.close()


def test_navigation_vec_env_sanitizes_nonfinite_reset_observations():
    from flygym_demo.odor_obstacle_navigation.ppo import NavigationVecEnv

    env = NavigationVecEnv(
        NonfiniteResetNavigationTask,
        num_envs=1,
        env_runner="sync",
    )
    try:
        obs = env.reset()

        assert np.all(np.isfinite(obs))
        np.testing.assert_allclose(obs, 0.0)
        assert env.reset_infos[0]["nonfinite_reset_observation_count"] == 1
    finally:
        env.close()


def test_navigation_vec_env_quarantines_nonfinite_step_outputs():
    from flygym_demo.odor_obstacle_navigation.ppo import NavigationVecEnv

    env = NavigationVecEnv(
        NonfiniteStepNavigationTask,
        num_envs=2,
        env_runner="sync",
    )
    try:
        env.reset()
        obs, rewards, dones, infos = env.step(np.zeros((2, 2), dtype=np.float32))

        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
        assert np.all(dones)
        np.testing.assert_allclose(rewards, 0.0)
        for info in infos:
            assert info["failure_reason"] == "nonfinite_env_output"
            assert info["nonfinite_env_output"]
            assert info["nonfinite_observation_count"] == 1
            assert info["nonfinite_reward"]
            assert np.all(np.isfinite(info["terminal_observation"]))
    finally:
        env.close()


def test_navigation_vec_env_supports_discrete_actions():
    from gymnasium import spaces
    from flygym_demo.odor_obstacle_navigation.ppo import NavigationVecEnv

    env = NavigationVecEnv(
        StubDiscreteNavigationTask,
        num_envs=2,
        env_runner="sync",
    )
    try:
        assert isinstance(env.action_space, spaces.Discrete)
        obs = env.reset()
        assert obs.shape == (2, 8)
        obs, rewards, dones, infos = env.step(np.array([0, 1], dtype=np.int64))

        assert obs.shape == (2, 8)
        assert rewards.shape == (2,)
        assert not np.any(dones)
        assert [info["step_count"] for info in infos] == [1, 1]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("task_factory_name", "action_shape"),
    [("continuous", (2,)), ("discrete", ())],
)
def test_sb3_ppo_smoke_learns_saves_loads_and_predicts(
    tmp_path: Path,
    task_factory_name: str,
    action_shape: tuple[int, ...],
):
    from stable_baselines3 import PPO

    from flygym_demo.odor_obstacle_navigation.ppo import NavigationVecEnv

    task_factory = (
        StubNavigationTask
        if task_factory_name == "continuous"
        else StubDiscreteNavigationTask
    )
    env = NavigationVecEnv(task_factory, num_envs=2, env_runner="sync")
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={"net_arch": [8]},
        seed=0,
        device="cpu",
    )
    try:
        model.learn(total_timesteps=8)
        checkpoint = tmp_path / "policy.zip"
        model.save(checkpoint)
        loaded = PPO.load(checkpoint, env=env, device="cpu")
        action, _ = loaded.predict(np.zeros(8, dtype=np.float32), deterministic=True)

        assert np.asarray(action).shape == action_shape
    finally:
        env.close()


def test_subproc_real_task_short_rollout_runs():
    config = _fast_config(enable_olfaction=False)
    runner = SubprocTaskRunner(
        NavigationTaskFactory(config),
        num_envs=2,
        mp_start_method="spawn",
    )
    try:
        obs, _ = runner.reset(seed=50)
        assert obs.shape == (2, 8)
        obs, rewards, terminated, truncated, _ = runner.step(
            np.ones((2, 2), dtype=np.float32)
        )
        assert obs.shape == (2, 8)
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
    finally:
        runner.close()


def test_benchmark_supports_cuda_vision_feature_mode():
    benchmark_path = (
        Path(__file__).resolve().parents[2]
        / "scripts/flygym/benchmark_odor_obstacle_navigation_training.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_odor_obstacle_navigation_training", benchmark_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    default_args = module.parse_args([])
    assert default_args.ccd_iterations == [300]
    assert module._supports_feature_mode("cuda", vision=True, olfaction=True)
    assert module._supports_feature_mode("cuda", vision=True, olfaction=False)
    args = module.parse_args(
        [
            "--ccd-iterations",
            "200",
            "250",
            "300",
            "350",
            "400",
            "--benchmark-layers",
            "env",
            "ppo",
        ]
    )
    assert args.ccd_iterations == [200, 250, 300, 350, 400]
    assert args.benchmark_layers == ["env", "ppo"]


def test_ccd_tuner_defaults_to_warp_stress_sweep():
    module = _load_ccd_tuner()
    args = module.parse_args([])

    assert args.ccd_iterations == [200, 250, 300, 350, 400]
    assert args.repeats == 5
    assert args.num_envs == 64
    assert args.feature_mode == "full"
    assert args.rollout_steps == 200
    assert args.decision_steps == 500


def test_ccd_tuner_counts_warnings():
    module = _load_ccd_tuner()

    assert (
        module.count_ccd_warnings(
            "Warning: opt.ccd_iterations, currently set to 100, "
            "needs to be increased."
        )
        == 1
    )
    assert module.count_ccd_warnings("Warning: unrelated") == 0


def test_ccd_tuner_obstacle_clearance_metric():
    module = _load_ccd_tuner()
    clearances = module.min_obstacle_clearances(
        np.array([[7.5, 0.0, 0.8], [9.0, 0.0, 0.8]], dtype=float),
        np.array([[7.5, 0.0]], dtype=float),
        obstacle_radius=1.0,
    )

    np.testing.assert_allclose(clearances, [-1.0, 0.5])


def test_ccd_tuner_recommends_lowest_stable_value():
    module = _load_ccd_tuner()
    row = module.TrialResult
    rows = [
        row(
            ccd_iterations=100,
            num_envs=64,
            repeat=0,
            ccd_warning_count=4,
            construction_wall_seconds=1.0,
            rollout_wall_seconds=10.0,
            phase_env_step_wall_seconds=9.0,
            simulator_substeps_per_second=1000.0,
            nan_or_inf_count=0,
            mean_min_obstacle_clearance=0.1,
            mean_forward_distance=5.0,
        ),
        row(
            ccd_iterations=200,
            num_envs=64,
            repeat=0,
            ccd_warning_count=0,
            construction_wall_seconds=1.0,
            rollout_wall_seconds=10.8,
            phase_env_step_wall_seconds=9.8,
            simulator_substeps_per_second=930.0,
            nan_or_inf_count=0,
            mean_min_obstacle_clearance=0.1,
            mean_forward_distance=5.0,
        ),
        row(
            ccd_iterations=300,
            num_envs=64,
            repeat=0,
            ccd_warning_count=0,
            construction_wall_seconds=1.0,
            rollout_wall_seconds=12.0,
            phase_env_step_wall_seconds=11.0,
            simulator_substeps_per_second=850.0,
            nan_or_inf_count=0,
            mean_min_obstacle_clearance=0.1,
            mean_forward_distance=5.0,
        ),
    ]

    assert (
        module.choose_recommended_ccd(
            rows,
            max_warning_count=0,
            max_slowdown_percent=10.0,
            max_mean_obstacle_clearance=0.5,
        )
        == 200
    )


def _load_ccd_tuner():
    tuner_path = (
        Path(__file__).resolve().parents[2]
        / "scripts/flygym/tune_warp_ccd_iterations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_warp_ccd_iterations", tuner_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StubTaskConfig:
    decision_steps = 2
    decision_interval = 0.1


class StubNavigationTask:
    observation_shape = (8,)
    action_space = "continuous"
    action_shape = (2,)
    action_n = None
    action_low = np.array([-0.5, -0.5], dtype=np.float32)
    action_high = np.array([1.5, 1.5], dtype=np.float32)
    config = StubTaskConfig()

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, seed: int | None = None):
        self.step_count = 0
        obs = np.zeros(8, dtype=np.float32)
        if seed is not None:
            obs[0] = seed / 1000
        return obs, {"seed": seed}

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)
        obs = np.full(8, self.step_count / 10, dtype=np.float32)
        reward = float(1.0 - 0.1 * np.linalg.norm(action))
        truncated = self.step_count >= 3
        return (
            obs,
            reward,
            False,
            truncated,
            {"success": False, "step_count": self.step_count},
        )

    def render(self):
        return [None]

    def close(self) -> None:
        pass


class DieOnResetNavigationTask(StubNavigationTask):
    def reset(self, seed: int | None = None):
        os.kill(os.getpid(), signal.SIGKILL)


class NonfiniteResetNavigationTask(StubNavigationTask):
    def reset(self, seed: int | None = None):
        obs, info = super().reset(seed=seed)
        obs[3] = np.nan
        return obs, info


class NonfiniteStepNavigationTask(StubNavigationTask):
    def step(self, action: np.ndarray):
        obs, _, _, _, info = super().step(action)
        obs[2] = np.nan
        return obs, float("inf"), False, False, info


class StubDiscreteNavigationTask:
    observation_shape = (8,)
    action_space = "discrete"
    action_shape = ()
    action_n = 3
    action_low = np.array([0], dtype=np.int64)
    action_high = np.array([action_n - 1], dtype=np.int64)
    config = StubTaskConfig()

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, seed: int | None = None):
        self.step_count = 0
        obs = np.zeros(8, dtype=np.float32)
        if seed is not None:
            obs[0] = seed / 1000
        return obs, {"seed": seed}

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = int(np.asarray(action).item())
        obs = np.full(8, self.step_count / 10, dtype=np.float32)
        obs[action] = 1.0
        reward = float(action == 0)
        truncated = self.step_count >= 3
        return (
            obs,
            reward,
            False,
            truncated,
            {"success": False, "step_count": self.step_count},
        )

    def render(self):
        return [None]

    def close(self) -> None:
        pass
