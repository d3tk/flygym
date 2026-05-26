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
    finally:
        task.close()


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
    class DieOnResetNavigationTask(StubNavigationTask):
        def reset(self, seed: int | None = None):
            os.kill(os.getpid(), signal.SIGKILL)

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
def test_ppo_smoke_with_stub_task(env_runner: str, mp_start_method: str):
    pytest.importorskip("torch")

    from flygym_demo.odor_obstacle_navigation.ppo import PPOConfig, PPOTrainer

    logged_metrics = []
    progress_events = []
    trainer = PPOTrainer(
        StubNavigationTask,
        PPOConfig(
            total_timesteps=8,
            num_envs=2,
            rollout_steps=4,
            update_epochs=1,
            minibatch_size=4,
            hidden_sizes=(8,),
            normalize_observations=False,
            seed=0,
            env_runner=env_runner,
            mp_start_method=mp_start_method,
        ),
    )
    try:
        metrics = trainer.train(
            metrics_callback=logged_metrics.append,
            progress_callback=progress_events.append,
            eval_interval=4,
            eval_episodes=1,
            eval_seed=10,
        )
        results = trainer.evaluate(num_episodes=1, seed=10)

        assert metrics["global_step"] >= 8
        optimizer_steps = [
            values["train/optimizer_step"]
            for values in logged_metrics
            if "train/optimizer_step" in values
        ]
        assert optimizer_steps[:2] == [1.0, 2.0]
        assert _expected_wandb_keys().issubset(logged_metrics[-1])
        assert {"rollout_start", "rollout_end", "update_start", "update_end"}.issubset(
            {event["stage"] for event in progress_events}
        )
        assert len(results) == 1
    finally:
        trainer.close()


def test_ppo_smoke_with_discrete_stub_task():
    pytest.importorskip("torch")

    from flygym_demo.odor_obstacle_navigation.ppo import PPOConfig, PPOTrainer

    trainer = PPOTrainer(
        StubDiscreteNavigationTask,
        PPOConfig(
            total_timesteps=8,
            num_envs=2,
            rollout_steps=4,
            update_epochs=1,
            minibatch_size=4,
            hidden_sizes=(8,),
            normalize_observations=False,
            seed=0,
            env_runner="sync",
        ),
    )
    try:
        metrics = trainer.train(eval_interval=None)
        action = trainer.act(np.zeros(8, dtype=np.float32))

        assert metrics["global_step"] >= 8
        assert np.asarray(action).shape == ()
        assert 0 <= int(action) < StubDiscreteNavigationTask.action_n
    finally:
        trainer.close()


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

    assert module._supports_feature_mode("cuda", vision=True, olfaction=True)
    assert module._supports_feature_mode("cuda", vision=True, olfaction=False)


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


def _expected_wandb_keys() -> set[str]:
    return {
        "eval/success_rate",
        "eval/mean_reward",
        "eval/mean_ep_length",
        "rollout/success_rate",
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/loss",
        "train/explained_variance",
        "train/learning_rate",
        "train/value_loss",
        "train/std",
        "train/policy_gradient_loss",
        "train/entropy_loss",
        "train/clip_range",
        "train/clip_fraction",
        "train/approx_kl",
        "train/optimizer_step",
        "time/fps",
        "speed/wall_time_seconds",
        "speed/total_timesteps_per_sec",
        "speed/simulator_substeps_per_sec",
        "speed/rollout_wall_seconds",
        "speed/simulated_seconds_per_sec",
        "speed/rollout_timesteps_per_sec",
        "speed/rollout_index",
        "speed/parent_cpu_percent",
        "global_step",
    }
