#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Sequence

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from flygym_demo.odor_obstacle_navigation import (  # noqa: E402
    NavigationTaskConfig,
    NavigationTaskFactory,
    PPOConfig,
    make_navigation_vec_env,
)
from flygym_demo.odor_obstacle_navigation.runners import SubprocWorkerError  # noqa: E402

FEATURE_MODES = {
    "full": (True, True),
    "no-vision": (False, True),
    "no-olfaction": (True, False),
    "no-sensors": (False, False),
}


@dataclass(slots=True)
class BenchmarkResult:
    benchmark_layer: str
    ccd_iterations: int
    ppo_device: str
    env_device: str
    runner: str
    mp_start_method: str
    feature_mode: str
    vision: bool
    olfaction: bool
    num_envs: int
    repeat: int
    rollout_steps: int
    rollouts: int
    decision_steps: int
    construction_wall_seconds: float
    train_wall_seconds: float
    total_wall_seconds: float
    rollout_wall_seconds: float
    update_wall_seconds: float
    rollout_decisions_per_second: float
    update_decisions_per_second: float
    train_decisions_per_second: float
    total_decisions_per_second: float
    simulator_substeps_per_second: float
    simulated_seconds_per_second: float
    worker_pid_count: int
    worker_affinity_count_min: int
    worker_affinity_count_max: int
    worker_cpu_percent_sum: float
    worker_cpu_percent_mean: float
    worker_cpu_percent_max: float
    warp_devices: str
    warp_cuda_devices: str
    selected_warp_device: str
    warp_cuda_graph_captured: str
    warp_last_action_source: str
    warp_step_count: int
    warp_step_total_wall_seconds: float
    warp_step_action_assign_wall_seconds: float
    warp_step_graph_launch_wall_seconds: float
    warp_step_state_reward_wall_seconds: float
    warp_step_observation_wall_seconds: float
    warp_step_odor_wall_seconds: float
    warp_step_vision_wall_seconds: float
    warp_step_observation_assemble_wall_seconds: float
    warp_step_info_wall_seconds: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark odor obstacle navigation training throughput."
    )
    parser.add_argument(
        "--runners",
        nargs="+",
        choices=["sync", "subproc", "warp"],
        default=["sync", "subproc"],
    )
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 20])
    parser.add_argument(
        "--mp-start-method",
        choices=["spawn", "forkserver", "fork"],
        default="spawn",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument(
        "--benchmark-layers",
        nargs="+",
        choices=["ppo", "env", "raw"],
        default=["ppo"],
        help="Benchmark full PPO, environment-only rollout, and/or raw GPUSimulation.",
    )
    parser.add_argument("--minibatch-size", type=int, default=1200)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppo-device", "--device", dest="ppo_device", type=str, default="cpu")
    parser.add_argument(
        "--ppo-devices",
        "--devices",
        dest="ppo_devices",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--env-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Environment stepping device. cuda uses the MuJoCo-Warp runner.",
    )
    parser.add_argument(
        "--env-devices",
        choices=["cpu", "cuda"],
        nargs="+",
        default=None,
    )
    parser.add_argument("--timestep", type=float, default=1e-4)
    parser.add_argument(
        "--ccd-iterations",
        type=int,
        nargs="+",
        default=[NavigationTaskConfig().ccd_iterations],
    )
    parser.add_argument("--decision-steps", type=int, default=500)
    parser.add_argument("--episode-time", type=float, default=5.0)
    parser.add_argument("--target-radius", type=float, default=2.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--progress-reward-weight", type=float, default=1.0)
    parser.add_argument("--time-penalty", type=float, default=0.01)
    parser.add_argument("--vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--olfaction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        choices=sorted(FEATURE_MODES),
        default=None,
        help=(
            "Feature matrix to run. If omitted, use the single --vision/--olfaction "
            "combination."
        ),
    )
    parser.add_argument("--randomize-layout", action="store_true")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/odor_obstacle_navigation/benchmarks/training_speed.csv"),
    )
    parser.add_argument(
        "--continue-after-worker-failure",
        action="store_true",
        help=(
            "Keep running larger num-envs sweeps after a subprocess worker dies. "
            "By default the benchmark exits immediately."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive.")
    if args.rollouts <= 0:
        raise ValueError("rollouts must be positive.")
    if args.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")
    if any(num_envs <= 0 for num_envs in args.num_envs):
        raise ValueError("all num-envs values must be positive.")
    if any(ccd_iterations <= 0 for ccd_iterations in args.ccd_iterations):
        raise ValueError("all ccd-iterations values must be positive.")

    ppo_devices = _requested_ppo_devices(args)
    env_devices = _requested_env_devices(args)
    feature_modes = _requested_feature_modes(args)
    _validate_ppo_devices(ppo_devices)
    _validate_env_devices(env_devices)
    if "raw" in args.benchmark_layers:
        _validate_env_devices(["cuda"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[BenchmarkResult] = []
    print(_system_summary(args, ppo_devices, env_devices, feature_modes), flush=True)
    if "ppo" in args.benchmark_layers:
        for ppo_device in ppo_devices:
            for env_device in env_devices:
                runners = _requested_runners(args, env_device)
                for runner in runners:
                    for feature_mode in feature_modes:
                        vision, olfaction = FEATURE_MODES[feature_mode]
                        if not _supports_feature_mode(env_device, vision, olfaction):
                            print(
                                "Skipping "
                                f"ppo_device={ppo_device} env_device={env_device} "
                                f"runner={runner} features={feature_mode}: "
                                "unsupported feature mode.",
                                flush=True,
                            )
                            continue
                        for ccd_iterations in args.ccd_iterations:
                            for num_envs in args.num_envs:
                                for repeat in range(args.repeats):
                                    try:
                                        result = _run_trial(
                                            args,
                                            ccd_iterations,
                                            ppo_device,
                                            env_device,
                                            runner,
                                            feature_mode,
                                            vision,
                                            olfaction,
                                            num_envs,
                                            repeat,
                                        )
                                    except SubprocWorkerError as exc:
                                        _handle_worker_failure(
                                            args,
                                            exc,
                                            benchmark_layer="ppo",
                                            ppo_device=ppo_device,
                                            env_device=env_device,
                                            runner=runner,
                                            feature_mode=feature_mode,
                                            num_envs=num_envs,
                                            repeat=repeat,
                                        )
                                        continue
                                    rows.append(result)
                                    _print_trial(result)
                                    _write_rows(args.output_csv, rows)
    if "env" in args.benchmark_layers:
        for env_device in env_devices:
            runners = _requested_runners(args, env_device)
            for runner in runners:
                for feature_mode in feature_modes:
                    vision, olfaction = FEATURE_MODES[feature_mode]
                    if not _supports_feature_mode(env_device, vision, olfaction):
                        continue
                    for ccd_iterations in args.ccd_iterations:
                        for num_envs in args.num_envs:
                            for repeat in range(args.repeats):
                                try:
                                    result = _run_env_trial(
                                        args,
                                        ccd_iterations,
                                        env_device,
                                        runner,
                                        feature_mode,
                                        vision,
                                        olfaction,
                                        num_envs,
                                        repeat,
                                    )
                                except SubprocWorkerError as exc:
                                    _handle_worker_failure(
                                        args,
                                        exc,
                                        benchmark_layer="env",
                                        ppo_device="",
                                        env_device=env_device,
                                        runner=runner,
                                        feature_mode=feature_mode,
                                        num_envs=num_envs,
                                        repeat=repeat,
                                    )
                                    continue
                                rows.append(result)
                                _print_trial(result)
                                _write_rows(args.output_csv, rows)
    if "raw" in args.benchmark_layers:
        for num_envs in args.num_envs:
            for repeat in range(args.repeats):
                result = _run_raw_trial(args, num_envs, repeat)
                rows.append(result)
                _print_trial(result)
                _write_rows(args.output_csv, rows)
    _print_summary(rows, args.output_csv)


def _handle_worker_failure(
    args: argparse.Namespace,
    exc: SubprocWorkerError,
    *,
    benchmark_layer: str,
    ppo_device: str,
    env_device: str,
    runner: str,
    feature_mode: str,
    num_envs: int,
    repeat: int,
) -> None:
    print(
        "Subprocess worker failure: "
        f"layer={benchmark_layer} ppo_device={ppo_device} env_device={env_device} "
        f"runner={runner} features={feature_mode} "
        f"num_envs={num_envs} repeat={repeat}: {exc}",
        flush=True,
    )
    if not args.continue_after_worker_failure:
        raise SystemExit(1) from exc


def _run_trial(
    args: argparse.Namespace,
    ccd_iterations: int,
    ppo_device: str,
    env_device: str,
    runner: str,
    feature_mode: str,
    vision: bool,
    olfaction: bool,
    num_envs: int,
    repeat: int,
) -> BenchmarkResult:
    seed = args.seed + 10_000 * repeat + num_envs
    task_config = NavigationTaskConfig(
        timestep=args.timestep,
        ccd_iterations=ccd_iterations,
        decision_steps=args.decision_steps,
        episode_time=args.episode_time,
        target_radius=args.target_radius,
        success_bonus=args.success_bonus,
        progress_reward_weight=args.progress_reward_weight,
        time_penalty=args.time_penalty,
        randomize_layout=args.randomize_layout,
        enable_vision=vision,
        enable_olfaction=olfaction,
        render=False,
    )
    total_timesteps = args.rollouts * args.rollout_steps * num_envs
    ppo_config = PPOConfig(
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        hidden_sizes=tuple(args.hidden_sizes),
        seed=seed,
        device=ppo_device,
        env_device=env_device,
        env_runner=runner,
        mp_start_method=args.mp_start_method,
    )
    _synchronize_devices(ppo_device, env_device)
    trial_started_at = time.perf_counter()
    construction_started_at = time.perf_counter()
    env = make_navigation_vec_env(NavigationTaskFactory(task_config), ppo_config)
    model = PPO("MlpPolicy", env, **ppo_config.sb3_kwargs())
    _synchronize_devices(ppo_device, env_device)
    construction_wall_seconds = time.perf_counter() - construction_started_at
    try:
        _synchronize_devices(ppo_device, env_device)
        train_started_at = time.perf_counter()
        model.learn(total_timesteps=total_timesteps)
        _synchronize_devices(ppo_device, env_device)
        train_wall_seconds = time.perf_counter() - train_started_at
        worker_summary = _worker_diagnostics_summary(env.diagnostics())
    finally:
        env.close()
    _synchronize_devices(ppo_device, env_device)
    total_wall_seconds = time.perf_counter() - trial_started_at
    rollout_wall_seconds = train_wall_seconds
    rollout_decisions = total_timesteps
    return BenchmarkResult(
        benchmark_layer="ppo",
        ccd_iterations=ccd_iterations,
        ppo_device=ppo_device,
        env_device=env_device,
        runner=runner,
        mp_start_method=args.mp_start_method if runner == "subproc" and env_device == "cpu" else "",
        feature_mode=feature_mode,
        vision=vision,
        olfaction=olfaction,
        num_envs=num_envs,
        repeat=repeat,
        rollout_steps=args.rollout_steps,
        rollouts=args.rollouts,
        decision_steps=args.decision_steps,
        construction_wall_seconds=construction_wall_seconds,
        train_wall_seconds=train_wall_seconds,
        total_wall_seconds=total_wall_seconds,
        rollout_wall_seconds=rollout_wall_seconds,
        update_wall_seconds=0.0,
        rollout_decisions_per_second=_rate(rollout_decisions, rollout_wall_seconds),
        update_decisions_per_second=0.0,
        train_decisions_per_second=_rate(total_timesteps, train_wall_seconds),
        total_decisions_per_second=_rate(total_timesteps, total_wall_seconds),
        simulator_substeps_per_second=_rate(
            rollout_decisions * args.decision_steps,
            rollout_wall_seconds,
        ),
        simulated_seconds_per_second=_rate(
            rollout_decisions * task_config.decision_interval,
            rollout_wall_seconds,
        ),
        worker_pid_count=worker_summary["worker_pid_count"],
        worker_affinity_count_min=worker_summary["worker_affinity_count_min"],
        worker_affinity_count_max=worker_summary["worker_affinity_count_max"],
        worker_cpu_percent_sum=0.0,
        worker_cpu_percent_mean=0.0,
        worker_cpu_percent_max=0.0,
        warp_devices=worker_summary["warp_devices"],
        warp_cuda_devices=worker_summary["warp_cuda_devices"],
        selected_warp_device=worker_summary["selected_warp_device"],
        warp_cuda_graph_captured=worker_summary["warp_cuda_graph_captured"],
        warp_last_action_source=worker_summary["warp_last_action_source"],
        warp_step_count=int(worker_summary["warp_step_count"]),
        warp_step_total_wall_seconds=worker_summary["warp_step_total_wall_seconds"],
        warp_step_action_assign_wall_seconds=worker_summary[
            "warp_step_action_assign_wall_seconds"
        ],
        warp_step_graph_launch_wall_seconds=worker_summary[
            "warp_step_graph_launch_wall_seconds"
        ],
        warp_step_state_reward_wall_seconds=worker_summary[
            "warp_step_state_reward_wall_seconds"
        ],
        warp_step_observation_wall_seconds=worker_summary[
            "warp_step_observation_wall_seconds"
        ],
        warp_step_odor_wall_seconds=worker_summary["warp_step_odor_wall_seconds"],
        warp_step_vision_wall_seconds=worker_summary["warp_step_vision_wall_seconds"],
        warp_step_observation_assemble_wall_seconds=worker_summary[
            "warp_step_observation_assemble_wall_seconds"
        ],
        warp_step_info_wall_seconds=worker_summary["warp_step_info_wall_seconds"],
    )


def _run_env_trial(
    args: argparse.Namespace,
    ccd_iterations: int,
    env_device: str,
    runner: str,
    feature_mode: str,
    vision: bool,
    olfaction: bool,
    num_envs: int,
    repeat: int,
) -> BenchmarkResult:
    from flygym_demo.odor_obstacle_navigation.runners import make_task_runner

    seed = args.seed + 10_000 * repeat + num_envs
    task_config = NavigationTaskConfig(
        timestep=args.timestep,
        ccd_iterations=ccd_iterations,
        decision_steps=args.decision_steps,
        episode_time=args.episode_time,
        target_radius=args.target_radius,
        success_bonus=args.success_bonus,
        progress_reward_weight=args.progress_reward_weight,
        time_penalty=args.time_penalty,
        randomize_layout=args.randomize_layout,
        enable_vision=vision,
        enable_olfaction=olfaction,
        render=False,
    )
    total_decisions = args.rollouts * args.rollout_steps * num_envs
    trial_started_at = time.perf_counter()
    construction_started_at = time.perf_counter()
    task_runner = make_task_runner(
        NavigationTaskFactory(task_config),
        num_envs=num_envs,
        env_device=env_device,
        env_runner=runner,
        mp_start_method=args.mp_start_method,
    )
    construction_wall_seconds = time.perf_counter() - construction_started_at
    try:
        obs, _ = task_runner.reset(seed=seed)
        del obs
        action = np.zeros((num_envs, 2), dtype=np.float32)
        rollout_started_at = time.perf_counter()
        for _ in range(args.rollouts * args.rollout_steps):
            _, _, terminated, truncated, _ = task_runner.step(action)
            done_indices = np.flatnonzero(np.logical_or(terminated, truncated))
            if done_indices.size:
                task_runner.reset_at(done_indices)
        _synchronize_devices("cpu", env_device)
        rollout_wall_seconds = time.perf_counter() - rollout_started_at
        diagnostics = {"workers": task_runner.diagnostics()}
        worker_summary = _worker_diagnostics_summary(diagnostics)
    finally:
        task_runner.close()
    total_wall_seconds = time.perf_counter() - trial_started_at
    return BenchmarkResult(
        benchmark_layer="env",
        ccd_iterations=ccd_iterations,
        ppo_device="",
        env_device=env_device,
        runner=runner,
        mp_start_method=args.mp_start_method
        if runner == "subproc" and env_device == "cpu"
        else "",
        feature_mode=feature_mode,
        vision=vision,
        olfaction=olfaction,
        num_envs=num_envs,
        repeat=repeat,
        rollout_steps=args.rollout_steps,
        rollouts=args.rollouts,
        decision_steps=args.decision_steps,
        construction_wall_seconds=construction_wall_seconds,
        train_wall_seconds=rollout_wall_seconds,
        total_wall_seconds=total_wall_seconds,
        rollout_wall_seconds=rollout_wall_seconds,
        update_wall_seconds=0.0,
        rollout_decisions_per_second=_rate(total_decisions, rollout_wall_seconds),
        update_decisions_per_second=0.0,
        train_decisions_per_second=_rate(total_decisions, rollout_wall_seconds),
        total_decisions_per_second=_rate(total_decisions, total_wall_seconds),
        simulator_substeps_per_second=_rate(
            total_decisions * args.decision_steps,
            rollout_wall_seconds,
        ),
        simulated_seconds_per_second=_rate(
            total_decisions * task_config.decision_interval,
            rollout_wall_seconds,
        ),
        worker_pid_count=worker_summary["worker_pid_count"],
        worker_affinity_count_min=worker_summary["worker_affinity_count_min"],
        worker_affinity_count_max=worker_summary["worker_affinity_count_max"],
        worker_cpu_percent_sum=0.0,
        worker_cpu_percent_mean=0.0,
        worker_cpu_percent_max=0.0,
        warp_devices=worker_summary["warp_devices"],
        warp_cuda_devices=worker_summary["warp_cuda_devices"],
        selected_warp_device=worker_summary["selected_warp_device"],
        warp_cuda_graph_captured=worker_summary["warp_cuda_graph_captured"],
        warp_last_action_source=worker_summary["warp_last_action_source"],
        warp_step_count=int(worker_summary["warp_step_count"]),
        warp_step_total_wall_seconds=worker_summary["warp_step_total_wall_seconds"],
        warp_step_action_assign_wall_seconds=worker_summary[
            "warp_step_action_assign_wall_seconds"
        ],
        warp_step_graph_launch_wall_seconds=worker_summary[
            "warp_step_graph_launch_wall_seconds"
        ],
        warp_step_state_reward_wall_seconds=worker_summary[
            "warp_step_state_reward_wall_seconds"
        ],
        warp_step_observation_wall_seconds=worker_summary[
            "warp_step_observation_wall_seconds"
        ],
        warp_step_odor_wall_seconds=worker_summary["warp_step_odor_wall_seconds"],
        warp_step_vision_wall_seconds=worker_summary["warp_step_vision_wall_seconds"],
        warp_step_observation_assemble_wall_seconds=worker_summary[
            "warp_step_observation_assemble_wall_seconds"
        ],
        warp_step_info_wall_seconds=worker_summary["warp_step_info_wall_seconds"],
    )


def _run_raw_trial(
    args: argparse.Namespace,
    num_envs: int,
    repeat: int,
) -> BenchmarkResult:
    from flygym.compose import ActuatorType
    from flygym_demo.benchmark.time_gpu_simulation import (
        ReplayTargetData,
        make_model,
        run_simulation,
    )

    ref_fly, *_ = make_model()
    jointdofs_order = ref_fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    replay_data = ReplayTargetData(args.timestep, jointdofs_order)
    sim_steps = args.rollouts * args.rollout_steps * args.decision_steps
    target_angles = replay_data.make_target_angles_all_worlds(num_envs, sim_steps)
    trial_started_at = time.perf_counter()
    wall_seconds = run_simulation(
        target_angles,
        enable_rendering=False,
        timestep=args.timestep,
        simplify_geom=False,
    )
    total_wall_seconds = time.perf_counter() - trial_started_at
    substeps = sim_steps * num_envs
    decision_equivalents = args.rollouts * args.rollout_steps * num_envs
    return BenchmarkResult(
        benchmark_layer="raw",
        ccd_iterations=0,
        ppo_device="",
        env_device="cuda",
        runner="raw",
        mp_start_method="",
        feature_mode="raw",
        vision=False,
        olfaction=False,
        num_envs=num_envs,
        repeat=repeat,
        rollout_steps=args.rollout_steps,
        rollouts=args.rollouts,
        decision_steps=args.decision_steps,
        construction_wall_seconds=total_wall_seconds - wall_seconds,
        train_wall_seconds=wall_seconds,
        total_wall_seconds=total_wall_seconds,
        rollout_wall_seconds=wall_seconds,
        update_wall_seconds=0.0,
        rollout_decisions_per_second=_rate(decision_equivalents, wall_seconds),
        update_decisions_per_second=0.0,
        train_decisions_per_second=_rate(decision_equivalents, wall_seconds),
        total_decisions_per_second=_rate(decision_equivalents, total_wall_seconds),
        simulator_substeps_per_second=_rate(substeps, wall_seconds),
        simulated_seconds_per_second=_rate(substeps * args.timestep, wall_seconds),
        worker_pid_count=0,
        worker_affinity_count_min=0,
        worker_affinity_count_max=0,
        worker_cpu_percent_sum=0.0,
        worker_cpu_percent_mean=0.0,
        worker_cpu_percent_max=0.0,
        warp_devices=",".join(_warp_device_names()),
        warp_cuda_devices=",".join(_warp_cuda_device_names()),
        selected_warp_device="",
        warp_cuda_graph_captured="raw",
        warp_last_action_source="",
        warp_step_count=0,
        warp_step_total_wall_seconds=0.0,
        warp_step_action_assign_wall_seconds=0.0,
        warp_step_graph_launch_wall_seconds=0.0,
        warp_step_state_reward_wall_seconds=0.0,
        warp_step_observation_wall_seconds=0.0,
        warp_step_odor_wall_seconds=0.0,
        warp_step_vision_wall_seconds=0.0,
        warp_step_observation_assemble_wall_seconds=0.0,
        warp_step_info_wall_seconds=0.0,
    )


def _system_summary(
    args: argparse.Namespace,
    ppo_devices: list[str],
    env_devices: list[str],
    feature_modes: list[str],
) -> str:
    affinity_count = None
    if hasattr(os, "sched_getaffinity"):
        affinity_count = len(os.sched_getaffinity(0))
    summary = {
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": affinity_count,
        "threads": {
            "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
            "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
            "NUMEXPR_NUM_THREADS": os.environ["NUMEXPR_NUM_THREADS"],
            "VECLIB_MAXIMUM_THREADS": os.environ["VECLIB_MAXIMUM_THREADS"],
        },
        "mujoco_gl": os.environ["MUJOCO_GL"],
        "pyopengl_platform": os.environ["PYOPENGL_PLATFORM"],
        "decision_steps": args.decision_steps,
        "ccd_iterations": args.ccd_iterations,
        "rollout_steps": args.rollout_steps,
        "rollouts": args.rollouts,
        "benchmark_layers": args.benchmark_layers,
        "ppo_devices": ppo_devices,
        "env_devices": env_devices,
        "feature_modes": feature_modes,
        "warp": {
            "devices": _warp_device_names(),
            "cuda_devices": _warp_cuda_device_names(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_devices": _cuda_device_names(),
        },
        "vision": args.vision,
        "olfaction": args.olfaction,
    }
    return json.dumps(summary, indent=2)


def _print_trial(result: BenchmarkResult) -> None:
    message = (
        "layer={layer:<3} ccd={ccd:<4} ppo={ppo:<7} env={env:<4} "
        "runner={runner:<7} "
        "envs={envs:<4} repeat={repeat:<2} "
        "features={features:<11} "
        "rollout_dec/s={rollout_decisions_per_second:>8.2f} "
        "update_dec/s={update_decisions_per_second:>8.2f} "
        "train_dec/s={train_decisions_per_second:>8.2f} "
        "total_dec/s={total_decisions_per_second:>8.2f} "
        "substeps/s={simulator_substeps_per_second:>10.0f} "
        "sim_s/s={simulated_seconds_per_second:>7.3f} "
        "construct={construction_wall_seconds:>6.2f}s "
        "update={update_wall_seconds:>6.2f}s "
        "train={train_wall_seconds:>6.2f}s".format(
            layer=result.benchmark_layer,
            ccd=result.ccd_iterations,
            ppo=result.ppo_device,
            env=result.env_device,
            runner=result.runner,
            envs=result.num_envs,
            repeat=result.repeat,
            features=result.feature_mode,
            rollout_decisions_per_second=result.rollout_decisions_per_second,
            update_decisions_per_second=result.update_decisions_per_second,
            train_decisions_per_second=result.train_decisions_per_second,
            total_decisions_per_second=result.total_decisions_per_second,
            simulator_substeps_per_second=result.simulator_substeps_per_second,
            simulated_seconds_per_second=result.simulated_seconds_per_second,
            construction_wall_seconds=result.construction_wall_seconds,
            update_wall_seconds=result.update_wall_seconds,
            train_wall_seconds=result.train_wall_seconds,
        )
    )
    if result.warp_step_count:
        message += (
            " warp_step_ms={step_ms:>7.2f} "
            "assign_ms={assign_ms:>6.2f} "
            "launch_ms={launch_ms:>6.2f} "
            "state_ms={state_ms:>6.2f} "
            "odor_ms={odor_ms:>6.2f} "
            "vision_ms={vision_ms:>6.2f} "
            "obs_asm_ms={obs_asm_ms:>6.2f} "
            "info_ms={info_ms:>6.2f} "
            "action={action_source}"
        ).format(
            step_ms=_milliseconds_per_step(
                result.warp_step_total_wall_seconds,
                result.warp_step_count,
            ),
            assign_ms=_milliseconds_per_step(
                result.warp_step_action_assign_wall_seconds,
                result.warp_step_count,
            ),
            launch_ms=_milliseconds_per_step(
                result.warp_step_graph_launch_wall_seconds,
                result.warp_step_count,
            ),
            state_ms=_milliseconds_per_step(
                result.warp_step_state_reward_wall_seconds,
                result.warp_step_count,
            ),
            odor_ms=_milliseconds_per_step(
                result.warp_step_odor_wall_seconds,
                result.warp_step_count,
            ),
            vision_ms=_milliseconds_per_step(
                result.warp_step_vision_wall_seconds,
                result.warp_step_count,
            ),
            obs_asm_ms=_milliseconds_per_step(
                result.warp_step_observation_assemble_wall_seconds,
                result.warp_step_count,
            ),
            info_ms=_milliseconds_per_step(
                result.warp_step_info_wall_seconds,
                result.warp_step_count,
            ),
            action_source=result.warp_last_action_source or "-",
        )
    print(message, flush=True)


def _write_rows(path: Path, rows: list[BenchmarkResult]) -> None:
    fieldnames = list(BenchmarkResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def _print_summary(rows: list[BenchmarkResult], path: Path) -> None:
    if not rows:
        print(f"\nNo benchmark rows were produced; no CSV was written to {path}")
        return
    groups: dict[
        tuple[str, int, str, str, str, str, str, int], list[BenchmarkResult]
    ] = {}
    for row in rows:
        groups.setdefault(
            (
                row.benchmark_layer,
                row.ccd_iterations,
                row.ppo_device,
                row.env_device,
                row.runner,
                row.mp_start_method,
                row.feature_mode,
                row.num_envs,
            ),
            [],
        ).append(row)
    summary = []
    for (
        benchmark_layer,
        ccd_iterations,
        ppo_device,
        env_device,
        runner,
        mp_start_method,
        feature_mode,
        num_envs,
    ), group_rows in groups.items():
        summary.append(
            {
                "ppo_device": ppo_device,
                "benchmark_layer": benchmark_layer,
                "ccd_iterations": ccd_iterations,
                "env_device": env_device,
                "runner": runner,
                "mp_start_method": mp_start_method,
                "feature_mode": feature_mode,
                "num_envs": num_envs,
                "mean_rollout_decisions_per_second": statistics.mean(
                    row.rollout_decisions_per_second for row in group_rows
                ),
                "mean_simulator_substeps_per_second": statistics.mean(
                    row.simulator_substeps_per_second for row in group_rows
                ),
                "mean_update_decisions_per_second": statistics.mean(
                    row.update_decisions_per_second for row in group_rows
                ),
                "mean_train_decisions_per_second": statistics.mean(
                    row.train_decisions_per_second for row in group_rows
                ),
                "mean_total_decisions_per_second": statistics.mean(
                    row.total_decisions_per_second for row in group_rows
                ),
            }
        )
    summary.sort(
        key=lambda row: row["mean_simulator_substeps_per_second"],
        reverse=True,
    )
    print(f"\nWrote benchmark rows to {path}")
    print("Top configurations by simulator substeps/s:")
    for row in summary[:10]:
        print(
            "layer={layer:<3} ccd={ccd:<4} ppo={ppo:<7} env={env:<4} "
            "runner={runner:<7} "
            "start={start:<10} envs={envs:<4} "
            "features={features:<11} "
            "rollout_dec/s={rollout_decisions_per_second:>8.2f} "
            "update_dec/s={update_decisions_per_second:>8.2f} "
            "train_dec/s={train_decisions_per_second:>8.2f} "
            "total_dec/s={total_decisions_per_second:>8.2f} "
            "substeps/s={substeps_per_second:>10.0f} "
            .format(
                layer=row["benchmark_layer"],
                ccd=row["ccd_iterations"],
                ppo=row["ppo_device"],
                env=row["env_device"],
                runner=row["runner"],
                start=row["mp_start_method"] or "-",
                envs=row["num_envs"],
                features=row["feature_mode"],
                rollout_decisions_per_second=row[
                    "mean_rollout_decisions_per_second"
                ],
                update_decisions_per_second=row[
                    "mean_update_decisions_per_second"
                ],
                train_decisions_per_second=row["mean_train_decisions_per_second"],
                total_decisions_per_second=row["mean_total_decisions_per_second"],
                substeps_per_second=row["mean_simulator_substeps_per_second"],
            )
        )


def _requested_ppo_devices(args: argparse.Namespace) -> list[str]:
    return args.ppo_devices if args.ppo_devices is not None else [args.ppo_device]


def _requested_env_devices(args: argparse.Namespace) -> list[str]:
    return args.env_devices if args.env_devices is not None else [args.env_device]


def _requested_runners(args: argparse.Namespace, env_device: str) -> list[str]:
    if env_device == "cuda":
        return ["warp"]
    return [runner for runner in args.runners if runner != "warp"]


def _requested_feature_modes(args: argparse.Namespace) -> list[str]:
    if args.feature_modes is not None:
        return list(args.feature_modes)
    for name, (vision, olfaction) in FEATURE_MODES.items():
        if vision == args.vision and olfaction == args.olfaction:
            return [name]
    raise RuntimeError("Unsupported feature combination.")


def _supports_feature_mode(env_device: str, vision: bool, olfaction: bool) -> bool:
    return True


def _validate_ppo_devices(ppo_devices: list[str]) -> None:
    for ppo_device in ppo_devices:
        torch_device = torch.device(ppo_device)
        if torch_device.type != "cuda":
            continue
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA PPO device requested but unavailable: {ppo_device}")
        if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA PPO device index out of range: {ppo_device}")


def _validate_env_devices(env_devices: list[str]) -> None:
    for env_device in env_devices:
        if env_device == "cpu":
            continue
        if env_device != "cuda":
            raise RuntimeError(f"Unsupported env_device: {env_device}")
        try:
            from flygym.warp.utils import check_gpu
        except ImportError as exc:
            raise RuntimeError(
                "CUDA env_device requires FlyGym's warp extra. "
                "Install/sync with --extra warp."
            ) from exc
        check_gpu()


def _synchronize_devices(ppo_device: str, env_device: str) -> None:
    torch_device = torch.device(ppo_device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    if env_device == "cuda":
        import warp as wp

        wp.synchronize()


def _cuda_device_names() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]


def _worker_diagnostics_summary(diagnostics: dict) -> dict[str, int | float | str]:
    workers = diagnostics.get("workers", [])
    pids = {
        int(worker["pid"])
        for worker in workers
        if isinstance(worker, dict) and worker.get("pid") is not None
    }
    affinity_counts = [
        int(worker["cpu_affinity_count"])
        for worker in workers
        if isinstance(worker, dict) and worker.get("cpu_affinity_count") is not None
    ]
    warp_devices = _join_unique_worker_values(workers, "warp_devices")
    warp_cuda_devices = _join_unique_worker_values(workers, "warp_cuda_devices")
    selected_warp_device = _join_unique_worker_values(workers, "selected_warp_device")
    warp_timing_fields = [
        "warp_step_total_wall_seconds",
        "warp_step_action_assign_wall_seconds",
        "warp_step_graph_launch_wall_seconds",
        "warp_step_state_reward_wall_seconds",
        "warp_step_observation_wall_seconds",
        "warp_step_odor_wall_seconds",
        "warp_step_vision_wall_seconds",
        "warp_step_observation_assemble_wall_seconds",
        "warp_step_info_wall_seconds",
    ]
    return {
        "worker_pid_count": len(pids),
        "worker_affinity_count_min": min(affinity_counts) if affinity_counts else 0,
        "worker_affinity_count_max": max(affinity_counts) if affinity_counts else 0,
        "warp_devices": warp_devices,
        "warp_cuda_devices": warp_cuda_devices,
        "selected_warp_device": selected_warp_device,
        "warp_cuda_graph_captured": _join_unique_worker_values(
            workers, "warp_cuda_graph_captured"
        ),
        "warp_last_action_source": _join_unique_worker_values(
            workers, "warp_last_action_source"
        ),
        "warp_step_count": int(_sum_worker_numeric_values(workers, "warp_step_count")),
        **{
            field: _sum_worker_numeric_values(workers, field)
            for field in warp_timing_fields
        },
    }


def _join_unique_worker_values(workers: list[dict], key: str) -> str:
    values = []
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        value = worker.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))
    return ",".join(dict.fromkeys(values))


def _sum_worker_numeric_values(workers: list[dict], key: str) -> float:
    total = 0.0
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        value = worker.get(key)
        if value is not None:
            total += float(value)
    return total


def _milliseconds_per_step(wall_seconds: float, step_count: int) -> float:
    return _rate(wall_seconds * 1000.0, step_count)


def _warp_device_names() -> list[str]:
    try:
        import warp as wp
    except ImportError:
        return []
    return [str(device) for device in wp.get_devices()]


def _warp_cuda_device_names() -> list[str]:
    try:
        import warp as wp
    except ImportError:
        return []
    return [str(device) for device in wp.get_devices() if device.is_cuda]


def _rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
