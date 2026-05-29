from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import traceback
from typing import Any, Callable

import numpy as np

_DEFAULT_RESPONSE_TIMEOUT_SECONDS = float(
    os.environ.get("FLYGYM_SUBPROC_RESPONSE_TIMEOUT", "3600")
)
_POLL_INTERVAL_SECONDS = 0.1


class SubprocWorkerError(RuntimeError):
    """Raised when a subprocess environment worker exits or stops responding."""


@dataclass(slots=True)
class TaskRunnerSpec:
    action_space: str
    action_low: np.ndarray
    action_high: np.ndarray
    action_shape: tuple[int, ...]
    action_n: int | None
    action_max_norm: float | None
    observation_shape: tuple[int, ...]
    decision_steps: float
    decision_interval: float


class SyncTaskRunner:
    def __init__(self, task_factory: Callable[[], Any], num_envs: int) -> None:
        self.tasks = [task_factory() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.spec = _task_spec(self.tasks[0])

    @property
    def action_low(self) -> np.ndarray:
        return self.spec.action_low

    @property
    def action_high(self) -> np.ndarray:
        return self.spec.action_high

    @property
    def action_space(self) -> str:
        return self.spec.action_space

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self.spec.action_shape

    @property
    def action_n(self) -> int | None:
        return self.spec.action_n

    @property
    def action_max_norm(self) -> float | None:
        return self.spec.action_max_norm

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self.spec.observation_shape

    @property
    def decision_steps(self) -> float:
        return self.spec.decision_steps

    @property
    def decision_interval(self) -> float:
        return self.spec.decision_interval

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
        seeds = _indexed_seeds(seed, self.num_envs)
        return self.reset_at(range(self.num_envs), seeds)

    def reset_at(
        self,
        indices,
        seeds: list[int | None] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        indices = list(indices)
        if not indices:
            return _empty_obs(self.observation_shape), []
        seeds = [None] * len(indices) if seeds is None else seeds
        results = [
            self.tasks[idx].reset(seed=seed)
            for idx, seed in zip(indices, seeds, strict=True)
        ]
        obs, infos = zip(*results, strict=True)
        return np.stack(obs).astype(np.float32), list(infos)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        results = [
            task.step(action)
            for task, action in zip(self.tasks, np.asarray(actions), strict=True)
        ]
        obs, rewards, terminated, truncated, infos = zip(*results, strict=True)
        return (
            np.stack(obs).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            list(infos),
        )

    def close(self) -> None:
        for task in self.tasks:
            task.close()

    def diagnostics(self) -> list[dict[str, Any]]:
        return [
            _task_diagnostics(task, rank=rank) for rank, task in enumerate(self.tasks)
        ]


class SubprocTaskRunner:
    def __init__(
        self,
        task_factory: Callable[[], Any],
        num_envs: int,
        *,
        mp_start_method: str = "spawn",
        response_timeout: float | None = None,
    ) -> None:
        self.num_envs = num_envs
        self._response_timeout = (
            _DEFAULT_RESPONSE_TIMEOUT_SECONDS
            if response_timeout is None
            else response_timeout
        )
        if self._response_timeout <= 0:
            raise ValueError("response_timeout must be positive.")
        self._ctx = mp.get_context(mp_start_method)
        self._closed = False
        self._conns = []
        self._processes = []
        for rank in range(num_envs):
            parent_conn, child_conn = self._ctx.Pipe()
            process = self._ctx.Process(
                target=_task_worker,
                args=(child_conn, task_factory, rank),
                daemon=False,
            )
            process.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._processes.append(process)
        try:
            self.spec = self._request_specs()[0]
        except BaseException:
            self.close()
            raise

    @property
    def action_low(self) -> np.ndarray:
        return self.spec.action_low

    @property
    def action_high(self) -> np.ndarray:
        return self.spec.action_high

    @property
    def action_space(self) -> str:
        return self.spec.action_space

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self.spec.action_shape

    @property
    def action_n(self) -> int | None:
        return self.spec.action_n

    @property
    def action_max_norm(self) -> float | None:
        return self.spec.action_max_norm

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self.spec.observation_shape

    @property
    def decision_steps(self) -> float:
        return self.spec.decision_steps

    @property
    def decision_interval(self) -> float:
        return self.spec.decision_interval

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
        seeds = _indexed_seeds(seed, self.num_envs)
        return self.reset_at(range(self.num_envs), seeds)

    def reset_at(
        self,
        indices,
        seeds: list[int | None] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        indices = list(indices)
        if not indices:
            return _empty_obs(self.observation_shape), []
        seeds = [None] * len(indices) if seeds is None else seeds
        for idx, seed in zip(indices, seeds, strict=True):
            self._send_command(idx, ("reset", seed))
        results = [self._recv(idx) for idx in indices]
        obs, infos = zip(*results, strict=True)
        return np.stack(obs).astype(np.float32), list(infos)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        action_dtype = np.int64 if self.action_space == "discrete" else np.float32
        for idx, action in enumerate(np.asarray(actions)):
            self._send_command(
                idx, ("step", np.asarray(action, dtype=action_dtype))
            )
        results = [self._recv(idx) for idx in range(self.num_envs)]
        obs, rewards, terminated, truncated, infos = zip(*results, strict=True)
        return (
            np.stack(obs).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            list(infos),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for idx, conn in enumerate(self._conns):
            process = self._processes[idx]
            if process.is_alive():
                try:
                    conn.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for idx, conn in enumerate(self._conns):
            process = self._processes[idx]
            if process.is_alive() and conn.poll(0):
                try:
                    self._recv(idx)
                except (SubprocWorkerError, RuntimeError, EOFError):
                    pass
            try:
                conn.close()
            except OSError:
                pass
        self._reap_workers(join_timeout=0.2)

    def _request_specs(self) -> list[TaskRunnerSpec]:
        for idx in range(self.num_envs):
            self._send_command(idx, ("spec", None))
        return [self._recv(idx) for idx in range(self.num_envs)]

    def diagnostics(self) -> list[dict[str, Any]]:
        for idx in range(self.num_envs):
            self._send_command(idx, ("diagnostics", None))
        return [self._recv(idx) for idx in range(self.num_envs)]

    def _send_command(self, idx: int, command: tuple[Any, Any]) -> None:
        self._ensure_worker_alive(idx)
        try:
            self._conns[idx].send(command)
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._raise_worker_failure(idx, exc)

    def _recv(self, idx: int):
        conn = self._conns[idx]
        deadline = time.monotonic() + self._response_timeout
        while not conn.poll(0):
            self._ensure_worker_alive(idx)
            if time.monotonic() >= deadline:
                self._raise_worker_failure(
                    idx,
                    TimeoutError(
                        f"no response within {self._response_timeout:g}s"
                    ),
                )
            time.sleep(_POLL_INTERVAL_SECONDS)
        try:
            status, payload = conn.recv()
        except (EOFError, ConnectionResetError, OSError) as exc:
            self._raise_worker_failure(idx, exc)
        if status == "error":
            raise SubprocWorkerError(f"Worker {idx} failed:\n{payload}")
        return payload

    def _ensure_worker_alive(self, idx: int) -> None:
        if not self._processes[idx].is_alive():
            self._raise_worker_failure(idx)

    def _raise_worker_failure(
        self,
        idx: int,
        cause: BaseException | None = None,
    ) -> None:
        message = (
            f"Worker {idx} is not healthy ({_worker_status(self._processes[idx])})."
        )
        if cause is None:
            raise SubprocWorkerError(message)
        raise SubprocWorkerError(message) from cause

    def _reap_workers(self, *, join_timeout: float) -> None:
        for process in self._processes:
            if not process.is_alive():
                process.join(timeout=0)
                continue
            process.join(timeout=join_timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=join_timeout)
            if process.is_alive():
                process.kill()
                process.join(timeout=join_timeout)


def make_task_runner(
    task_factory: Callable[[], Any],
    *,
    num_envs: int,
    env_device: str,
    env_runner: str,
    mp_start_method: str,
):
    if env_device == "cuda":
        if env_runner not in {"warp", "subproc"}:
            raise ValueError(
                "CUDA environment stepping uses the Warp runner; "
                f"got env_runner={env_runner!r}."
            )
        from flygym_demo.odor_obstacle_navigation.warp_runner import (
            WarpNavigationTaskRunner,
        )

        config = getattr(task_factory, "config", None)
        if config is None:
            raise TypeError(
                "WarpNavigationTaskRunner requires a NavigationTaskFactory."
            )
        return WarpNavigationTaskRunner(config, num_envs=num_envs)
    if env_device != "cpu":
        raise ValueError(f"Unsupported env_device: {env_device}")
    if env_runner == "sync":
        return SyncTaskRunner(task_factory, num_envs)
    if env_runner == "subproc":
        return SubprocTaskRunner(
            task_factory,
            num_envs,
            mp_start_method=mp_start_method,
        )
    raise ValueError(f"Unsupported env_runner: {env_runner}")


def _task_worker(conn, task_factory: Callable[[], Any], rank: int) -> None:
    from flygym.utils.stderr_filter import install_ccd_iterations_warning_filter

    install_ccd_iterations_warning_filter()
    task = None
    try:
        task = task_factory()
        while True:
            command, payload = conn.recv()
            if command == "spec":
                conn.send(("ok", _task_spec(task)))
            elif command == "diagnostics":
                conn.send(("ok", _task_diagnostics(task, rank=rank)))
            elif command == "reset":
                obs, info = task.reset(seed=payload)
                conn.send(("ok", (np.asarray(obs, dtype=np.float32), info)))
            elif command == "step":
                obs, reward, terminated, truncated, info = task.step(payload)
                conn.send(
                    (
                        "ok",
                        (
                            np.asarray(obs, dtype=np.float32),
                            float(reward),
                            bool(terminated),
                            bool(truncated),
                            info,
                        ),
                    )
                )
            elif command == "close":
                conn.send(("ok", None))
                return
            else:
                raise ValueError(f"Unknown worker command: {command}")
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        if task is not None:
            task.close()
        conn.close()


def _task_spec(task) -> TaskRunnerSpec:
    config = getattr(task, "config", None)
    return TaskRunnerSpec(
        action_space=_task_action_space(task),
        action_low=np.asarray(task.action_low, dtype=np.float32),
        action_high=np.asarray(task.action_high, dtype=np.float32),
        action_shape=tuple(getattr(task, "action_shape", task.action_low.shape)),
        action_n=getattr(task, "action_n", None),
        action_max_norm=getattr(task, "action_max_norm", None),
        observation_shape=tuple(getattr(task, "observation_shape", ())),
        decision_steps=float(getattr(config, "decision_steps", 1)),
        decision_interval=float(getattr(config, "decision_interval", 1.0)),
    )


def _task_diagnostics(task, *, rank: int) -> dict[str, Any]:
    config = getattr(task, "config", None)
    return {
        "rank": int(rank),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "cpu_affinity": _cpu_affinity(),
        "cpu_affinity_count": _cpu_affinity_count(),
        "cpu_seconds": _process_cpu_seconds(os.getpid()),
        "rss_bytes": _process_rss_bytes(os.getpid()),
        "wall_time_seconds": time.time(),
        "observation_shape": tuple(getattr(task, "observation_shape", ())),
        "action_space": _task_action_space(task),
        "action_max_norm": getattr(task, "action_max_norm", None),
        "decision_steps": int(getattr(config, "decision_steps", 0)),
        "decision_interval": float(getattr(config, "decision_interval", 0.0)),
        "enable_vision": bool(getattr(config, "enable_vision", False)),
        "enable_olfaction": bool(getattr(config, "enable_olfaction", False)),
        "randomize_layout": bool(getattr(config, "randomize_layout", False)),
    }


def _task_action_space(task) -> str:
    action_space = getattr(task, "action_space_name", None)
    if action_space is not None:
        return str(action_space)
    config = getattr(task, "config", None)
    action_space = getattr(config, "action_space", None)
    if action_space is not None:
        return str(action_space)
    action_space = getattr(task, "action_space", "continuous")
    if isinstance(action_space, str):
        return action_space
    return "continuous"


def _cpu_affinity() -> list[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    return sorted(os.sched_getaffinity(0))


def _cpu_affinity_count() -> int | None:
    affinity = _cpu_affinity()
    return None if affinity is None else len(affinity)


def _process_cpu_seconds(pid: int) -> float | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", maxsplit=1)[1].split()
        ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return (float(fields[11]) + float(fields[12])) / ticks_per_second
    except (FileNotFoundError, IndexError, KeyError, OSError, ValueError):
        return None


def _process_rss_bytes(pid: int) -> int | None:
    try:
        statm = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()
        return int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _worker_status(process: mp.Process) -> str:
    if process.is_alive():
        return "alive"
    exitcode = process.exitcode
    if exitcode is None:
        return "not started"
    if exitcode == 0:
        return "exited cleanly"
    if exitcode < 0:
        try:
            signal_name = signal.Signals(-exitcode).name
        except (ValueError, AttributeError):
            signal_name = str(-exitcode)
        return f"killed by signal {signal_name} ({exitcode})"
    return f"exited with code {exitcode}"


def _indexed_seeds(seed: int | None, n: int) -> list[int | None]:
    if seed is None:
        return [None] * n
    return [seed + idx for idx in range(n)]


def _empty_obs(observation_shape: tuple[int, ...]) -> np.ndarray:
    return np.empty((0, *observation_shape), dtype=np.float32)
