from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import traceback
from typing import Any, Callable

import numpy as np


@dataclass(slots=True)
class TaskRunnerSpec:
    action_low: np.ndarray
    action_high: np.ndarray
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


class SubprocTaskRunner:
    def __init__(
        self,
        task_factory: Callable[[], Any],
        num_envs: int,
        *,
        mp_start_method: str = "spawn",
    ) -> None:
        self.num_envs = num_envs
        self._ctx = mp.get_context(mp_start_method)
        self._closed = False
        self._conns = []
        self._processes = []
        for _ in range(num_envs):
            parent_conn, child_conn = self._ctx.Pipe()
            process = self._ctx.Process(
                target=_task_worker,
                args=(child_conn, task_factory),
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
            self._conns[idx].send(("reset", seed))
        results = [self._recv(idx) for idx in indices]
        obs, infos = zip(*results, strict=True)
        return np.stack(obs).astype(np.float32), list(infos)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        for conn, action in zip(self._conns, np.asarray(actions), strict=True):
            conn.send(("step", np.asarray(action, dtype=np.float32)))
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
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for idx, conn in enumerate(self._conns):
            try:
                self._recv(idx)
            except RuntimeError:
                pass
            conn.close()
        for process in self._processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    def _request_specs(self) -> list[TaskRunnerSpec]:
        for conn in self._conns:
            conn.send(("spec", None))
        return [self._recv(idx) for idx in range(self.num_envs)]

    def _recv(self, idx: int):
        try:
            status, payload = self._conns[idx].recv()
        except EOFError as exc:
            raise RuntimeError(f"Worker {idx} closed unexpectedly.") from exc
        if status == "error":
            raise RuntimeError(f"Worker {idx} failed:\n{payload}")
        return payload


def make_task_runner(
    task_factory: Callable[[], Any],
    *,
    num_envs: int,
    env_runner: str,
    mp_start_method: str,
):
    if env_runner == "sync":
        return SyncTaskRunner(task_factory, num_envs)
    if env_runner == "subproc":
        return SubprocTaskRunner(
            task_factory,
            num_envs,
            mp_start_method=mp_start_method,
        )
    raise ValueError(f"Unsupported env_runner: {env_runner}")


def _task_worker(conn, task_factory: Callable[[], Any]) -> None:
    task = None
    try:
        task = task_factory()
        while True:
            command, payload = conn.recv()
            if command == "spec":
                conn.send(("ok", _task_spec(task)))
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
        action_low=np.asarray(task.action_low, dtype=np.float32),
        action_high=np.asarray(task.action_high, dtype=np.float32),
        observation_shape=tuple(getattr(task, "observation_shape", ())),
        decision_steps=float(getattr(config, "decision_steps", 1)),
        decision_interval=float(getattr(config, "decision_interval", 1.0)),
    )


def _indexed_seeds(seed: int | None, n: int) -> list[int | None]:
    if seed is None:
        return [None] * n
    return [seed + idx for idx in range(n)]


def _empty_obs(observation_shape: tuple[int, ...]) -> np.ndarray:
    return np.empty((0, *observation_shape), dtype=np.float32)
