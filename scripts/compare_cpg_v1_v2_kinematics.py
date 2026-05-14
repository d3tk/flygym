#!/usr/bin/env python3
"""Compare open-loop CPG kinematics: FlyGym v1 reference vs v2.

Steps the v1 and v2 CPG networks with the same parameters and records:

  - oscillator phases and magnitudes (should match if implementations agree);
  - per-timestep **commanded** leg joint angles from each version's
    ``PreprogrammedSteps`` (same ``single_steps_untethered.pkl`` path by default);
  - finite-difference angular speeds (L2 norm across all actuated DOFs per step).

**Fair v1 vs v2 MuJoCo defaults:** v2 uses ``actuator_gain=45``, ``adhesion_gain=40``,
``spawn_z=0.5`` mm, ``warmup_duration=0.05`` s, ``forcerange`` ±65, v1-like leg
compliance from ``make_locomotion_fly``, and leg-only floor collisions matching v1
``Fly(floor_collisions="legs")``. The v2 camera is body-fixed with an offset matching
v1’s ``camera_right`` preset. Override with CLI flags if you want Tutorial-2-style
higher gains or v1-like contact solver parameters.

When recording video (default), the script also writes a **side-by-side MP4** (v1
gymnasium fly vs v2 fly). The NPZ then includes ``sim_v1_*`` and ``sim_v2_*`` time
series: joint command vs achieved state, thorax position, v2 actuator forces and
ground-contact sensors, tarsus5 contact forces, endpoint positions, and v1
observation-derived torques / contact / end-effector positions. A plain-text
``*_kinematics_compare.txt`` summarizes open-loop vs MuJoCo metrics side by side.

Optional dependencies for the v1 half of the video::

    uv sync --dev --extra cpg-v1-compare

The v1 *locomotion* modules used for the **NPZ** (``steps.py`` / ``cpg_controller.py``)
are loaded from ``.flygym-1.0-ref`` via ``importlib`` and minimal ``sys.modules`` shims
so that path never imports ``gymnasium``. Recording the MP4 **purges** those shims and
imports the full ``flygym_gymnasium`` stack (which requires ``gymnasium``).

Example (include optional deps for MP4)::

    uv run --extra cpg-v1-compare python scripts/compare_cpg_v1_v2_kinematics.py \\
        --duration 1.0 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import types
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_v1_locomotion_modules(ref_root: Path) -> tuple[types.ModuleType, types.ModuleType]:
    """Load v1 ``steps`` (PreprogrammedSteps) and ``cpg_controller`` (CPGNetwork) without gymnasium."""
    fg = "flygym_gymnasium"
    ex = f"{fg}.examples"
    loc = f"{ex}.locomotion"

    for name in (fg, ex, loc):
        sys.modules.setdefault(name, types.ModuleType(name))

    loc_mod = sys.modules[loc]
    loc_mod.__path__ = [str(ref_root / "flygym_gymnasium/examples/locomotion")]

    util = types.ModuleType(f"{fg}.util")

    def get_data_path(package: str, file: str) -> Path:
        return ref_root / "flygym_gymnasium" / file

    util.get_data_path = get_data_path  # type: ignore[attr-defined]
    sys.modules[f"{fg}.util"] = util

    steps_path = ref_root / "flygym_gymnasium/examples/locomotion/steps.py"
    spec = importlib.util.spec_from_file_location(f"{loc}.steps", steps_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load spec for {steps_path}")
    steps_mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{loc}.steps"] = steps_mod
    spec.loader.exec_module(steps_mod)
    loc_mod.PreprogrammedSteps = steps_mod.PreprogrammedSteps

    cpg_path = ref_root / "flygym_gymnasium/examples/locomotion/cpg_controller.py"
    spec2 = importlib.util.spec_from_file_location(f"{loc}.cpg_controller", cpg_path)
    if spec2 is None or spec2.loader is None:
        raise RuntimeError(f"Cannot load spec for {cpg_path}")
    cpg_mod = importlib.util.module_from_spec(spec2)
    sys.modules[f"{loc}.cpg_controller"] = cpg_mod
    spec2.loader.exec_module(cpg_mod)
    loc_mod.CPGNetwork = cpg_mod.CPGNetwork

    return steps_mod, cpg_mod


def _concat_perm_to_doforder(pre_v2, dof_order: list) -> tuple["np.ndarray", list]:
    import numpy as np
    from flygym.examples.locomotion.preprogrammed import _dof_spec_to_jointdof

    legs_v2 = list(pre_v2.legs)
    concat_index_by_dof: dict[object, int] = {}
    for li, leg in enumerate(legs_v2):
        for dof_i, spec in enumerate(pre_v2.dofs_per_leg):
            jd = _dof_spec_to_jointdof(leg, spec)
            concat_index_by_dof[jd] = li * 7 + dof_i
    perm = np.array([concat_index_by_dof[d] for d in dof_order], dtype=np.int32)
    return perm, legs_v2


def _purge_flygym_gymnasium_from_sys_modules() -> None:
    """Remove stub or partial ``flygym_gymnasium`` entries so the real package can load."""
    for name in list(sys.modules):
        if name == "flygym_gymnasium" or name.startswith("flygym_gymnasium."):
            del sys.modules[name]


def _v1_actuated_joint_columns(fly) -> list[int]:
    """Map original actuated joint column index -> column in ``obs['joints']``."""
    import numpy as np

    order = np.asarray(fly._monitored_joint_order)
    cols = []
    for j in range(len(fly.actuated_joints)):
        hit = np.where(order == j)[0]
        if hit.size != 1:
            raise RuntimeError("unexpected monitored_joint_order mapping for v1 fly")
        cols.append(int(hit[0]))
    return cols


def _load_v1_cpg_network_class(ref_root: Path):
    """Load v1 ``CPGNetwork`` without importing ``examples/locomotion/__init__.py``."""
    import importlib.util
    import types

    loc_dir = ref_root / "flygym_gymnasium/examples/locomotion"
    steps_path = loc_dir / "steps.py"
    cpg_path = loc_dir / "cpg_controller.py"

    if "flygym_gymnasium.examples" not in sys.modules:
        ex = types.ModuleType("flygym_gymnasium.examples")
        ex.__path__ = [str(ref_root / "flygym_gymnasium/examples")]
        sys.modules["flygym_gymnasium.examples"] = ex

    spec_s = importlib.util.spec_from_file_location(
        "flygym_gymnasium.examples.locomotion.steps", steps_path
    )
    if spec_s is None or spec_s.loader is None:
        raise RuntimeError(f"Cannot load {steps_path}")
    steps_mod = importlib.util.module_from_spec(spec_s)
    sys.modules["flygym_gymnasium.examples.locomotion.steps"] = steps_mod
    spec_s.loader.exec_module(steps_mod)

    loc = types.ModuleType("flygym_gymnasium.examples.locomotion")
    loc.__path__ = [str(loc_dir)]
    loc.PreprogrammedSteps = steps_mod.PreprogrammedSteps
    sys.modules["flygym_gymnasium.examples.locomotion"] = loc

    spec_c = importlib.util.spec_from_file_location(
        "flygym_gymnasium.examples.locomotion.cpg_controller", cpg_path
    )
    if spec_c is None or spec_c.loader is None:
        raise RuntimeError(f"Cannot load {cpg_path}")
    cpg_mod = importlib.util.module_from_spec(spec_c)
    sys.modules["flygym_gymnasium.examples.locomotion.cpg_controller"] = cpg_mod
    spec_c.loader.exec_module(cpg_mod)

    return steps_mod.PreprogrammedSteps, cpg_mod.CPGNetwork


def _run_v1_gymnasium_cpg_rollout_frames(
    *,
    ref_root: Path,
    pickle_path: Path,
    dt: float,
    n_steps: int,
    seed: int,
    intrinsic_frequency: float,
    coupling_strength: float,
    phase_biases: "np.ndarray",
    spawn_z_mm: float,
    adhesion_force: float,
    warmup_duration: float,
    cam_play_speed: float,
    cam_fps: int,
    floor_collisions: str = "legs",
) -> tuple[list, dict]:
    """Run v1 CPG + v1 ``SingleFlySimulation``; RGB frames + per-step MuJoCo logs."""
    import numpy as np

    _purge_flygym_gymnasium_from_sys_modules()
    ref_str = str(ref_root.resolve())
    if ref_str in sys.path:
        sys.path.remove(ref_str)
    sys.path.insert(0, ref_str)

    try:
        import gymnasium  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "v1 MuJoCo side-by-side video needs gymnasium. Install with:\n"
            "  uv sync --dev --extra cpg-v1-compare\n"
            "or: uv pip install gymnasium opencv-python-headless"
        ) from e
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "v1 cameras need OpenCV. Install with:\n"
            "  uv sync --dev --extra cpg-v1-compare"
        ) from e

    from flygym_gymnasium import Fly, SingleFlySimulation, YawOnlyCamera

    PreprogrammedSteps, CPGNetwork = _load_v1_cpg_network_class(ref_root)

    pre = PreprogrammedSteps(path=str(pickle_path))
    pb = np.asarray(phase_biases, dtype=np.float64)
    nu = np.ones(6, dtype=np.float64) * intrinsic_frequency
    R = np.ones(6, dtype=np.float64)
    w = (pb > 0).astype(np.float64) * coupling_strength
    alpha = np.ones(6, dtype=np.float64) * 20.0
    cpg = CPGNetwork(
        dt,
        intrinsic_freqs=nu,
        intrinsic_amps=R,
        coupling_weights=w,
        phase_biases=pb,
        convergence_coefs=alpha,
        seed=seed,
    )

    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        init_pose="stretch",
        control="position",
        spawn_pos=(0.0, 0.0, float(spawn_z_mm)),
        adhesion_force=float(adhesion_force),
        actuator_gain=45.0,
        floor_collisions=floor_collisions,
    )
    cam = YawOnlyCamera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_right",
        targeted_fly_names=fly.name,
        play_speed=float(cam_play_speed),
        fps=int(cam_fps),
        window_size=(320, 240),
        play_speed_text=False,
        timestamp_text=False,
    )
    sim = SingleFlySimulation(fly=fly, cameras=[cam], timestep=dt)
    sim.reset()

    settle_n = max(0, int(round(float(warmup_duration) / dt)))
    if settle_n > 0:
        ph = np.full(6, np.pi)
        one = np.ones(6)
        for _ in range(settle_n):
            jp = []
            adh: list[bool] = []
            for i, leg in enumerate(pre.legs):
                jp.append(
                    np.asarray(pre.get_joint_angles(leg, ph[i], one[i]), dtype=np.float64).reshape(
                        -1
                    )
                )
                adh.append(bool(pre.get_adhesion_onoff(leg, float(ph[i]))))
            sim.step(
                {
                    "joints": np.concatenate(jp),
                    "adhesion": np.array(adh, dtype=np.int32),
                }
            )

    n_act = len(fly.actuated_joints)
    joint_cols = _v1_actuated_joint_columns(fly)

    logs = {
        "joint_cmd": np.zeros((n_steps, n_act), dtype=np.float64),
        "joint_pos": np.zeros((n_steps, n_act), dtype=np.float64),
        "joint_vel": np.zeros((n_steps, n_act), dtype=np.float64),
        "joint_torque": np.zeros((n_steps, n_act), dtype=np.float64),
        "thorax_pos": np.zeros((n_steps, 3), dtype=np.float64),
        "thorax_rot_euler": np.zeros((n_steps, 3), dtype=np.float64),
        "ee_pos": np.zeros((n_steps, 6, 3), dtype=np.float64),
        "adhesion_cmd": np.zeros((n_steps, 6), dtype=np.float64),
        "contact_force_norm_mean": np.zeros(n_steps, dtype=np.float64),
        "ee_height_mean": np.zeros(n_steps, dtype=np.float64),
    }

    for t in range(n_steps):
        cpg.step()
        joints_angles = []
        adhesion_onoff: list[bool] = []
        for i, leg in enumerate(pre.legs):
            ja = pre.get_joint_angles(leg, cpg.curr_phases[i], cpg.curr_magnitudes[i])
            joints_angles.append(np.asarray(ja, dtype=np.float64).reshape(-1))
            adhesion_onoff.append(
                bool(pre.get_adhesion_onoff(leg, float(cpg.curr_phases[i])))
            )
        cmd = np.concatenate(joints_angles)
        action = {
            "joints": cmd,
            "adhesion": np.array(adhesion_onoff, dtype=np.int32),
        }
        obs, _r, _term, _trunc, _info = sim.step(action)
        sim.render()

        jmat = obs["joints"]
        logs["joint_cmd"][t] = cmd
        logs["joint_pos"][t] = jmat[0, joint_cols]
        logs["joint_vel"][t] = jmat[1, joint_cols]
        logs["joint_torque"][t] = jmat[2, joint_cols]
        logs["thorax_pos"][t] = obs["fly"][0].astype(np.float64)
        logs["thorax_rot_euler"][t] = obs["fly"][2].astype(np.float64)
        logs["ee_pos"][t] = obs["end_effectors"].astype(np.float64)
        logs["adhesion_cmd"][t] = np.asarray(adhesion_onoff, dtype=np.float64)
        cf = obs["contact_forces"]
        logs["contact_force_norm_mean"][t] = float(np.linalg.norm(cf))
        logs["ee_height_mean"][t] = float(np.mean(obs["end_effectors"][:, 2]))

    return list(cam._frames), logs


def _simulate_rollout_with_logs(
    sim,
    fly,
    *,
    default_pose: "np.ndarray",
    joint_dof: "np.ndarray",
    adhesion: "np.ndarray",
    camera_key: str,
    dof_order: list,
    warmup_duration: float,
) -> tuple[list, dict]:
    """v2 open-loop rollout: frames + per-step MuJoCo kinematics / contact / effort."""
    import numpy as np
    from flygym.anatomy import BodySegment
    from flygym.compose.fly import ActuatorType
    from flygym.examples.locomotion.common import LocomotionAction, apply_locomotion_action

    jd_all = fly.get_jointdofs_order()
    idx = np.array([jd_all.index(d) for d in dof_order], dtype=np.int32)
    body_order = fly.get_bodysegs_order()
    thorax_i = body_order.index(BodySegment("c_thorax"))
    tarsus5_i = np.array(
        [body_order.index(BodySegment(f"{leg}_tarsus5")) for leg in fly.get_legs_order()],
        dtype=np.int32,
    )
    tarsus5_segments = [BodySegment(f"{leg}_tarsus5") for leg in fly.get_legs_order()]

    n = joint_dof.shape[0]
    if adhesion.shape[0] != n:
        raise ValueError("adhesion rows must match joint_dof rows")

    logs = {
        "joint_cmd": np.zeros((n, 42), dtype=np.float64),
        "joint_pos": np.zeros((n, 42), dtype=np.float64),
        "joint_vel": np.zeros((n, 42), dtype=np.float64),
        "thorax_pos": np.zeros((n, 3), dtype=np.float64),
        "thorax_quat": np.zeros((n, 4), dtype=np.float64),
        "ee_pos": np.zeros((n, 6, 3), dtype=np.float64),
        "adhesion_cmd": np.zeros((n, 6), dtype=np.float64),
        "actuator_force": np.zeros((n, 42), dtype=np.float64),
        "contact_flag": np.zeros((n, 6), dtype=np.float64),
        "contact_force": np.zeros((n, 6, 3), dtype=np.float64),
        "contact_force_norm": np.zeros((n, 6), dtype=np.float64),
        "tarsus5_contact_force": np.zeros((n, 6, 3), dtype=np.float64),
    }
    contact_sensor_ids = getattr(sim, "_intern_groundcontactsensorids_by_fly", None)
    has_ground_contact_sensors = (
        contact_sensor_ids is not None
        and fly.name in contact_sensor_ids
        and len(contact_sensor_ids[fly.name]) > 0
    )

    sim.reset()
    apply_locomotion_action(
        sim,
        fly.name,
        LocomotionAction(
            joint_angles=default_pose,
            adhesion_onoff=np.ones(6, dtype=bool),
        ),
    )
    sim.warmup(duration_s=float(warmup_duration))

    for t in range(n):
        apply_locomotion_action(
            sim,
            fly.name,
            LocomotionAction(
                joint_angles=joint_dof[t],
                adhesion_onoff=adhesion[t],
            ),
        )
        sim.step()
        body_pos = sim.get_body_positions(fly.name)
        logs["joint_cmd"][t] = joint_dof[t]
        logs["joint_pos"][t] = sim.get_joint_angles(fly.name)[idx]
        logs["joint_vel"][t] = sim.get_joint_velocities(fly.name)[idx]
        logs["thorax_pos"][t] = body_pos[thorax_i]
        logs["thorax_quat"][t] = sim.get_body_rotations(fly.name)[thorax_i]
        logs["ee_pos"][t] = body_pos[tarsus5_i]
        logs["adhesion_cmd"][t] = adhesion[t].astype(np.float64)
        logs["actuator_force"][t] = sim.get_actuator_forces(fly.name, ActuatorType.POSITION)
        if has_ground_contact_sensors:
            cflag, cforces, *_ = sim.get_ground_contact_info(fly.name)
            logs["contact_flag"][t] = cflag
            logs["contact_force"][t] = cforces
            logs["contact_force_norm"][t] = np.linalg.norm(cforces, axis=1)
        logs["tarsus5_contact_force"][t] = sim.get_bodysegment_contact_forces(
            fly.name, tarsus5_segments
        )
        sim.render_as_needed()

    frames = sim.renderer.frames[camera_key]
    return list(frames), logs


def _run_v2_fk_with_logs(
    sim,
    fly,
    *,
    joint_dof: "np.ndarray",
    dof_order: list,
) -> dict:
    """Set v2 commanded poses and run MuJoCo forward kinematics without stepping."""
    import mujoco as mj
    import numpy as np
    from flygym.anatomy import BodySegment

    jd_all = fly.get_jointdofs_order()
    qpos_idx = np.array([jd_all.index(d) for d in dof_order], dtype=np.int32)
    qpos_adrs = sim._intern_qposadrs_by_fly[fly.name][qpos_idx]
    body_order = fly.get_bodysegs_order()
    thorax_i = body_order.index(BodySegment("c_thorax"))
    tarsus5_i = np.array(
        [body_order.index(BodySegment(f"{leg}_tarsus5")) for leg in fly.get_legs_order()],
        dtype=np.int32,
    )

    n = joint_dof.shape[0]
    logs = {
        "joint_cmd": np.asarray(joint_dof, dtype=np.float64).copy(),
        "joint_pos": np.zeros((n, 42), dtype=np.float64),
        "joint_vel": np.zeros((n, 42), dtype=np.float64),
        "thorax_pos": np.zeros((n, 3), dtype=np.float64),
        "thorax_quat": np.zeros((n, 4), dtype=np.float64),
        "ee_pos": np.zeros((n, 6, 3), dtype=np.float64),
    }

    sim.reset()
    sim.mj_data.qvel[:] = 0
    for t in range(n):
        sim.mj_data.qpos[qpos_adrs] = joint_dof[t]
        mj.mj_forward(sim.mj_model, sim.mj_data)
        body_pos = sim.get_body_positions(fly.name)
        logs["joint_pos"][t] = sim.get_joint_angles(fly.name)[qpos_idx]
        logs["thorax_pos"][t] = body_pos[thorax_i]
        logs["thorax_quat"][t] = sim.get_body_rotations(fly.name)[thorax_i]
        logs["ee_pos"][t] = body_pos[tarsus5_i]
    return logs


def _simulate_v2_fixed_pose_contact_logs(
    sim,
    fly,
    *,
    joint_pose: "np.ndarray",
    dof_order: list,
    n_steps: int,
    warmup_duration: float,
) -> dict:
    """Run v2 at one commanded pose and log contact without camera side effects."""
    import numpy as np
    from flygym.anatomy import BodySegment
    from flygym.compose.fly import ActuatorType
    from flygym.examples.locomotion.common import LocomotionAction, apply_locomotion_action

    jd_all = fly.get_jointdofs_order()
    idx = np.array([jd_all.index(d) for d in dof_order], dtype=np.int32)
    body_order = fly.get_bodysegs_order()
    thorax_i = body_order.index(BodySegment("c_thorax"))
    tarsus5_i = np.array(
        [body_order.index(BodySegment(f"{leg}_tarsus5")) for leg in fly.get_legs_order()],
        dtype=np.int32,
    )
    tarsus5_segments = [BodySegment(f"{leg}_tarsus5") for leg in fly.get_legs_order()]

    logs = {
        "joint_cmd": np.tile(np.asarray(joint_pose, dtype=np.float64), (n_steps, 1)),
        "joint_pos": np.zeros((n_steps, 42), dtype=np.float64),
        "joint_vel": np.zeros((n_steps, 42), dtype=np.float64),
        "thorax_pos": np.zeros((n_steps, 3), dtype=np.float64),
        "thorax_quat": np.zeros((n_steps, 4), dtype=np.float64),
        "ee_pos": np.zeros((n_steps, 6, 3), dtype=np.float64),
        "adhesion_cmd": np.ones((n_steps, 6), dtype=np.float64),
        "actuator_force": np.zeros((n_steps, 42), dtype=np.float64),
        "contact_flag": np.zeros((n_steps, 6), dtype=np.float64),
        "contact_force": np.zeros((n_steps, 6, 3), dtype=np.float64),
        "contact_force_norm": np.zeros((n_steps, 6), dtype=np.float64),
        "tarsus5_contact_force": np.zeros((n_steps, 6, 3), dtype=np.float64),
    }
    contact_sensor_ids = getattr(sim, "_intern_groundcontactsensorids_by_fly", None)
    has_ground_contact_sensors = (
        contact_sensor_ids is not None
        and fly.name in contact_sensor_ids
        and len(contact_sensor_ids[fly.name]) > 0
    )

    sim.reset()
    apply_locomotion_action(
        sim,
        fly.name,
        LocomotionAction(
            joint_angles=joint_pose,
            adhesion_onoff=np.ones(6, dtype=bool),
        ),
    )
    sim.warmup(duration_s=float(warmup_duration))

    for t in range(n_steps):
        apply_locomotion_action(
            sim,
            fly.name,
            LocomotionAction(
                joint_angles=joint_pose,
                adhesion_onoff=np.ones(6, dtype=bool),
            ),
        )
        sim.step()
        body_pos = sim.get_body_positions(fly.name)
        logs["joint_pos"][t] = sim.get_joint_angles(fly.name)[idx]
        logs["joint_vel"][t] = sim.get_joint_velocities(fly.name)[idx]
        logs["thorax_pos"][t] = body_pos[thorax_i]
        logs["thorax_quat"][t] = sim.get_body_rotations(fly.name)[thorax_i]
        logs["ee_pos"][t] = body_pos[tarsus5_i]
        logs["actuator_force"][t] = sim.get_actuator_forces(fly.name, ActuatorType.POSITION)
        if has_ground_contact_sensors:
            cflag, cforces, *_ = sim.get_ground_contact_info(fly.name)
            logs["contact_flag"][t] = cflag
            logs["contact_force"][t] = cforces
            logs["contact_force_norm"][t] = np.linalg.norm(cforces, axis=1)
        logs["tarsus5_contact_force"][t] = sim.get_bodysegment_contact_forces(
            fly.name, tarsus5_segments
        )
    return logs


def _run_v1_fk_with_logs(
    *,
    ref_root: Path,
    pickle_path: Path,
    joint_cmd: "np.ndarray",
    spawn_z_mm: float,
) -> dict:
    """Set v1 commanded poses and run dm_control forward kinematics without stepping."""
    import numpy as np

    _purge_flygym_gymnasium_from_sys_modules()
    ref_str = str(ref_root.resolve())
    if ref_str in sys.path:
        sys.path.remove(ref_str)
    sys.path.insert(0, ref_str)

    from flygym_gymnasium import Fly, SingleFlySimulation

    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=False,
        init_pose="stretch",
        control="position",
        spawn_pos=(0.0, 0.0, float(spawn_z_mm)),
        adhesion_force=40.0,
        actuator_gain=45.0,
        floor_collisions="none",
    )
    sim = SingleFlySimulation(fly=fly, cameras=[], timestep=1e-4)
    sim.reset()
    joint_cols = _v1_actuated_joint_columns(fly)

    n = joint_cmd.shape[0]
    logs = {
        "joint_cmd": np.asarray(joint_cmd, dtype=np.float64).copy(),
        "joint_pos": np.zeros((n, joint_cmd.shape[1]), dtype=np.float64),
        "joint_vel": np.zeros((n, joint_cmd.shape[1]), dtype=np.float64),
        "thorax_pos": np.zeros((n, 3), dtype=np.float64),
        "thorax_rot_euler": np.zeros((n, 3), dtype=np.float64),
        "ee_pos": np.zeros((n, 6, 3), dtype=np.float64),
    }
    joint_qpos_names = [f"{fly.name}/{joint}" for joint in fly.actuated_joints]

    for t in range(n):
        for qpos_name, value in zip(joint_qpos_names, joint_cmd[t]):
            sim.physics.named.data.qpos[qpos_name] = value
        sim.physics.data.qvel[:] = 0
        sim.physics.forward()
        obs = fly.get_observation(sim)
        logs["joint_pos"][t] = obs["joints"][0, joint_cols]
        logs["thorax_pos"][t] = obs["fly"][0].astype(np.float64)
        logs["thorax_rot_euler"][t] = obs["fly"][2].astype(np.float64)
        logs["ee_pos"][t] = obs["end_effectors"].astype(np.float64)
    return logs


def _endpoint_relative_to_thorax(logs: dict) -> "np.ndarray":
    import numpy as np

    return logs["ee_pos"] - logs["thorax_pos"][:, np.newaxis, :]


def _thorax_velocity(logs: dict, dt: float) -> "np.ndarray":
    import numpy as np

    if len(logs["thorax_pos"]) < 2:
        return np.zeros_like(logs["thorax_pos"])
    return np.gradient(logs["thorax_pos"], float(dt), axis=0)


def _quat_wxyz_to_euler_xyz(quat: "np.ndarray") -> "np.ndarray":
    """Convert MuJoCo wxyz quaternions to xyz fixed-axis Euler angles."""
    import numpy as np

    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(
        np.abs(sinp) >= 1,
        np.sign(sinp) * (np.pi / 2),
        np.arcsin(sinp),
    )

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=1)


def _thorax_euler_xyz(logs: dict) -> "np.ndarray":
    import numpy as np

    if "thorax_rot_euler" in logs:
        return np.asarray(logs["thorax_rot_euler"], dtype=np.float64)
    if "thorax_quat" in logs:
        return _quat_wxyz_to_euler_xyz(logs["thorax_quat"])
    return np.zeros((len(logs["thorax_pos"]), 3), dtype=np.float64)


def _sim_tracking_stats(logs: dict, *, dt: float | None = None) -> dict:
    """Scalar summaries for one simulator log dict."""
    import numpy as np

    legs = ("lf", "lm", "lh", "rf", "rm", "rh")
    err = logs["joint_cmd"] - logs["joint_pos"]
    rmse_t = np.sqrt(np.mean(err**2, axis=1))
    vel_l2 = np.linalg.norm(logs["joint_vel"], axis=1)
    thorax = logs["thorax_pos"]
    thorax_euler = _thorax_euler_xyz(logs)
    dx_mm = float(thorax[-1, 0] - thorax[0, 0])
    dy_mm = float(thorax[-1, 1] - thorax[0, 1])
    dz_mm = float(thorax[-1, 2] - thorax[0, 2])
    out = {
        "mean_rmse_cmd_vs_pos_rad": float(rmse_t.mean()),
        "max_rmse_cmd_vs_pos_rad": float(rmse_t.max()),
        "mean_L2_joint_vel": float(vel_l2.mean()),
        "thorax_delta_x_mm": dx_mm,
        "thorax_delta_y_mm": dy_mm,
        "thorax_delta_z_mm": dz_mm,
        "thorax_mean_height_mm": float(np.mean(thorax[:, 2])),
        "thorax_min_height_mm": float(np.min(thorax[:, 2])),
        "thorax_max_height_mm": float(np.max(thorax[:, 2])),
        "thorax_roll_delta_rad": float(thorax_euler[-1, 0] - thorax_euler[0, 0]),
        "thorax_pitch_delta_rad": float(thorax_euler[-1, 1] - thorax_euler[0, 1]),
        "thorax_yaw_delta_rad": float(thorax_euler[-1, 2] - thorax_euler[0, 2]),
    }
    if dt is not None:
        thorax_vel = _thorax_velocity(logs, dt)
        out.update(
            {
                "thorax_mean_forward_velocity_mm_s": float(np.mean(thorax_vel[:, 0])),
                "thorax_mean_lateral_velocity_mm_s": float(np.mean(thorax_vel[:, 1])),
                "thorax_mean_vertical_velocity_mm_s": float(np.mean(thorax_vel[:, 2])),
                "thorax_rms_velocity_mm_s": float(
                    np.sqrt(np.mean(np.sum(thorax_vel**2, axis=1)))
                ),
            }
        )
    if "contact_flag" in logs:
        out["mean_contact_frac"] = float(np.mean(logs["contact_flag"]))
        out["contact_frac_by_leg"] = [
            float(np.mean(logs["contact_flag"][:, i])) for i in range(len(legs))
        ]
        out["contact_frac_by_leg_labels"] = list(legs)
    if "contact_force_norm_mean" in logs:
        out["mean_contact_force_scalar"] = float(np.mean(logs["contact_force_norm_mean"]))
    if "contact_force_norm" in logs and "contact_force_norm_mean" not in logs:
        out["mean_contact_force_scalar"] = float(np.mean(logs["contact_force_norm"]))
        out["mean_contact_force_norm_by_leg"] = [
            float(np.mean(logs["contact_force_norm"][:, i])) for i in range(len(legs))
        ]
    if "ee_height_mean" in logs:
        out["mean_ee_height_mm"] = float(np.mean(logs["ee_height_mean"]))
    if "ee_pos" in logs:
        ee_rel = _endpoint_relative_to_thorax(logs)
        out["mean_ee_rel_z_mm"] = float(np.mean(ee_rel[:, :, 2]))
        out["mean_ee_rel_x_span_mm"] = float(np.mean(np.ptp(ee_rel[:, :, 0], axis=0)))
        out["mean_ee_rel_y_span_mm"] = float(np.mean(np.ptp(ee_rel[:, :, 1], axis=0)))
        out["mean_ee_rel_z_span_mm"] = float(np.mean(np.ptp(ee_rel[:, :, 2], axis=0)))
        out["ee_rel_x_span_mm_by_leg"] = [
            float(v) for v in np.ptp(ee_rel[:, :, 0], axis=0)
        ]
        out["ee_rel_y_span_mm_by_leg"] = [
            float(v) for v in np.ptp(ee_rel[:, :, 1], axis=0)
        ]
        out["ee_rel_z_span_mm_by_leg"] = [
            float(v) for v in np.ptp(ee_rel[:, :, 2], axis=0)
        ]
        out["mean_ee_rel_z_mm_by_leg"] = [
            float(v) for v in np.mean(ee_rel[:, :, 2], axis=0)
        ]
        out["endpoint_leg_labels"] = list(legs)
    if "adhesion_cmd" in logs:
        out["mean_adhesion_cmd_frac"] = float(np.mean(logs["adhesion_cmd"]))
        if "contact_flag" in logs:
            stance = logs["adhesion_cmd"] > 0
            swing = ~stance
            contact = logs["contact_flag"] > 0
            if np.any(stance):
                out["contact_during_adhesion_frac"] = float(np.mean(contact[stance]))
            if np.any(swing):
                out["contact_during_swing_frac"] = float(np.mean(contact[swing]))
            out["contact_during_adhesion_frac_by_leg"] = [
                float(np.mean(contact[stance[:, i], i])) if np.any(stance[:, i]) else None
                for i in range(len(legs))
            ]
            out["contact_during_swing_frac_by_leg"] = [
                float(np.mean(contact[swing[:, i], i])) if np.any(swing[:, i]) else None
                for i in range(len(legs))
            ]
        if "tarsus5_contact_force" in logs:
            tarsus5_contact = np.linalg.norm(logs["tarsus5_contact_force"], axis=2) > 1e-9
            out["tarsus5_contact_frac"] = float(np.mean(tarsus5_contact))
            out["tarsus5_contact_frac_by_leg"] = [
                float(np.mean(tarsus5_contact[:, i])) for i in range(len(legs))
            ]
            tarsus5_force_norm = np.linalg.norm(logs["tarsus5_contact_force"], axis=2)
            out["mean_tarsus5_contact_force_norm_by_leg"] = [
                float(np.mean(tarsus5_force_norm[:, i])) for i in range(len(legs))
            ]
            stance = logs["adhesion_cmd"] > 0
            swing = ~stance
            if np.any(stance):
                out["tarsus5_contact_during_adhesion_frac"] = float(
                    np.mean(tarsus5_contact[stance])
                )
            if np.any(swing):
                out["tarsus5_contact_during_swing_frac"] = float(
                    np.mean(tarsus5_contact[swing])
                )
            out["tarsus5_contact_during_adhesion_frac_by_leg"] = [
                float(np.mean(tarsus5_contact[stance[:, i], i]))
                if np.any(stance[:, i])
                else None
                for i in range(len(legs))
            ]
            out["tarsus5_contact_during_swing_frac_by_leg"] = [
                float(np.mean(tarsus5_contact[swing[:, i], i]))
                if np.any(swing[:, i])
                else None
                for i in range(len(legs))
            ]
    return out


def _v1_like_contact_params():
    """Contact parameters matching FlyGym v1 ``Fly`` defaults."""
    from flygym.compose.physics import ContactParams

    return ContactParams(
        sliding_friction=1.0,
        torsional_friction=0.005,
        rolling_friction=0.0001,
        solver_refaccl_timeconst=2e-4,
        solver_refaccl_dampratio=1e3,
        solver_impedance_min=0.999,
        solver_impedance_max=0.9999,
        solver_impedance_min2max_width=0.001,
        solver_impedance_transitionmidpoint=0.5,
        solver_impedance_transitionsharpness=2.0,
    )


def _contact_meta_lines(contact_params) -> list[str]:
    return [
        f"v2_contact_friction={contact_params.get_friction_tuple()}",
        f"v2_contact_solref={contact_params.get_solref_tuple()}",
        f"v2_contact_solimp={contact_params.get_solimp_tuple()}",
    ]


def _v2_contact_bodies(value: str, contact_bodies_preset_cls):
    if value == "none":
        return []
    return contact_bodies_preset_cls(value)


def _json_ready(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _flatten_metrics(prefix: str, value, out: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_metrics(f"{prefix}{key}.", item, out)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _flatten_metrics(f"{prefix}{idx}.", item, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix[:-1]] = value


def _write_summary_files(path_json: Path, path_csv: Path, summary: dict) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    flat: dict[str, object] = {}
    _flatten_metrics("", _json_ready(summary), flat)
    with path_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat))
        writer.writeheader()
        writer.writerow(flat)


def _parity_summary(stage: str, openloop: dict, v1_stats: dict | None, v2_stats: dict | None) -> dict:
    thresholds = {
        "phase_abs_max": 1e-12,
        "magnitude_abs_max": 1e-12,
        "dof_reorder_abs_max": 1e-12,
        "thorax_delta_x_rel_error_max": 0.20,
        "endpoint_span_rel_error_max": 0.20,
        "swing_tarsus5_contact_frac_max": 0.25,
    }
    checks = {
        "openloop_phase": openloop["d_phase"] <= thresholds["phase_abs_max"],
        "openloop_magnitude": openloop["d_mag"] <= thresholds["magnitude_abs_max"],
        "v2_dof_reorder": openloop["reorder_err"] <= thresholds["dof_reorder_abs_max"],
    }
    parity: dict[str, float | None] = {
        "thorax_delta_x_rel_error": None,
        "endpoint_x_span_rel_error": None,
        "endpoint_y_span_rel_error": None,
        "endpoint_z_span_rel_error": None,
        "v2_tarsus5_contact_during_swing_frac": None,
    }
    if v1_stats is not None and v2_stats is not None and stage in {"contact", "full"}:
        v1_dx = abs(float(v1_stats.get("thorax_delta_x_mm", 0.0)))
        v2_dx = abs(float(v2_stats.get("thorax_delta_x_mm", 0.0)))
        if v1_dx > 1e-12:
            err = abs(v2_dx - v1_dx) / v1_dx
            parity["thorax_delta_x_rel_error"] = err
            checks["thorax_delta_x_within_20pct"] = (
                err <= thresholds["thorax_delta_x_rel_error_max"]
            )
        for axis in ("x", "y", "z"):
            key = f"mean_ee_rel_{axis}_span_mm"
            v1_span = abs(float(v1_stats.get(key, 0.0)))
            v2_span = abs(float(v2_stats.get(key, 0.0)))
            if v1_span > 1e-12:
                err = abs(v2_span - v1_span) / v1_span
                parity[f"endpoint_{axis}_span_rel_error"] = err
                checks[f"endpoint_{axis}_span_within_20pct"] = (
                    err <= thresholds["endpoint_span_rel_error_max"]
                )
        swing_contact = v2_stats.get("tarsus5_contact_during_swing_frac")
        if swing_contact is not None:
            swing_contact = float(swing_contact)
            parity["v2_tarsus5_contact_during_swing_frac"] = swing_contact
            checks["swing_tarsus5_contact_below_25pct"] = (
                swing_contact <= thresholds["swing_tarsus5_contact_frac_max"]
            )
    elif v1_stats is not None and v2_stats is not None and stage == "fk":
        for axis in ("x", "y", "z"):
            key = f"mean_ee_rel_{axis}_span_mm"
            v1_span = abs(float(v1_stats.get(key, 0.0)))
            v2_span = abs(float(v2_stats.get(key, 0.0)))
            if v1_span > 1e-12:
                err = abs(v2_span - v1_span) / v1_span
                parity[f"endpoint_{axis}_span_rel_error"] = err
                checks[f"fk_endpoint_{axis}_span_within_20pct"] = (
                    err <= thresholds["endpoint_span_rel_error_max"]
                )
    elif v1_stats is not None and v2_stats is not None and stage == "tracking":
        v1_rmse = float(v1_stats.get("mean_rmse_cmd_vs_pos_rad", 0.0))
        v2_rmse = float(v2_stats.get("mean_rmse_cmd_vs_pos_rad", 0.0))
        parity["tracking_v1_mean_rmse_cmd_vs_pos_rad"] = v1_rmse
        parity["tracking_v2_mean_rmse_cmd_vs_pos_rad"] = v2_rmse
        checks["tracking_states_finite"] = all(
            math.isfinite(value)
            for value in (
                v1_rmse,
                v2_rmse,
                float(v1_stats.get("thorax_delta_x_mm", 0.0)),
                float(v2_stats.get("thorax_delta_x_mm", 0.0)),
            )
        )
    return {
        "stage": stage,
        "thresholds": thresholds,
        "checks": checks,
        "passed": bool(checks and all(checks.values())),
        "parity": parity,
    }


def _write_stage_report(path: Path, summary: dict) -> None:
    lines = [
        f"CPG v1 vs v2 staged comparison: {summary['stage']}",
        "=" * 72,
        "",
        "Pass/fail:",
    ]
    checks = summary.get("status", {}).get("checks", {})
    if checks:
        for name, passed in checks.items():
            lines.append(f"  {'PASS' if passed else 'FAIL'} {name}")
    else:
        lines.append("  no checks recorded")
    lines += ["", "Open-loop:"]
    for key, value in summary.get("openloop", {}).items():
        if isinstance(value, (int, float, str, bool)):
            lines.append(f"  {key}: {value}")
    for label in ("v1", "v2"):
        stats = summary.get(label)
        if not stats:
            continue
        lines += ["", label.upper() + ":"]
        for key, value in stats.items():
            if isinstance(value, (int, float, str, bool)):
                lines.append(f"  {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_kinematics_report(
    path: Path,
    *,
    openloop: dict,
    v1_sim: dict,
    v2_sim: dict,
    meta_lines: list[str],
) -> None:
    lines = [
        "CPG v1 vs v2 — open-loop and MuJoCo rollout comparison",
        "=" * 72,
        "",
        "Configuration:",
        *[f"  {m}" for m in meta_lines],
        "",
        "Open-loop (same CPG params; PreprogrammedSteps differ on right-leg roll/yaw):",
        f"  phase max abs diff: {openloop['d_phase']}",
        f"  magnitude max abs diff: {openloop['d_mag']}",
        f"  RMSE(joint v2 concat - v1 concat) all legs [rad]: {openloop['rmse_all']:.6f}",
        f"  RMSE left legs only [rad]: {openloop['rmse_left']:.6f}",
        f"  RMSE right legs only [rad]: {openloop['rmse_right']:.6f}",
        f"  mean L2 finite-diff joint speed (cmd) v1 / v2 [rad/s]: "
        f"{openloop['mean_speed_v1']:.4f} / {openloop['mean_speed_v2']:.4f}",
        "",
        "MuJoCo v1 (gymnasium NeuroMechFly, default position gains):",
        f"  mean RMSE(cmd vs sensed pos) per step [rad]: {v1_sim['mean_rmse_cmd_vs_pos_rad']:.6f}",
        f"  max  RMSE(cmd vs sensed pos) per step [rad]: {v1_sim['max_rmse_cmd_vs_pos_rad']:.6f}",
        f"  mean L2 joint velocity [rad/s]: {v1_sim['mean_L2_joint_vel']:.4f}",
        f"  thorax Δx, Δy, Δz over rollout [mm]: "
        f"{v1_sim['thorax_delta_x_mm']:.4f}, {v1_sim['thorax_delta_y_mm']:.4f}, {v1_sim['thorax_delta_z_mm']:.4f}",
    ]
    if "mean_contact_force_scalar" in v1_sim:
        lines.append(
            f"  mean aggregate contact-force scalar (v1 obs) [arb.]: {v1_sim['mean_contact_force_scalar']:.4f}"
        )
    if "mean_ee_height_mm" in v1_sim:
        lines.append(f"  mean tarsi height (EE z mean) [mm]: {v1_sim['mean_ee_height_mm']:.4f}")
    if "mean_ee_rel_z_mm" in v1_sim:
        lines += [
            f"  mean endpoint z relative to thorax [mm]: {v1_sim['mean_ee_rel_z_mm']:.4f}",
            f"  mean endpoint relative x/y/z span [mm]: "
            f"{v1_sim['mean_ee_rel_x_span_mm']:.4f}, "
            f"{v1_sim['mean_ee_rel_y_span_mm']:.4f}, "
            f"{v1_sim['mean_ee_rel_z_span_mm']:.4f}",
        ]
    lines += [
        "",
        "MuJoCo v2 (FlyGym 2, --actuator-gain from CLI):",
        f"  mean RMSE(cmd vs qpos) per step [rad]: {v2_sim['mean_rmse_cmd_vs_pos_rad']:.6f}",
        f"  max  RMSE(cmd vs qpos) per step [rad]: {v2_sim['max_rmse_cmd_vs_pos_rad']:.6f}",
        f"  mean L2 joint velocity [rad/s]: {v2_sim['mean_L2_joint_vel']:.4f}",
        f"  thorax Δx, Δy, Δz over rollout [mm]: "
        f"{v2_sim['thorax_delta_x_mm']:.4f}, {v2_sim['thorax_delta_y_mm']:.4f}, {v2_sim['thorax_delta_z_mm']:.4f}",
    ]
    if "mean_contact_frac" in v2_sim:
        lines.append(
            f"  mean legs in ground contact (sensor flag, 0–1 per leg): {v2_sim['mean_contact_frac']:.4f}"
        )
    if "mean_contact_force_scalar" in v2_sim:
        lines.append(
            f"  mean per-leg |F_contact| (L2 over xyz) averaged over legs & time [arb.]: "
            f"{v2_sim['mean_contact_force_scalar']:.4f}"
        )
    if "mean_ee_rel_z_mm" in v2_sim:
        lines += [
            f"  mean endpoint z relative to thorax [mm]: {v2_sim['mean_ee_rel_z_mm']:.4f}",
            f"  mean endpoint relative x/y/z span [mm]: "
            f"{v2_sim['mean_ee_rel_x_span_mm']:.4f}, "
            f"{v2_sim['mean_ee_rel_y_span_mm']:.4f}, "
            f"{v2_sim['mean_ee_rel_z_span_mm']:.4f}",
        ]
    if "contact_during_adhesion_frac" in v2_sim:
        lines += [
            f"  v2 contact fraction while adhesion commanded on/off: "
            f"{v2_sim['contact_during_adhesion_frac']:.4f}, "
            f"{v2_sim.get('contact_during_swing_frac', float('nan')):.4f}",
        ]
    if "tarsus5_contact_during_adhesion_frac" in v2_sim:
        lines += [
            f"  v2 tarsus5 contact fraction while adhesion commanded on/off: "
            f"{v2_sim['tarsus5_contact_during_adhesion_frac']:.4f}, "
            f"{v2_sim.get('tarsus5_contact_during_swing_frac', float('nan')):.4f}",
        ]
    lines += [
        "Notes:",
        "  • v1 and v2 use different fly models and units/layouts; compare trends, not raw joint equality.",
        "  • Open-loop RMSE(right) mostly reflects v2 anatomical sign flips in the pickle splines.",
        "  • Sim tracking RMSE shows how well position actuators follow CPG commands in each stack.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _resize_frame_hw(frame, height: int, width: int):
    import numpy as np
    from PIL import Image

    arr = np.asarray(frame)
    return np.asarray(
        Image.fromarray(arr).resize((width, height), Image.Resampling.LANCZOS)
    )


def _side_by_side_video(
    frames_left: list,
    frames_right: list,
    out_path: Path,
    *,
    label_left: str,
    label_right: str,
    fps: float,
) -> None:
    import imageio.v3 as iio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    n = min(len(frames_left), len(frames_right))
    if n == 0:
        raise RuntimeError("No frames to write for side-by-side video.")

    ref = np.asarray(frames_right[0])
    h_tgt, w_tgt = ref.shape[:2]

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    stacked: list[np.ndarray] = []
    for i in range(n):
        a = np.asarray(frames_left[i])
        b = np.asarray(frames_right[i])
        if a.shape[:2] != (h_tgt, w_tgt):
            a = _resize_frame_hw(a, h_tgt, w_tgt)
        if b.shape[:2] != (h_tgt, w_tgt):
            b = _resize_frame_hw(b, h_tgt, w_tgt)
        comb = np.concatenate([a, b], axis=1)
        img = Image.fromarray(comb)
        draw = ImageDraw.Draw(img)
        _, w = a.shape[:2]

        def _shadow_text(x: int, y: int, text: str) -> None:
            for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
            draw.text((x, y), text, fill=(255, 255, 255), font=font)

        _shadow_text(6, 4, label_left)
        _shadow_text(w + 6, 4, label_right)
        stacked.append(np.asarray(img))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        out_path,
        np.stack(stacked, axis=0),
        fps=fps,
        codec="libx264",
        quality=8,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pickle",
        type=Path,
        default=None,
        help="Path to single_steps_untethered.pkl (default: flygym v2 assets path)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1e-4,
        help="CPG integration step when --no-video (ignored when recording video; sim timestep is used)",
    )
    parser.add_argument("--duration", type=float, default=1.0, help="Simulated seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--intrinsic-frequency",
        type=float,
        default=12.0,
        help="CPG intrinsic frequency (Hz) for both networks",
    )
    parser.add_argument(
        "--coupling-strength",
        type=float,
        default=10.0,
        help="Tripod coupling strength (both networks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for NPZ and MP4 (default: <repo>/debug_outputs)",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="cpg_v1_v2",
        help="Output file stem: <stem>.npz, <stem>_side_by_side.mp4, <stem>_kinematics_compare.txt, <stem>_summary.json/csv",
    )
    parser.add_argument(
        "--stage",
        choices=["openloop", "fk", "tracking", "contact", "full"],
        default=None,
        help="Staged debugger mode. Default is ``full`` unless --no-video is supplied, "
        "in which case it is ``openloop`` for backward compatibility.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip MuJoCo; use --timestep for CPG only (faster, smaller NPZ)",
    )
    parser.add_argument(
        "--no-mp4",
        action="store_true",
        help="When MuJoCo runs: skip encoding the side-by-side MP4 (still writes NPZ, "
        "kinematics report, and full sim logs—useful for batch sweeps)",
    )
    parser.add_argument(
        "--actuator-gain",
        type=float,
        default=45.0,
        help="v2 position actuator kp (FlyGym v1 ``Fly`` default is 45; increase e.g. "
        "80–150 if you need more open-loop translation at the cost of harsher contact)",
    )
    parser.add_argument(
        "--adhesion-gain",
        type=float,
        default=40.0,
        help="v2 leg adhesion gain (v1 reference stays fixed at adhesion_force=40)",
    )
    parser.add_argument(
        "--spawn-z-mm",
        type=float,
        default=0.5,
        help="World spawn height in mm (v1 ``Fly`` default spawn z is 0.5)",
    )
    parser.add_argument(
        "--warmup-duration",
        type=float,
        default=0.05,
        help="v2 ``Simulation.warmup`` duration; v1 rollout runs the same duration of "
        "neutral-pose physics steps before the CPG loop",
    )
    parser.add_argument(
        "--contact-fixed-duration",
        type=float,
        default=0.05,
        help="Seconds of fixed default-pose contact probing to save in --stage contact",
    )
    parser.add_argument(
        "--render-playback-speed",
        type=float,
        default=0.2,
        help="Renderer playback_speed for v2 (v1 ``YawOnlyCamera`` default is 0.2)",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=25,
        help="Output video fps for v2 (v1 compare camera uses 25)",
    )
    parser.add_argument(
        "--v1-like-contact-params",
        action="store_true",
        help="Use FlyGym v1 ``Fly`` contact friction / solref / solimp for v2 ground pairs "
        "instead of FlyGym 2 ContactParams defaults.",
    )
    parser.add_argument(
        "--v2-contact-bodies",
        type=str,
        default="legs_only",
        choices=["none", "all", "legs_thorax_abdomen_head", "legs_only", "tibia_tarsus_only"],
        help="Body segments that collide with the ground in the v2 rollout. "
        "Default ``legs_only`` matches v1 Fly floor_collisions='legs'. ``none`` is "
        "used by tracking mode to isolate actuator behavior.",
    )
    args = parser.parse_args()
    if args.stage is None:
        args.stage = "openloop" if args.no_video else "full"
    if args.no_video and args.stage != "openloop":
        raise SystemExit("--no-video is only compatible with --stage openloop")

    repo = _repo_root()
    ref_root = repo / ".flygym-1.0-ref"
    if not ref_root.is_dir():
        raise SystemExit(f"Missing v1 reference tree: {ref_root}")

    out_dir = args.output_dir or (repo / "debug_outputs")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{args.stem}.npz"
    mp4_path = out_dir / f"{args.stem}_side_by_side.mp4"
    report_path = out_dir / f"{args.stem}_kinematics_compare.txt"
    summary_json_path = out_dir / f"{args.stem}_summary.json"
    summary_csv_path = out_dir / f"{args.stem}_summary.csv"

    sys.path.insert(0, str(repo / "src"))
    import numpy as np
    from flygym import Simulation, assets_dir
    from flygym.anatomy import ContactBodiesPreset
    from flygym.compose import FlatGroundWorld
    from flygym.compose.physics import ContactParams
    from flygym.examples.locomotion.common import get_default_locomotion_dof_order
    from flygym.examples.locomotion.common import make_locomotion_fly
    from flygym.examples.locomotion.cpg_controller import CPGNetwork as CPGNetworkV2
    from flygym.examples.locomotion.cpg_controller import get_cpg_biases
    from flygym.examples.locomotion.preprogrammed import PreprogrammedSteps as PreV2
    from flygym.utils.math import Rotation3D

    pickle_path = args.pickle or (assets_dir / "behavior/single_steps_untethered.pkl")
    if not pickle_path.is_file():
        raise SystemExit(f"Pickle not found: {pickle_path}")

    steps_mod, _cpg_mod = _load_v1_locomotion_modules(ref_root)
    PreV1 = steps_mod.PreprogrammedSteps
    CPGNetworkV1 = sys.modules["flygym_gymnasium.examples.locomotion"].CPGNetwork

    pre_v1 = PreV1(path=str(pickle_path))
    pre_v2 = PreV2(path=str(pickle_path))

    dof_order = list(get_default_locomotion_dof_order())
    dt = float(args.timestep)
    fly = None
    sim = None
    camera_key: str | None = None
    output_fps = float(args.render_fps)
    effective_v2_contact_bodies = "none" if args.stage == "tracking" else args.v2_contact_bodies

    if args.stage != "openloop":
        contact_params = (
            _v1_like_contact_params()
            if args.v1_like_contact_params
            else ContactParams()
        )
        fly = make_locomotion_fly(
            name="cpg_compare",
            actuator_gain=args.actuator_gain,
            actuator_forcerange=(-65.0, 65.0),
            add_adhesion=True,
            adhesion_gain=args.adhesion_gain,
            colorize=True,
        )
        dof_order_fly = list(fly.get_actuated_jointdofs_order("position"))
        if dof_order_fly != dof_order:
            raise SystemExit(
                "Fly actuator DOF order differs from get_default_locomotion_dof_order(); "
                f"update script to use fly order ({len(dof_order_fly)} dofs)."
            )
        # Approximate FlyGym v1 ``camera_right`` (config.yaml): pos [0,-8,1], euler [1.57,0,0]
        body_cam = fly.add_tracking_camera(
            name="body_cam",
            pos_offset=(0.0, -8.0, 1.0),
            rotation=Rotation3D("euler", (1.57, 0.0, 0.0)),
            fovy=30.0,
        )
        world = FlatGroundWorld()
        spawn_pos = [0.0, 0.0, float(args.spawn_z_mm)]
        spawn_rot = Rotation3D("quat", [1.0, 0.0, 0.0, 0.0])
        world.add_fly(
            fly,
            spawn_pos,
            spawn_rot,
            bodysegs_with_ground_contact=_v2_contact_bodies(
                effective_v2_contact_bodies, ContactBodiesPreset
            ),
            ground_contact_params=contact_params,
        )
        sim = Simulation(world)
        _ = sim.set_renderer(
            body_cam,
            camera_res=(240, 320),
            playback_speed=float(args.render_playback_speed),
            output_fps=int(args.render_fps),
        )
        camera_key = body_cam.full_identifier
        dt = float(sim.timestep)
        if abs(dt - args.timestep) > 1e-12:
            print(
                f"Note: using sim timestep dt={dt:g} for CPG (matches video); "
                f"--timestep {args.timestep:g} ignored unless --no-video."
            )

    n_steps = int(args.duration / dt)
    if n_steps < 2:
        raise SystemExit("duration too short for finite-difference speeds")

    phase_biases = get_cpg_biases("tripod")
    nu = np.ones(6) * args.intrinsic_frequency
    R = np.ones(6)
    w = (phase_biases > 0) * args.coupling_strength
    alpha = np.ones(6) * 20.0

    cpg1 = CPGNetworkV1(
        dt,
        intrinsic_freqs=nu,
        intrinsic_amps=R,
        coupling_weights=w,
        phase_biases=phase_biases,
        convergence_coefs=alpha,
        seed=args.seed,
    )
    cpg2 = CPGNetworkV2(
        dt,
        intrinsic_freqs=nu,
        intrinsic_amps=R,
        coupling_weights=w,
        phase_biases=phase_biases,
        convergence_coefs=alpha,
        seed=args.seed,
    )

    perm, legs_v2 = _concat_perm_to_doforder(pre_v2, dof_order)

    phases_v1 = np.zeros((n_steps, 6))
    phases_v2 = np.zeros((n_steps, 6))
    mags_v1 = np.zeros((n_steps, 6))
    mags_v2 = np.zeros((n_steps, 6))
    ja_v1 = np.zeros((n_steps, 42))
    ja_v2_concat = np.zeros((n_steps, 42))
    ja_v2_dof = np.zeros((n_steps, 42))
    adhesion = np.zeros((n_steps, 6), dtype=bool)

    for t in range(n_steps):
        cpg1.step()
        cpg2.step()
        phases_v1[t] = cpg1.curr_phases
        phases_v2[t] = cpg2.curr_phases
        mags_v1[t] = cpg1.curr_magnitudes
        mags_v2[t] = cpg2.curr_magnitudes

        ja_v1[t] = np.concatenate(
            [
                pre_v1.get_joint_angles(
                    leg.upper(),
                    cpg1.curr_phases[i],
                    cpg1.curr_magnitudes[i],
                ).ravel()
                for i, leg in enumerate(legs_v2)
            ]
        )
        ja_v2_concat[t] = np.concatenate(
            [
                pre_v2.get_joint_angles(
                    leg, cpg2.curr_phases[i], cpg2.curr_magnitudes[i]
                ).ravel()
                for i, leg in enumerate(legs_v2)
            ]
        )
        ja_v2_dof[t] = pre_v2.get_joint_angles_by_dof_order(
            cpg2.curr_phases,
            cpg2.curr_magnitudes,
            dof_order,
        )
        adhesion[t] = pre_v2.get_adhesion_onoff_by_phase(cpg1.curr_phases)

    d_phase = np.abs(phases_v1 - phases_v2).max()
    d_mag = np.abs(mags_v1 - mags_v2).max()

    reorder_err = float(np.max(np.abs(ja_v2_concat[:, perm] - ja_v2_dof)))

    ja_diff = ja_v2_concat - ja_v1
    rmse_all = float(np.sqrt(np.mean(ja_diff**2)))
    rmse_left = float(np.sqrt(np.mean((ja_diff[:, :21]) ** 2)))
    rmse_right = float(np.sqrt(np.mean((ja_diff[:, 21:]) ** 2)))
    speed_v1 = np.diff(ja_v1, axis=0) / dt
    speed_v2 = np.diff(ja_v2_concat, axis=0) / dt
    speed_l2_v1 = np.linalg.norm(speed_v1, axis=1)
    speed_l2_v2 = np.linalg.norm(speed_v2, axis=1)

    print("CPG phase max abs diff (v1 vs v2):", d_phase)
    print("CPG magnitude max abs diff (v1 vs v2):", d_mag)
    print("v2 concat permuted to actuator order vs v2 dof-order max abs:", reorder_err)
    print("RMSE(ja_v2_concat - ja_v1) all legs [rad]:", round(rmse_all, 6))
    print("RMSE left legs only (lf,lm,lh) [rad]:", round(rmse_left, 6))
    print("RMSE right legs only (rf,rm,rh) [rad]:", round(rmse_right, 6))
    print(
        "mean L2 joint speed (rad/s): v1",
        float(speed_l2_v1.mean()),
        "v2",
        float(speed_l2_v2.mean()),
    )

    time_grid = np.arange(n_steps) * dt
    openloop = {
        "d_phase": float(d_phase),
        "d_mag": float(d_mag),
        "reorder_err": reorder_err,
        "rmse_all": rmse_all,
        "rmse_left": rmse_left,
        "rmse_right": rmse_right,
        "mean_speed_v1": float(speed_l2_v1.mean()),
        "mean_speed_v2": float(speed_l2_v2.mean()),
    }

    base_npz: dict = dict(
        meta=np.array(
            [
                str(pickle_path),
                dt,
                args.duration,
                args.seed,
                args.intrinsic_frequency,
                args.coupling_strength,
                str(npz_path),
                str(mp4_path)
                if (args.stage not in {"openloop", "fk"} and not args.no_mp4)
                else "",
                str(report_path),
                args.actuator_gain,
                args.adhesion_gain,
                args.spawn_z_mm,
                args.warmup_duration,
                args.render_playback_speed,
                args.render_fps,
                bool(args.no_mp4),
                args.stage,
                effective_v2_contact_bodies,
            ],
            dtype=object,
        ),
        time=time_grid,
        phases_v1=phases_v1,
        phases_v2=phases_v2,
        magnitudes_v1=mags_v1,
        magnitudes_v2=mags_v2,
        joint_concat_v1=ja_v1,
        joint_concat_v2=ja_v2_concat,
        joint_doforder_v1_legacy=ja_v1[:, perm],
        joint_doforder_v2=ja_v2_dof,
        concat_perm_to_doforder=perm,
        adhesion=adhesion,
        joint_speed_l2_v1=speed_l2_v1,
        joint_speed_l2_v2=speed_l2_v2,
        legs=np.array(legs_v2, dtype=object),
    )

    def _summary_config() -> dict:
        return {
            "pickle": str(pickle_path),
            "dt": dt,
            "duration": args.duration,
            "n_steps": n_steps,
            "seed": args.seed,
            "intrinsic_frequency_hz": args.intrinsic_frequency,
            "coupling_strength": args.coupling_strength,
            "v2_actuator_gain": args.actuator_gain,
            "v2_adhesion_gain": args.adhesion_gain,
            "v1_actuator_gain": 45.0,
            "v1_adhesion_force": 40.0,
            "v1_floor_collisions": "none" if args.stage == "tracking" else "legs",
            "spawn_z_mm": args.spawn_z_mm,
            "warmup_s": args.warmup_duration,
            "contact_fixed_duration_s": args.contact_fixed_duration,
            "v2_contact_bodies": effective_v2_contact_bodies,
            "v1_like_contact_params": bool(args.v1_like_contact_params),
        }

    if args.stage == "openloop":
        status = _parity_summary(args.stage, openloop, None, None)
        summary = {
            "stage": args.stage,
            "config": _summary_config(),
            "openloop": openloop,
            "status": status,
            "artifacts": {
                "npz": str(npz_path),
                "report": str(report_path),
                "summary_json": str(summary_json_path),
                "summary_csv": str(summary_csv_path),
            },
        }
        np.savez(npz_path, **base_npz)
        _write_summary_files(summary_json_path, summary_csv_path, summary)
        _write_stage_report(report_path, summary)
        print("Wrote", npz_path.resolve())
        print("Wrote", report_path.resolve())
        print("Wrote", summary_json_path.resolve())
        print("Wrote", summary_csv_path.resolve())
    elif args.stage == "fk":
        assert sim is not None and fly is not None
        logs_v2 = _run_v2_fk_with_logs(
            sim,
            fly,
            joint_dof=ja_v2_dof,
            dof_order=dof_order,
        )
        logs_v1 = _run_v1_fk_with_logs(
            ref_root=ref_root,
            pickle_path=pickle_path,
            joint_cmd=ja_v1,
            spawn_z_mm=float(args.spawn_z_mm),
        )
        s1 = _sim_tracking_stats(logs_v1, dt=dt)
        s2 = _sim_tracking_stats(logs_v2, dt=dt)
        status = _parity_summary(args.stage, openloop, s1, s2)
        summary = {
            "stage": args.stage,
            "config": _summary_config(),
            "openloop": openloop,
            "v1": s1,
            "v2": s2,
            "status": status,
            "artifacts": {
                "npz": str(npz_path),
                "report": str(report_path),
                "summary_json": str(summary_json_path),
                "summary_csv": str(summary_csv_path),
            },
        }
        np.savez(
            npz_path,
            **base_npz,
            fk_v1_joint_cmd=logs_v1["joint_cmd"],
            fk_v1_joint_pos=logs_v1["joint_pos"],
            fk_v1_thorax_pos=logs_v1["thorax_pos"],
            fk_v1_thorax_rot_euler=logs_v1["thorax_rot_euler"],
            fk_v1_ee_pos=logs_v1["ee_pos"],
            fk_v1_ee_rel_pos=_endpoint_relative_to_thorax(logs_v1),
            fk_v1_thorax_vel=_thorax_velocity(logs_v1, dt),
            fk_v2_joint_cmd=logs_v2["joint_cmd"],
            fk_v2_joint_pos=logs_v2["joint_pos"],
            fk_v2_thorax_pos=logs_v2["thorax_pos"],
            fk_v2_thorax_quat=logs_v2["thorax_quat"],
            fk_v2_thorax_euler=_thorax_euler_xyz(logs_v2),
            fk_v2_thorax_vel=_thorax_velocity(logs_v2, dt),
            fk_v2_ee_pos=logs_v2["ee_pos"],
            fk_v2_ee_rel_pos=_endpoint_relative_to_thorax(logs_v2),
        )
        _write_summary_files(summary_json_path, summary_csv_path, summary)
        _write_stage_report(report_path, summary)
        print("Wrote", npz_path.resolve())
        print("Wrote", report_path.resolve())
        print("Wrote", summary_json_path.resolve())
        print("Wrote", summary_csv_path.resolve())
    else:
        assert sim is not None and fly is not None and camera_key is not None
        print(
            "MuJoCo compare settings: v2 kp / adhesion / spawn_z / warmup "
            f"({args.actuator_gain}, {args.adhesion_gain}, {args.spawn_z_mm} mm, "
            f"{args.warmup_duration} s); v2 camera uses v1 camera_right-style offset; "
            f"v2 contact params are {'v1-like' if args.v1_like_contact_params else 'FlyGym 2 defaults'}."
        )
        default_pose = pre_v2.default_pose_by_dof_order(dof_order)
        fixed_v2_logs = None
        fixed_v2_stats = None
        if args.stage == "contact" and args.contact_fixed_duration > 0:
            fixed_n_steps = max(1, int(round(float(args.contact_fixed_duration) / dt)))
            print(
                "Running v2 fixed-pose contact probe for",
                fixed_n_steps,
                "steps.",
            )
            fixed_v2_logs = _simulate_v2_fixed_pose_contact_logs(
                sim,
                fly,
                joint_pose=default_pose,
                dof_order=dof_order,
                n_steps=fixed_n_steps,
                warmup_duration=args.warmup_duration,
            )
            fixed_v2_stats = _sim_tracking_stats(fixed_v2_logs, dt=dt)
        frames_v2, logs_v2 = _simulate_rollout_with_logs(
            sim,
            fly,
            default_pose=default_pose,
            joint_dof=ja_v2_dof,
            adhesion=adhesion,
            camera_key=camera_key,
            dof_order=dof_order,
            warmup_duration=args.warmup_duration,
        )
        frames_v1, logs_v1 = _run_v1_gymnasium_cpg_rollout_frames(
            ref_root=ref_root,
            pickle_path=pickle_path,
            dt=dt,
            n_steps=n_steps,
            seed=args.seed,
            intrinsic_frequency=args.intrinsic_frequency,
            coupling_strength=args.coupling_strength,
            phase_biases=phase_biases,
            spawn_z_mm=float(args.spawn_z_mm),
            adhesion_force=40.0,
            warmup_duration=float(args.warmup_duration),
            cam_play_speed=float(args.render_playback_speed),
            cam_fps=int(args.render_fps),
            floor_collisions="none" if args.stage == "tracking" else "legs",
        )

        s1 = _sim_tracking_stats(logs_v1, dt=dt)
        s2 = _sim_tracking_stats(logs_v2, dt=dt)

        print("--- MuJoCo rollout kinematics (isolated v1 vs v2 stacks) ---")
        print(
            "v1 mean / max RMSE(cmd vs sensed joint pos) [rad]:",
            round(s1["mean_rmse_cmd_vs_pos_rad"], 6),
            "/",
            round(s1["max_rmse_cmd_vs_pos_rad"], 6),
        )
        print(
            "v2 mean / max RMSE(cmd vs qpos) [rad]:",
            round(s2["mean_rmse_cmd_vs_pos_rad"], 6),
            "/",
            round(s2["max_rmse_cmd_vs_pos_rad"], 6),
        )
        print(
            "v1 thorax Δx, Δy, Δz [mm]:",
            round(s1["thorax_delta_x_mm"], 4),
            round(s1["thorax_delta_y_mm"], 4),
            round(s1["thorax_delta_z_mm"], 4),
        )
        print(
            "v2 thorax Δx, Δy, Δz [mm]:",
            round(s2["thorax_delta_x_mm"], 4),
            round(s2["thorax_delta_y_mm"], 4),
            round(s2["thorax_delta_z_mm"], 4),
        )
        print(
            "mean L2 joint velocity [rad/s] v1 / v2:",
            round(s1["mean_L2_joint_vel"], 4),
            "/",
            round(s2["mean_L2_joint_vel"], 4),
        )
        if "mean_contact_frac" in s2:
            print("v2 mean ground-contact flag (per leg, 0–1):", round(s2["mean_contact_frac"], 4))
        if fixed_v2_stats is not None:
            print(
                "v2 fixed-pose mean ground-contact flag (per leg, 0–1):",
                round(fixed_v2_stats["mean_contact_frac"], 4),
            )
        if "mean_ee_rel_x_span_mm" in s1 and "mean_ee_rel_x_span_mm" in s2:
            print(
                "mean endpoint relative x/y/z span [mm] v1:",
                round(s1["mean_ee_rel_x_span_mm"], 4),
                round(s1["mean_ee_rel_y_span_mm"], 4),
                round(s1["mean_ee_rel_z_span_mm"], 4),
                "v2:",
                round(s2["mean_ee_rel_x_span_mm"], 4),
                round(s2["mean_ee_rel_y_span_mm"], 4),
                round(s2["mean_ee_rel_z_span_mm"], 4),
            )

        meta_lines = [
            f"pickle={pickle_path}",
            f"dt={dt}, duration={args.duration}, n_steps={n_steps}",
            f"seed={args.seed}, intrinsic_frequency_hz={args.intrinsic_frequency}, "
            f"coupling_strength={args.coupling_strength}",
            f"v2_actuator_gain={args.actuator_gain}, v2_adhesion_gain={args.adhesion_gain}, "
            f"spawn_z_mm={args.spawn_z_mm}, warmup_s={args.warmup_duration}",
            f"render_playback_speed={args.render_playback_speed}, render_fps={args.render_fps}",
            f"v1 Fly: actuator_gain=45 (fixed), adhesion_force=40 (fixed), "
            f"floor_collisions={'none' if args.stage == 'tracking' else 'legs'}",
            f"v2_contact_bodies={effective_v2_contact_bodies}",
            *(
                _contact_meta_lines(contact_params)
                if "contact_params" in locals()
                else []
            ),
        ]
        _write_kinematics_report(
            report_path,
            openloop=openloop,
            v1_sim=s1,
            v2_sim=s2,
            meta_lines=meta_lines,
        )
        print("Wrote", report_path.resolve())

        sim_npz = {
            "sim_v1_joint_cmd": logs_v1["joint_cmd"],
            "sim_v1_joint_pos": logs_v1["joint_pos"],
            "sim_v1_joint_vel": logs_v1["joint_vel"],
            "sim_v1_joint_torque_obs": logs_v1["joint_torque"],
            "sim_v1_thorax_pos": logs_v1["thorax_pos"],
            "sim_v1_thorax_rot_euler": logs_v1["thorax_rot_euler"],
            "sim_v1_thorax_vel": _thorax_velocity(logs_v1, dt),
            "sim_v1_ee_pos": logs_v1["ee_pos"],
            "sim_v1_ee_rel_pos": _endpoint_relative_to_thorax(logs_v1),
            "sim_v1_adhesion_cmd": logs_v1["adhesion_cmd"],
            "sim_v1_contact_force_norm_mean": logs_v1["contact_force_norm_mean"],
            "sim_v1_ee_height_mean": logs_v1["ee_height_mean"],
            "sim_v2_joint_cmd": logs_v2["joint_cmd"],
            "sim_v2_joint_pos": logs_v2["joint_pos"],
            "sim_v2_joint_vel": logs_v2["joint_vel"],
            "sim_v2_thorax_pos": logs_v2["thorax_pos"],
            "sim_v2_thorax_quat": logs_v2["thorax_quat"],
            "sim_v2_thorax_euler": _thorax_euler_xyz(logs_v2),
            "sim_v2_thorax_vel": _thorax_velocity(logs_v2, dt),
            "sim_v2_ee_pos": logs_v2["ee_pos"],
            "sim_v2_ee_rel_pos": _endpoint_relative_to_thorax(logs_v2),
            "sim_v2_adhesion_cmd": logs_v2["adhesion_cmd"],
            "sim_v2_actuator_force": logs_v2["actuator_force"],
            "sim_v2_contact_flag": logs_v2["contact_flag"],
            "sim_v2_contact_force": logs_v2["contact_force"],
            "sim_v2_contact_force_norm": logs_v2["contact_force_norm"],
            "sim_v2_tarsus5_contact_force": logs_v2["tarsus5_contact_force"],
        }
        if fixed_v2_logs is not None:
            sim_npz.update(
                {
                    "contact_fixed_v2_joint_cmd": fixed_v2_logs["joint_cmd"],
                    "contact_fixed_v2_joint_pos": fixed_v2_logs["joint_pos"],
                    "contact_fixed_v2_joint_vel": fixed_v2_logs["joint_vel"],
                    "contact_fixed_v2_thorax_pos": fixed_v2_logs["thorax_pos"],
                    "contact_fixed_v2_thorax_quat": fixed_v2_logs["thorax_quat"],
                    "contact_fixed_v2_thorax_euler": _thorax_euler_xyz(fixed_v2_logs),
                    "contact_fixed_v2_thorax_vel": _thorax_velocity(fixed_v2_logs, dt),
                    "contact_fixed_v2_ee_pos": fixed_v2_logs["ee_pos"],
                    "contact_fixed_v2_ee_rel_pos": _endpoint_relative_to_thorax(
                        fixed_v2_logs
                    ),
                    "contact_fixed_v2_adhesion_cmd": fixed_v2_logs["adhesion_cmd"],
                    "contact_fixed_v2_actuator_force": fixed_v2_logs["actuator_force"],
                    "contact_fixed_v2_contact_flag": fixed_v2_logs["contact_flag"],
                    "contact_fixed_v2_contact_force": fixed_v2_logs["contact_force"],
                    "contact_fixed_v2_contact_force_norm": fixed_v2_logs[
                        "contact_force_norm"
                    ],
                    "contact_fixed_v2_tarsus5_contact_force": fixed_v2_logs[
                        "tarsus5_contact_force"
                    ],
                }
            )
        status = _parity_summary(args.stage, openloop, s1, s2)
        summary = {
            "stage": args.stage,
            "config": _summary_config(),
            "openloop": openloop,
            "v1": s1,
            "v2": s2,
            "v2_fixed_pose": fixed_v2_stats,
            "status": status,
            "artifacts": {
                "npz": str(npz_path),
                "report": str(report_path),
                "summary_json": str(summary_json_path),
                "summary_csv": str(summary_csv_path),
                "mp4": str(mp4_path)
                if args.stage not in {"openloop", "fk"} and not args.no_mp4
                else "",
            },
        }
        np.savez(npz_path, **base_npz, **sim_npz)
        print("Wrote", npz_path.resolve())
        _write_summary_files(summary_json_path, summary_csv_path, summary)
        print("Wrote", summary_json_path.resolve())
        print("Wrote", summary_csv_path.resolve())

        if len(frames_v1) != len(frames_v2):
            print(
                "warning: frame count mismatch v1",
                len(frames_v1),
                "v2",
                len(frames_v2),
                "(MP4 uses the shorter sequence)",
            )
        if args.stage not in {"openloop", "fk"} and not args.no_mp4:
            _side_by_side_video(
                frames_v1,
                frames_v2,
                mp4_path,
                label_left="v1 (gymnasium stack)",
                label_right="v2 (FlyGym 2)",
                fps=output_fps,
            )
            print("Wrote", mp4_path.resolve())
        else:
            print("Skipped side-by-side MP4 because --no-mp4 was supplied.")


if __name__ == "__main__":
    main()
