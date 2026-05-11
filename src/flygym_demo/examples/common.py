from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
from jaxtyping import Float

from flygym import Simulation
from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
from flygym.compose import ActuatorType, Fly, KinematicPosePreset
from flygym.utils.math import Rotation3D


@dataclass
class ControllerOutput:
    """Joint position targets and adhesion states for one simulation step."""

    position_targets: Float[np.ndarray, "n_position_actuators"]
    adhesion: Float[np.ndarray, "6"]
    metadata: dict[str, Any] = field(default_factory=dict)


class StepController(Protocol):
    def reset(self) -> None: ...

    def step(self, sim: Simulation, drive: Any | None = None) -> ControllerOutput: ...


def make_walking_fly(
    name: str = "nmf",
    *,
    axis_order: AxisOrder = AxisOrder.YAW_PITCH_ROLL,
    actuator_kp: float = 50.0,
    adhesion_gain: float = 20.0,
    add_camera: bool = True,
    colorize: bool = True,
) -> Fly:
    """Create the standard 42-DoF walking fly used by tutorial controllers.

    The helper intentionally mirrors the FlyGym v1 tutorial setup: the fly uses the
    leg-only skeleton with active-leg position actuators, v1-like adhesion gain, a
    tracking camera by default, and colored visual materials unless explicitly opted
    out with ``colorize=False``.
    """

    neutral_pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(axis_order)
    skeleton = Skeleton(axis_order=axis_order, joint_preset=JointPreset.LEGS_ONLY)
    fly = Fly(name=name)
    fly.add_joints(skeleton, neutral_pose=neutral_pose)
    if colorize:
        fly.colorize()
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs,
        ActuatorType.POSITION,
        neutral_input=neutral_pose,
        kp=actuator_kp,
    )
    fly.add_leg_adhesion(gain=adhesion_gain)
    if add_camera:
        fly.add_tracking_camera(name="trackcam")
    return fly


def get_neutral_position_targets(
    fly: Fly,
    actuator_type: ActuatorType = ActuatorType.POSITION,
) -> Float[np.ndarray, "n_position_actuators"]:
    """Return neutral actuator inputs in actuator order.

    ``Simulation.get_joint_angles()`` returns every joint DoF in skeleton order (66
    values for the leg-only walking model), which is not a valid control vector for
    the 42 active position actuators. This helper instead reads each actuator's
    neutral input from the fly model in ``fly.get_actuated_jointdofs_order()`` order.
    """

    actuator_type = ActuatorType(actuator_type)
    return np.asarray(
        [
            fly.jointdof_to_neutralaction_by_type[actuator_type][jointdof]
            for jointdof in fly.get_actuated_jointdofs_order(actuator_type)
        ],
        dtype=np.float32,
    )


def apply_controller_output(
    sim: Simulation,
    fly_name: str,
    output: ControllerOutput,
    *,
    actuator_type: ActuatorType = ActuatorType.POSITION,
) -> None:
    sim.set_actuator_inputs(fly_name, actuator_type, output.position_targets)
    sim.set_leg_adhesion_states(fly_name, output.adhesion.astype(np.float32))


def settle_simulation(
    sim: Simulation,
    controller: StepController | None = None,
    duration_s: float = 0.05,
    fly_name: str | None = None,
    render: bool = False,
) -> None:
    """Settle a simulation with valid actuator-order controls.

    This controlled pre-roll replaces raw ``sim.warmup()`` in tutorial demos. If a
    controller is supplied, its outputs are applied during settling. Otherwise each
    fly receives its neutral position-actuator targets and all adhesion actuators are
    held enabled. No records are returned, so callers naturally discard warmup data.
    """

    if duration_s <= 0:
        return
    if fly_name is None:
        fly_name = next(iter(sim.world.fly_lookup))
    fly = sim.world.fly_lookup[fly_name]
    neutral_targets = get_neutral_position_targets(fly, ActuatorType.POSITION)
    n_steps = int(round(duration_s / sim.timestep))
    for _ in range(n_steps):
        if hasattr(sim.world, "step"):
            sim.world.step(sim, sim.timestep)  # type: ignore[misc]
        if controller is None:
            sim.set_actuator_inputs(fly_name, ActuatorType.POSITION, neutral_targets)
            sim.set_leg_adhesion_states(fly_name, np.ones(6, dtype=np.float32))
        else:
            output = controller.step(sim)
            apply_controller_output(sim, fly_name, output)
        sim.step()
        if render and sim.renderer is not None:
            sim.render_as_needed()


def run_closed_loop(
    sim: Simulation,
    controller: StepController,
    duration_s: float,
    *,
    fly_name: str | None = None,
    drive_fn: Callable[[int, float, Simulation], Any] | None = None,
    render: bool = False,
    warmup_s: float = 0.0,
) -> list[dict[str, Any]]:
    """Run a v2 direct-control simulation loop.

    The world may optionally expose ``reset(sim)`` and ``step(sim, dt)`` hooks. Any
    warmup is performed through ``settle_simulation`` and discarded from returned
    records.
    """

    if fly_name is None:
        fly_name = next(iter(sim.world.fly_lookup))

    sim.reset()
    if hasattr(sim.world, "reset"):
        sim.world.reset(sim)  # type: ignore[misc]
    controller.reset()
    if warmup_s > 0:
        settle_simulation(
            sim,
            controller=controller,
            duration_s=warmup_s,
            fly_name=fly_name,
            render=render,
        )

    records: list[dict[str, Any]] = []
    n_steps = int(round(duration_s / sim.timestep))
    for step_idx in range(n_steps):
        t = sim.time
        if hasattr(sim.world, "step"):
            sim.world.step(sim, sim.timestep)  # type: ignore[misc]
        drive = drive_fn(step_idx, t, sim) if drive_fn is not None else None
        output = controller.step(sim, drive)
        apply_controller_output(sim, fly_name, output)
        sim.step()
        if render and sim.renderer is not None:
            sim.render_as_needed()
        records.append(
            {
                "time": sim.time,
                "joint_angles": sim.get_joint_angles(fly_name).copy(),
                "body_positions": sim.get_body_positions(fly_name).copy(),
                "position_targets": output.position_targets.copy(),
                "adhesion": output.adhesion.copy(),
                "metadata": dict(output.metadata),
            }
        )
    return records


def spawn_rotation_yaw(yaw_rad: float = 0.0) -> Rotation3D:
    """Convenience helper for z-axis spawn rotations."""

    half = yaw_rad / 2
    return Rotation3D("quat", [np.cos(half), 0.0, 0.0, np.sin(half)])
