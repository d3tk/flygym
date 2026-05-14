from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from flygym.anatomy import BodySegment, ContactBodiesPreset, LEGS
from flygym.compose import ActuatorType, Fly, OdorWorld
from flygym.examples.locomotion import make_locomotion_fly
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D


def make_olfaction_fly(
    name: str = "nmf",
    *,
    draw_sensor_markers: bool = False,
    colorize: bool = True,
) -> Fly:
    fly = make_locomotion_fly(name=name, colorize=colorize)
    fly.add_olfaction(draw_sensor_markers=draw_sensor_markers)
    return fly


def add_fly_to_odor_world(
    world: OdorWorld,
    fly: Fly,
    spawn_position: tuple[float, float, float],
    spawn_orientation: tuple[float, float, float] | Rotation3D = (0.0, 0.0, 0.0),
    *,
    add_ground_contact_sensors: bool = False,
) -> None:
    if isinstance(spawn_orientation, Rotation3D):
        spawn_rotation = spawn_orientation
    else:
        spawn_rotation = Rotation3D("quat", _euler_xyz_to_quat_wxyz(spawn_orientation))
    world.add_fly(
        fly,
        spawn_position,
        spawn_rotation,
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=add_ground_contact_sensors,
    )


def make_olfaction_observation(
    sim: Simulation,
    fly_name: str,
) -> dict[str, np.ndarray]:
    fly = sim.world.fly_lookup[fly_name]
    body_positions = sim.get_body_positions(fly_name)
    body_order = fly.get_bodysegs_order()
    thorax_idx = body_order.index(BodySegment("c_thorax"))
    thorax_body_id = sim._internal_bodyids_by_fly[fly_name][thorax_idx]
    thorax_xmat = sim.mj_data.xmat[thorax_body_id].reshape(3, 3)

    tarsus5 = [BodySegment(f"{leg}_tarsus5") for leg in LEGS]
    end_effectors = np.array(
        [body_positions[body_order.index(segment)] for segment in tarsus5],
        dtype=np.float32,
    )
    contact_forces = sim.get_bodysegment_contact_forces(fly_name, tarsus5)
    actuated_dofs = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    all_dofs = fly.get_jointdofs_order()
    actuated_indices = [all_dofs.index(dof) for dof in actuated_dofs]
    actuator_forces = sim.get_actuator_forces(fly_name, ActuatorType.POSITION)
    joint_obs = np.vstack(
        (
            sim.get_joint_angles(fly_name)[actuated_indices],
            sim.get_joint_velocities(fly_name)[actuated_indices],
            actuator_forces,
        )
    )

    return {
        "joints": joint_obs.astype(np.float32),
        "fly": body_positions[[thorax_idx]].astype(np.float32),
        "contact_forces": contact_forces.astype(np.float32),
        "end_effectors": end_effectors,
        "fly_orientation": thorax_xmat[:, 0].astype(np.float32),
        "cardinal_vectors": thorax_xmat.T.astype(np.float32),
        "odor_intensity": sim.get_odor_intensity(fly_name).astype(np.float32),
    }


def add_world_camera(
    world: OdorWorld,
    name: str,
    *,
    pos: tuple[float, float, float],
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fovy: float = 45.0,
    mode: str = "fixed",
):
    return world.mjcf_root.worldbody.add(
        "camera",
        name=name,
        mode=mode,
        pos=pos,
        euler=euler,
        fovy=fovy,
    )


def _euler_xyz_to_quat_wxyz(euler: tuple[float, float, float]) -> tuple[float, ...]:
    quat_xyzw = Rotation.from_euler("xyz", euler).as_quat()
    return (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
