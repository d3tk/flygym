from __future__ import annotations

from os import PathLike

import numpy as np

from flygym import Simulation
from flygym.compose import OdorWorld
from flygym.examples.locomotion import (
    HybridTurningController,
    apply_locomotion_action,
)
from flygym.examples.olfaction.common import (
    add_fly_to_odor_world,
    add_world_camera,
    make_olfaction_fly,
    make_olfaction_observation,
)


def run_simulation(
    odor_source: np.ndarray,
    peak_odor_intensity: np.ndarray,
    marker_colors: np.ndarray | None = None,
    spawn_pos: tuple[float, float, float] = (0, 0, 0.8),
    spawn_orientation: tuple[float, float, float] = (0, 0, 0),
    run_time: float = 5,
    decision_interval: float = 0.05,
    attractive_gain: float = -500,
    aversive_gain: float = 80,
    attractive_palps_antennae_weights: tuple[float, float] = (1, 9),
    aversive_palps_antennae_weights: tuple[float, float] = (0, 10),
    target_pos: np.ndarray | None = None,
    distance_threshold: float = 2,
    video_path: PathLike | None = None,
    enable_rendering: bool = False,
    return_sim: bool = False,
) -> list[dict[str, np.ndarray]] | tuple[list[dict[str, np.ndarray]], Simulation]:
    world = OdorWorld(
        odor_source=odor_source,
        peak_odor_intensity=peak_odor_intensity,
        diffuse_func=lambda x: x**-2,
        marker_colors=marker_colors,
        marker_size=0.3,
    )
    if target_pos is None:
        target_pos = odor_source[0, :2]

    fly = make_olfaction_fly()
    add_fly_to_odor_world(world, fly, spawn_pos, spawn_orientation)

    camera = None
    if enable_rendering:
        camera = add_world_camera(
            world,
            "birdeye_cam",
            pos=(float(odor_source[:, 0].max()) / 2, 0, 35),
            fovy=45,
        )

    sim = Simulation(world)
    renderer = None
    if camera is not None:
        renderer = sim.set_renderer(
            [camera],
            camera_res=(240, 320),
            playback_speed=0.1,
            output_fps=25,
        )
    controller = HybridTurningController(
        timestep=sim.timestep,
        output_dof_order=fly.get_actuated_jointdofs_order("position"),
    )

    obs_hist = []
    num_decision_steps = int(run_time / decision_interval)
    physics_steps_per_decision_step = int(decision_interval / sim.timestep)

    obs = make_olfaction_observation(sim, fly.name)
    for _ in range(num_decision_steps):
        attractive_intensities = np.average(
            obs["odor_intensity"][0, :].reshape(2, 2),
            axis=0,
            weights=attractive_palps_antennae_weights,
        )
        aversive_intensities = np.average(
            obs["odor_intensity"][1, :].reshape(2, 2),
            axis=0,
            weights=aversive_palps_antennae_weights,
        )
        attractive_bias = (
            attractive_gain
            * (attractive_intensities[0] - attractive_intensities[1])
            / attractive_intensities.mean()
        )
        aversive_bias = (
            aversive_gain
            * (aversive_intensities[0] - aversive_intensities[1])
            / aversive_intensities.mean()
        )
        effective_bias = aversive_bias + attractive_bias
        effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)

        control_signal = np.ones(2)
        side_to_modulate = int(effective_bias_norm > 0)
        control_signal[side_to_modulate] -= np.abs(effective_bias_norm) * 0.8

        for _ in range(physics_steps_per_decision_step):
            action = controller.step(control_signal, sim, fly.name)
            apply_locomotion_action(sim, fly.name, action)
            sim.step()
            obs = make_olfaction_observation(sim, fly.name)
            obs_hist.append(obs)
            if renderer is not None:
                sim.render_as_needed()

        if np.linalg.norm(obs["fly"][0, :2] - target_pos) < distance_threshold:
            break

    if renderer is not None and video_path is not None:
        renderer.save_video(video_path)

    if return_sim:
        return obs_hist, sim
    return obs_hist


if __name__ == "__main__":
    sources = np.array([[24, 0, 1.5], [8, -4, 1.5], [16, 4, 1.5]])
    peaks = np.array([[1, 0], [0, 1], [0, 1]])
    colors = np.array(
        [
            [255, 127, 14, 255],
            [31, 119, 180, 255],
            [31, 119, 180, 255],
        ],
        dtype=float,
    )
    run_simulation(
        sources,
        peaks,
        colors / 255,
        spawn_orientation=(0, 0, np.pi / 2),
        video_path="./outputs/odor_taxis.mp4",
        enable_rendering=True,
    )
