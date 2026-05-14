from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

from flygym import assets_dir
from flygym.compose import OdorPlumeWorld
from flygym.examples.olfaction import (
    PlumeNavigationController,
    PlumeNavigationTask,
    WalkingState,
)
from flygym.examples.olfaction.common import add_world_camera, make_olfaction_fly
from flygym.examples.olfaction.simple_odor_taxis import run_simulation
from flygym.utils.math import Rotation3D


def test_olfaction_tutorial_notebooks_are_valid_json():
    repo_root = Path(__file__).resolve().parents[2]
    notebook_names = [
        "8_olfaction_basics.ipynb",
        "9_advanced_olfaction.ipynb",
    ]
    for notebook_name in notebook_names:
        notebook_path = repo_root / "tutorials" / notebook_name
        notebook_text = notebook_path.read_text()
        notebook = json.loads(notebook_text)
        assert notebook["nbformat"] == 4
        assert "flygym_gymnasium" not in notebook_text


@pytest.mark.parametrize(
    ("odor_source", "peak_odor_intensity"),
    [
        (
            np.array([[24, 0, 1.5], [8, -4, 1.5], [16, 4, 1.5]], dtype=float),
            np.array([[1, 0], [0, 1], [0, 1]], dtype=float),
        ),
        (
            np.array([[3, 3, 1.5], [3, -3, 1.5]], dtype=float),
            np.eye(2),
        ),
        (
            np.array([[-5, 0, 1.5]], dtype=float),
            np.array([[1, 1e-3]], dtype=float),
        ),
    ],
)
def test_odor_taxis_example_reaches_source(odor_source, peak_odor_intensity):
    target_pos = odor_source[0, :2]

    obs_hist = run_simulation(
        odor_source,
        peak_odor_intensity=peak_odor_intensity,
        spawn_orientation=(0, 0, np.pi / 2),
        run_time=5,
        target_pos=target_pos,
        distance_threshold=2,
    )

    assert np.linalg.norm(obs_hist[-1]["fly"][0, :2] - target_pos) <= 2


def test_plume_arena_readout_matches_reference_data():
    pytest.importorskip("h5py")
    plume_data_path = assets_dir / "olfaction/plume_short.hdf5"
    arena = OdorPlumeWorld(
        plume_data_path=plume_data_path,
        plume_simulation_fps=20,
        intensity_scale_factor=1.0,
    )
    positions_1 = np.array([[0, 80, 1], [10, 80, 1], [52, 80, 1], [75, 105, 1]])
    positions_2 = np.array([[75, 15, 1], [10, 80, 1], [239, 159, 1], [240, 160, 1]])

    intensities_1 = arena.get_olfaction(positions_1)
    intensities_2 = arena.get_olfaction(positions_2)

    assert intensities_1[0, :] == pytest.approx(
        [0, 0.45141602, 0.58837891, 0.17651369]
    )
    assert intensities_2[0, :3] == pytest.approx([0, 0.45141602, 0])
    assert np.isnan(intensities_2[0, 3])
    arena.close()


def test_plume_navigation_controller_reference_tendencies():
    controller = PlumeNavigationController(dt=1e-3, random_seed=0)
    state_hist_no_encounter = []
    state_hist_all_encounter = []
    for _ in range(100000):
        state, _, _ = controller.decide_state(
            encounter_flag=False,
            fly_heading=np.array([-1, 1]),
        )
        state_hist_no_encounter.append(state)
    counter_no_encounter = Counter(state_hist_no_encounter)

    controller = PlumeNavigationController(dt=1e-3, random_seed=0)
    for _ in range(100000):
        state, _, _ = controller.decide_state(
            encounter_flag=True,
            fly_heading=np.array([-1, 1]),
        )
        state_hist_all_encounter.append(state)
    counter_all_encounter = Counter(state_hist_all_encounter)

    assert (
        counter_all_encounter[WalkingState.FORWARD]
        > counter_no_encounter[WalkingState.FORWARD] * 1.5
    )
    assert (
        counter_all_encounter[WalkingState.TURN_LEFT]
        > counter_all_encounter[WalkingState.TURN_RIGHT] * 2
    )
    ratio = (
        counter_no_encounter[WalkingState.TURN_LEFT]
        / counter_no_encounter[WalkingState.TURN_RIGHT]
    )
    assert 0.75 < ratio < 1.25


def test_plume_navigation_task_observation_and_overlay():
    pytest.importorskip("h5py")
    plume_data_path = assets_dir / "olfaction/plume_short.hdf5"
    arena = OdorPlumeWorld(
        plume_data_path=plume_data_path,
        main_camera_name="birdeye_cam",
        plume_simulation_fps=20,
    )
    fly = make_olfaction_fly(draw_sensor_markers=True)
    camera = add_world_camera(
        arena,
        "birdeye_cam",
        pos=(0.50 * arena.arena_size[0], 0.15 * arena.arena_size[1], arena.arena_size[1]),
        euler=(np.deg2rad(15), 0, 0),
        fovy=60,
    )
    task = PlumeNavigationTask(
        fly=fly,
        arena=arena,
        cameras=[camera],
        spawn_position=(40, 80, 0.8),
        spawn_rotation=Rotation3D("quat", (0, 0, 0, -1)),
        camera_res=(80, 120),
    )

    obs_hist = []
    info_hist = []
    rendered_images = []
    for _ in range(int(0.05 / task.timestep)):
        obs, _, _, _, info = task.step(np.array([1, 1]))
        obs_hist.append(obs)
        info_hist.append(info)
        img = task.render()[0]
        if img is not None:
            rendered_images.append(img)

    expected_obs_keys = {
        "joints",
        "fly",
        "contact_forces",
        "end_effectors",
        "fly_orientation",
        "odor_intensity",
        "cardinal_vectors",
    }
    expected_info_keys = {"net_corrections", "joints", "adhesion", "flip"}
    assert all(set(obs) == expected_obs_keys for obs in obs_hist)
    assert all(set(info) == expected_info_keys for info in info_hist)
    assert np.array([obs["odor_intensity"] for obs in obs_hist]).shape[1:] == (1, 4)
    assert len(rendered_images) >= 2
    assert not np.all(rendered_images[0] == rendered_images[-1])
    arena.close()


def test_plume_simulation_two_step_reference():
    pytest.importorskip("phi.torch")
    from flygym.examples.olfaction.simulate_plume_dataset import (
        generate_simulation_inputs,
        get_simulation_parameters,
        run_simulation as run_plume_simulation,
    )

    np.random.seed(0)
    params = get_simulation_parameters(simulation_time=2)
    (
        dt,
        arena_size,
        inflow_pos,
        inflow_radius,
        inflow_scaler,
        velocity_grid_size,
        smoke_grid_size,
        simulation_steps,
    ) = params
    wind_hist, velocity, smoke, inflow = generate_simulation_inputs(
        simulation_steps,
        arena_size,
        inflow_pos,
        inflow_radius,
        inflow_scaler,
        velocity_grid_size,
        smoke_grid_size,
    )
    smoke_hist = run_plume_simulation(wind_hist, velocity, smoke, inflow, dt, arena_size)

    assert wind_hist[0] == pytest.approx(np.array([0, 0]))
    assert wind_hist[1] == pytest.approx(np.array([0.35281047, 0.08003144]))
    assert smoke_hist[0].sum() == pytest.approx(10.15325)
    assert smoke_hist[1].sum() == pytest.approx(20.316465)
