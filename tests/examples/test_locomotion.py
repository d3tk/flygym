"""Tests for v2-native locomotion example controllers."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from flygym import Simulation, assets_dir
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld, MixedTerrainWorld
from flygym.examples.locomotion import (
    CPGController,
    CPGNetwork,
    HybridController,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    RuleBasedController,
    apply_locomotion_action,
    calculate_ddt,
    get_cpg_biases,
    make_locomotion_fly,
    make_tripod_cpg_network,
)
from flygym.examples.locomotion.hybrid_controller import _step_phase_gain
from flygym.utils.math import Rotation3D


def test_calculate_ddt_shapes_and_values():
    theta = np.zeros(3)
    r = np.ones(3)
    w = np.zeros((3, 3))
    phi = np.zeros((3, 3))
    nu = np.ones(3) * 2
    R = np.ones(3) * 3
    alpha = np.ones(3) * 4

    dtheta_dt, dr_dt = calculate_ddt(theta, r, w, phi, nu, R, alpha)

    np.testing.assert_allclose(dtheta_dt, np.ones(3) * 4 * np.pi)
    np.testing.assert_allclose(dr_dt, np.ones(3) * 8)


def test_cpg_network_seed_is_deterministic():
    kwargs = dict(
        timestep=1e-3,
        intrinsic_freqs=np.ones(6),
        intrinsic_amps=np.ones(6),
        coupling_weights=np.zeros((6, 6)),
        phase_biases=np.zeros((6, 6)),
        convergence_coefs=np.ones(6),
        seed=123,
    )
    net1 = CPGNetwork(**kwargs)
    net2 = CPGNetwork(**kwargs)

    np.testing.assert_allclose(net1.curr_phases, net2.curr_phases)
    np.testing.assert_allclose(net1.curr_magnitudes, net2.curr_magnitudes)


def test_get_cpg_biases_supported_gaits():
    for gait in ["tripod", "tetrapod", "wave"]:
        biases = get_cpg_biases(gait)
        assert biases.shape == (6, 6)
        np.testing.assert_allclose(np.diag(biases), 0)
    with pytest.raises(ValueError):
        get_cpg_biases("unknown")


@pytest.fixture(scope="module")
def preprogrammed_steps():
    return PreprogrammedSteps()


def test_preprogrammed_steps_default_pose_and_adhesion(preprogrammed_steps):
    assert preprogrammed_steps.default_pose.shape == (42,)
    assert np.all(np.isfinite(preprogrammed_steps.default_pose))

    phases = np.zeros(6)
    adhesion = preprogrammed_steps.get_adhesion_onoff_by_phase(phases)
    assert adhesion.shape == (6,)
    assert adhesion.dtype == bool


def test_preprogrammed_steps_right_roll_yaw_are_anatomical(preprogrammed_steps):
    path = assets_dir / "behavior/single_steps_untethered.pkl"
    with open(path, "rb") as f:
        raw_data = pickle.load(f)
    raw_phase_grid = np.linspace(0, 2 * np.pi, len(raw_data["joint_RFCoxa"]))
    raw_rf_roll = CubicSpline(
        raw_phase_grid,
        raw_data["joint_RFCoxa_roll"],
        bc_type="periodic",
    )

    phase = 0.5
    converted = preprogrammed_steps.get_joint_angles("rf", phase)[1]
    np.testing.assert_allclose(converted, -raw_rf_roll(phase))


def test_make_tripod_cpg_network_default_intrinsic_frequency_is_twelve_hz():
    net = make_tripod_cpg_network(1e-4, seed=0)
    np.testing.assert_allclose(net.intrinsic_freqs, 12.0)


def test_make_tripod_cpg_network_accepts_kinematic_step_cycle_hz(preprogrammed_steps):
    f = preprogrammed_steps.step_cycle_frequency_hz
    net = make_tripod_cpg_network(1e-4, intrinsic_frequency=f, seed=0)
    np.testing.assert_allclose(net.intrinsic_freqs, f)


def test_make_locomotion_fly_uses_v1_like_leg_compliance():
    fly = make_locomotion_fly(name="compliance_fly")
    actuated = set(fly.get_actuated_jointdofs_order("position"))

    for jointdof, joint in fly.jointdof_to_mjcfjoint.items():
        if jointdof in actuated:
            assert joint.stiffness == pytest.approx(0.05)
            assert joint.damping == pytest.approx(0.06)
        elif jointdof.child.link.startswith("tarsus"):
            assert joint.stiffness == pytest.approx(7.5)
            assert joint.damping == pytest.approx(1e-2)


def test_make_locomotion_fly_adhesion_actuators_allow_zero_control():
    fly = make_locomotion_fly(name="adhesion_fly")

    assert len(fly.leg_to_adhesionactuator) == 6
    for actuator in fly.leg_to_adhesionactuator.values():
        np.testing.assert_allclose(actuator.ctrlrange, [0, 100])


def test_rule_based_controller_first_step(preprogrammed_steps):
    controller = RuleBasedController(
        timestep=1e-3,
        preprogrammed_steps=preprogrammed_steps,
        seed=1,
    )
    action = controller.step()

    assert isinstance(action, LocomotionAction)
    assert action.joint_angles.shape == (42,)
    assert action.adhesion_onoff.shape == (6,)
    assert controller.mask_is_stepping.any()
    assert controller.curr_step == 1


def test_contact_force_aggregation_order_and_shape():
    fly = make_locomotion_fly(name="contact_fly")
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        [0, 0, 0.8],
        Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
    )
    sim = Simulation(world)
    sim.reset()
    sim.step()

    body_segments = [BodySegment("lf_tarsus5"), BodySegment("rf_tarsus5")]
    forces = sim.get_bodysegment_contact_forces(fly.name, body_segments)

    assert forces.shape == (2, 3)
    assert np.all(np.isfinite(forces))


def test_cpg_controller_short_flat_simulation(preprogrammed_steps):
    fly = make_locomotion_fly(name="cpg_fly")
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        [0, 0, 0.8],
        Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
    )
    sim = Simulation(world)
    sim.reset()

    controller = CPGController(
        make_tripod_cpg_network(sim.timestep, seed=2),
        preprogrammed_steps,
        fly.get_actuated_jointdofs_order("position"),
    )
    for _ in range(5):
        action = controller.step()
        apply_locomotion_action(sim, fly.name, action)
        sim.step()

    assert np.all(np.isfinite(sim.get_joint_angles(fly.name)))


def test_rule_based_controller_short_flat_simulation(preprogrammed_steps):
    fly = make_locomotion_fly(name="rule_fly")
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        [0, 0, 0.8],
        Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
    )
    sim = Simulation(world)
    sim.reset()

    controller = RuleBasedController(
        timestep=sim.timestep,
        preprogrammed_steps=preprogrammed_steps,
        output_dof_order=fly.get_actuated_jointdofs_order("position"),
        seed=3,
    )
    for _ in range(5):
        action = controller.step()
        apply_locomotion_action(sim, fly.name, action)
        sim.step()

    assert np.all(np.isfinite(sim.get_joint_angles(fly.name)))


def test_hybrid_controller_short_mixed_simulation(preprogrammed_steps):
    fly = make_locomotion_fly(name="hybrid_fly")
    world = MixedTerrainWorld()
    world.add_fly(
        fly,
        [0, 0, 1.2],
        Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False,
    )
    sim = Simulation(world)
    sim.reset()

    controller = HybridController(
        timestep=sim.timestep,
        preprogrammed_steps=preprogrammed_steps,
        output_dof_order=fly.get_actuated_jointdofs_order("position"),
    )
    for _ in range(3):
        action = controller.step(sim, fly.name)
        apply_locomotion_action(sim, fly.name, action)
        sim.step()

    assert np.all(np.isfinite(sim.get_joint_angles(fly.name)))
    assert "net_corrections" in controller.last_info
    assert controller.last_info["net_corrections"].shape == (6,)


def test_hybrid_controller_persistence_lasts_twenty_steps(preprogrammed_steps):
    controller = HybridController(
        timestep=1e-4,
        preprogrammed_steps=preprogrammed_steps,
    )
    controller.retraction_persistence_counter[0] = 19
    controller._update_persistence_counter()
    assert controller.retraction_persistence_counter[0] == 20

    controller._update_persistence_counter()
    assert controller.retraction_persistence_counter[0] == 0


def test_hybrid_controller_extends_swing_without_mutating_steps(preprogrammed_steps):
    original_swing_period = {
        leg: value.copy() for leg, value in preprogrammed_steps.swing_period.items()
    }
    controller = HybridController(
        timestep=1e-4,
        preprogrammed_steps=preprogrammed_steps,
    )

    leg = "lf"
    swing_start, swing_end = original_swing_period[leg]
    phase = swing_end + controller.swing_extension / 2

    assert preprogrammed_steps.get_adhesion_onoff(leg, phase)
    assert not controller._get_adhesion_onoff(leg, phase)
    assert _step_phase_gain(
        np.mean([swing_start, swing_end]), original_swing_period[leg]
    ) == pytest.approx(0.8)
    for leg, value in original_swing_period.items():
        np.testing.assert_allclose(preprogrammed_steps.swing_period[leg], value)


def test_hybrid_turning_controller_short_flat_simulation(preprogrammed_steps):
    fly = make_locomotion_fly(name="turning_fly")
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        [0, 0, 0.8],
        Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False,
    )
    sim = Simulation(world)
    sim.reset()

    controller = HybridTurningController(
        timestep=sim.timestep,
        preprogrammed_steps=preprogrammed_steps,
        output_dof_order=fly.get_actuated_jointdofs_order("position"),
    )
    descending_signal = np.array([1.2, -0.4])
    for _ in range(3):
        action = controller.step(descending_signal, sim, fly.name)
        apply_locomotion_action(sim, fly.name, action)
        sim.step()

    np.testing.assert_allclose(controller.cpg_network.intrinsic_amps[:3], 1.2)
    np.testing.assert_allclose(controller.cpg_network.intrinsic_amps[3:], 0.4)
    assert np.all(controller.cpg_network.intrinsic_freqs[:3] > 0)
    assert np.all(controller.cpg_network.intrinsic_freqs[3:] < 0)
    assert np.all(np.isfinite(sim.get_joint_angles(fly.name)))


def test_locomotion_tutorial_notebooks_are_valid_json():
    repo_root = Path(__file__).resolve().parents[2]
    notebook_names = [
        "2_replaying_experimental_recordings.ipynb",
        "4_cpg_controller.ipynb",
        "5_rule_based_controller.ipynb",
        "6_hybrid_controller.ipynb",
        "7_turning_controller.ipynb",
    ]
    for notebook_name in notebook_names:
        notebook_path = repo_root / "tutorials" / notebook_name
        notebook = json.loads(notebook_path.read_text())
        assert notebook["nbformat"] == 4
        assert "np.bool" not in notebook_path.read_text()
