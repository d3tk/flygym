import numpy as np
import pytest

from flygym import Simulation
from flygym.anatomy import ContactBodiesPreset
from flygym.compose import ActuatorType
from flygym.utils.math import Rotation3D
from flygym_demo.examples import make_walking_fly, run_closed_loop
from flygym_demo.examples.locomotion import (
    HybridTurningController,
    PreprogrammedSteps,
    RuleBasedController,
)
from flygym_demo.examples.locomotion.cpg_controller import CPGController
from flygym_demo.examples.olfaction import add_odor_sensors, read_odor
from flygym_demo.examples.path_integration import LinearModel, path_integrate
from flygym_demo.examples.vision import Retina
from flygym_demo.examples.worlds import (
    BlocksTerrainWorld,
    GappedTerrainWorld,
    MixedTerrainWorld,
    MovingFlyWorld,
    MovingObjectWorld,
    OdorWorld,
)
from flygym_demo.examples.worlds.terrain import FlatDemoWorld


def _make_sim(world):
    fly = make_walking_fly(add_camera=False)
    world.add_fly(
        fly,
        spawn_position=[0, 0, 1.5],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False,
    )
    return Simulation(world), fly


def test_preprogrammed_steps_match_v2_actuator_order():
    sim, fly = _make_sim(FlatDemoWorld())
    steps = PreprogrammedSteps()
    phases = np.zeros(6)
    targets = steps.get_joint_angles_for_order(
        phases,
        np.ones(6),
        fly.get_actuated_jointdofs_order(ActuatorType.POSITION),
    )
    assert targets.shape == (42,)
    assert np.all(np.isfinite(targets))
    assert sim.mj_model.nu >= 42


def test_cpg_rule_and_hybrid_closed_loop_smoke():
    for controller_cls in (CPGController, RuleBasedController, HybridTurningController):
        sim, fly = _make_sim(FlatDemoWorld())
        records = run_closed_loop(sim, controller_cls(sim.timestep), 0.0003, fly_name=fly.name)
        assert len(records) == 3
        assert records[-1]["position_targets"].shape == (42,)
        assert np.all(np.isfinite(records[-1]["position_targets"]))


@pytest.mark.parametrize(
    "world",
    [
        GappedTerrainWorld(x_range=(-1, 2), y_range=(-2, 2)),
        BlocksTerrainWorld(x_range=(-1, 2), y_range=(-2, 2)),
        MixedTerrainWorld(),
        OdorWorld(),
        MovingObjectWorld(),
        MovingFlyWorld(),
    ],
)
def test_demo_worlds_compile(world):
    sim, fly = _make_sim(world)
    sim.reset()
    if hasattr(world, "reset"):
        world.reset(sim)
    assert sim.mj_model.nbody > 0
    assert fly.name in sim.world.fly_lookup


def test_odor_sites_and_world_readout():
    fly = make_walking_fly(add_camera=False)
    sites = add_odor_sensors(fly)
    world = OdorWorld(
        odor_source=np.array([[10, 0, 0], [20, 0, 0]]),
        peak_odor_intensity=np.eye(2),
    )
    world.add_fly(
        fly,
        spawn_position=[0, 0, 1.5],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
        add_ground_contact_sensors=False,
    )
    sim = Simulation(world)
    odor = read_odor(sim, sites)
    assert odor.shape == (2, 4)
    assert np.all(np.isfinite(odor))


def test_retina_small_map_conversion():
    retina = Retina(
        ommatidia_id_map=np.array([[0, 1], [2, 2]], dtype=np.int16),
        pale_type_mask=np.array([0, 1]),
        nrows=2,
        ncols=2,
    )
    raw = np.zeros((2, 2, 3), dtype=np.uint8)
    raw[..., 1] = 255
    raw[..., 2] = 128
    vision = retina.raw_image_to_hex_pxls(raw)
    assert vision.shape == (2, 2)
    human = retina.hex_pxls_to_human_readable(vision)
    assert human.shape == (2, 2, 2)


def test_path_integration_linear_model_smoke():
    deltas = np.array([[1, 0, 0.1], [1, 0, -0.1], [0.5, 0.1, 0.0]])
    states = path_integrate(np.zeros(3), deltas)
    model = LinearModel().fit(deltas, states[1:])
    pred = model.predict(deltas[:1])
    assert states.shape == (4, 3)
    assert pred.shape == (1, 3)


def test_head_stabilization_zero_wrapper():
    from flygym_demo.examples.head_stabilization import HeadStabilizationInferenceWrapper

    wrapper = HeadStabilizationInferenceWrapper()
    pred = wrapper.predict(np.zeros(42), np.zeros(6))
    assert pred.shape == (2,)
