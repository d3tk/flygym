import numpy as np
import pytest

from flygym import Simulation
from flygym.compose import Fly, OdorWorld
from flygym.utils.math import Rotation3D


def test_add_olfaction_sensor_order_and_markers():
    fly = Fly(name="odor_fly")
    sensors = fly.add_olfaction(draw_sensor_markers=True)

    assert list(sensors) == [
        "l_maxillary_palp",
        "r_maxillary_palp",
        "l_antenna",
        "r_antenna",
    ]
    for sensor_name in sensors:
        assert fly.mjcf_root.find("site", sensor_name) is not None
        assert fly.mjcf_root.find("geom", f"{sensor_name}_marker") is not None


def test_olfaction_sensor_positions_shape():
    fly = Fly(name="sensor_fly")
    fly.add_olfaction()
    world = OdorWorld()
    world.add_fly(
        fly,
        [0, 0, 1.0],
        Rotation3D("quat", [1, 0, 0, 0]),
        add_ground_contact_sensors=False,
    )
    sim = Simulation(world)
    sim.step()

    positions = sim.get_olfaction_sensor_positions(fly.name)

    assert positions.shape == (4, 3)
    assert np.all(np.isfinite(positions))


def test_odor_world_intensity_matches_manual_calculation():
    odor_source = np.array([[0, 0, 0], [3, 0, 0]], dtype=float)
    peak = np.array([[2, 0], [0, 4]], dtype=float)
    sensor_positions = np.array([[1, 0, 0], [0, 2, 0]], dtype=float)
    world = OdorWorld(odor_source=odor_source, peak_odor_intensity=peak)

    intensity = world.get_olfaction(sensor_positions)

    expected = np.zeros((2, 2))
    for source, source_peak in zip(odor_source, peak):
        dist = np.linalg.norm(sensor_positions - source, axis=1)
        expected += source_peak[:, np.newaxis] * dist[np.newaxis, :] ** -2
    np.testing.assert_allclose(intensity, expected)


def test_get_odor_intensity_requires_fly_olfaction():
    fly = Fly(name="no_odor_fly")
    world = OdorWorld()
    world.add_fly(
        fly,
        [0, 0, 1.0],
        Rotation3D("quat", [1, 0, 0, 0]),
        add_ground_contact_sensors=False,
    )
    sim = Simulation(world)

    with pytest.raises(ValueError, match="add_olfaction"):
        sim.get_odor_intensity(fly.name)
