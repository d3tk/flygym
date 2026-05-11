from __future__ import annotations

import mujoco as mj
import numpy as np

from flygym.compose import Fly
from flygym.utils.api1to2 import BODY_NAMES_OLD2NEW


ODOR_SENSOR_SPECS = {
    "LMaxillaryPalp_sensor": ("Rostrum", [-0.15, 0.15, -0.15]),
    "RMaxillaryPalp_sensor": ("Rostrum", [-0.15, -0.15, -0.15]),
    "LAntenna_sensor": ("LFuniculus", [0.02, 0.00, -0.10]),
    "RAntenna_sensor": ("RFuniculus", [0.02, 0.00, -0.10]),
}


def add_odor_sensors(fly: Fly) -> dict[str, object]:
    """Add four odor sensor sites to a fly and return the MJCF site elements."""

    sites = {}
    for sensor_name, (legacy_parent, rel_pos) in ODOR_SENSOR_SPECS.items():
        parent_name = BODY_NAMES_OLD2NEW[legacy_parent]
        parent = fly.mjcf_root.find("body", parent_name)
        site = parent.add("site", name=sensor_name, pos=rel_pos, size=(0.03,), rgba=(0.9, 0.73, 0.08, 1))
        sites[sensor_name] = site
    return sites


def get_odor_sensor_positions(sim, sites: dict[str, object]) -> np.ndarray:
    positions = []
    for site in sites.values():
        site_id = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_SITE, site.full_identifier)
        positions.append(sim.mj_data.site_xpos[site_id].copy())
    return np.asarray(positions)


def read_odor(sim, sites: dict[str, object]) -> np.ndarray:
    if not hasattr(sim.world, "get_olfaction"):
        raise TypeError("Simulation world does not expose get_olfaction(sensor_pos)")
    return sim.world.get_olfaction(get_odor_sensor_positions(sim, sites))
