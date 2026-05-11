from __future__ import annotations

import importlib
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from flygym import Simulation
from flygym.anatomy import ContactBodiesPreset
from flygym.compose import ActuatorType
from flygym.utils.math import Rotation3D
from flygym_demo.examples import (
    get_neutral_position_targets,
    make_walking_fly,
    run_closed_loop,
    settle_simulation,
)
from flygym_demo.examples.locomotion.cpg_controller import CPGController
from flygym_demo.examples.worlds.terrain import FlatDemoWorld
from scripts.tutorials.run_all import TUTORIAL_MODULES


def _make_sim():
    fly = make_walking_fly(add_camera=False)
    world = FlatDemoWorld()
    world.add_fly(
        fly,
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False,
    )
    return Simulation(world), fly


def test_make_walking_fly_is_colorized_by_default():
    fly = make_walking_fly(add_camera=False)
    material_names = {
        material.name for material in fly.mjcf_root.asset.find_all("material")
    }
    assert material_names
    assert "eye" in material_names


def test_neutral_position_targets_are_in_actuator_order():
    fly = make_walking_fly(add_camera=False)
    targets = get_neutral_position_targets(fly)
    assert targets.shape == (42,)
    assert targets.shape == (
        len(fly.get_actuated_jointdofs_order(ActuatorType.POSITION)),
    )
    assert np.all(np.isfinite(targets))


def test_controlled_warmup_uses_actuator_sized_targets(monkeypatch):
    sim, fly = _make_sim()
    seen_lengths = []
    original_set_actuator_inputs = sim.set_actuator_inputs

    def spy_set_actuator_inputs(fly_name, actuator_type, inputs):
        seen_lengths.append(len(inputs))
        return original_set_actuator_inputs(fly_name, actuator_type, inputs)

    def fail_warmup(duration_s=0.05):  # pragma: no cover - called only on regression
        raise AssertionError("run_closed_loop must use settle_simulation, not sim.warmup")

    monkeypatch.setattr(sim, "set_actuator_inputs", spy_set_actuator_inputs)
    monkeypatch.setattr(sim, "warmup", fail_warmup)
    settle_simulation(sim, duration_s=sim.timestep, fly_name=fly.name)
    controller = CPGController(sim.timestep)
    records = run_closed_loop(
        sim,
        controller,
        sim.timestep,
        fly_name=fly.name,
        warmup_s=sim.timestep,
    )
    assert records
    assert seen_lengths
    assert set(seen_lengths) == {42}


def test_tutorial_scripts_quick_mode_create_manifest_outputs(tmp_path: Path):
    for module_name in TUTORIAL_MODULES:
        module = importlib.import_module(f"scripts.tutorials.{module_name}")
        created = module.main(["--output-dir", str(tmp_path), "--quick"])
        out_dir = tmp_path / module.SPEC.name
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["artifacts"] == list(module.SPEC.artifacts)
        assert len(created) == len(module.SPEC.artifacts)
        for artifact in module.SPEC.artifacts:
            path = out_dir / artifact
            assert path.is_file()
            assert path.stat().st_size > 0


def test_quick_videos_contain_frames(tmp_path: Path):
    module = importlib.import_module("scripts.tutorials.turning")
    module.main(["--output-dir", str(tmp_path), "--quick"])
    frames = iio.imread(
        tmp_path / module.SPEC.name / "turning_video.mp4", index=None
    )
    assert frames.shape[0] > 0
