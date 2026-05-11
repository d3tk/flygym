from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import importlib

from scripts.tutorials.common import ensure_headless_mujoco_gl

TUTORIAL_MODULES = (
    "gym_basics_and_kinematic_replay",
    "cpg_controller",
    "rule_based_controller",
    "hybrid_controller",
    "turning",
    "olfaction_basics",
    "advanced_olfaction",
    "vision_basics",
    "advanced_vision",
    "path_integration",
    "head_stabilization",
)


def main(argv: list[str] | None = None) -> None:
    ensure_headless_mujoco_gl()
    parser = argparse.ArgumentParser(
        description="Generate all script-based tutorial outputs."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    for module_name in TUTORIAL_MODULES:
        module = importlib.import_module(f"scripts.tutorials.{module_name}")
        module_args = ["--output-dir", str(args.output_dir)]
        if args.quick:
            module_args.append("--quick")
        created = module.main(module_args)
        print(f"{module.SPEC.name}: {len(created)} artifacts")


if __name__ == "__main__":
    main()
