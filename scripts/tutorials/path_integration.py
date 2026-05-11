from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.tutorials.common import TutorialSpec, main_for

SPEC = TutorialSpec(
    name="path_integration",
    artifacts=(
        "all_trajectory_plot.png",
        "end_effector_plot.png",
        "contact_plot.png",
        "predictor_plot.png",
        "cumulative_integration_plot.png",
    ),
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
