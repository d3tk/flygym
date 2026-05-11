from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.tutorials.common import TutorialSpec, main_for

SPEC = TutorialSpec(
    name="rule_based_controller",
    artifacts=(
        "preprogrammed_step_plot.png",
        "rules_graph.png",
        "control_signal_plot.png",
        "rule_based_controller_video.mp4",
    ),
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
