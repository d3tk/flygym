from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.tutorials.common import TutorialSpec, main_for

SPEC = TutorialSpec(
    name="head_stabilization",
    artifacts=(
        "rendering_video.mp4",
        "trajectory_plot.png",
        "training_plot.png",
        "neck_actuation_plot.png",
        "head_thorax_plot.png",
    ),
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
