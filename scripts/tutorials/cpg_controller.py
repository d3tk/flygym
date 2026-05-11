from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.tutorials.common import TutorialSpec, main_for

SPEC = TutorialSpec(
    name="cpg_controller",
    artifacts=(
        "cpg_phase_plot.png",
        "cpg_signal_plot.png",
        "adhesion_plot.png",
        "cpg_controller_video.mp4",
        "cpg_controller_adhesion_video.mp4",
    ),
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
