from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.tutorials.common import TutorialSpec, main_for

SPEC = TutorialSpec(
    name="advanced_vision",
    artifacts=(
        "two_flies_walking_video.mp4",
        "retina_activity_plot.png",
        "vision_video.mp4",
    ),
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
