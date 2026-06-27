"""zftrack - zebrafish tracking from top-down arena video.

A lightweight, dependency-light multi-object tracker built around classical
computer vision:

    median background subtraction  ->  blob detection  ->  Kalman + Hungarian
    data association  ->  annotated video + trajectory CSV

The arena and camera are assumed static (true for typical lab behavioural
recordings), so a median background model cleanly cancels the dish, holders and
vignette, leaving only the moving fish.
"""

from .background import build_background
from .detector import BlobDetector, Detection
from .tracker import MultiObjectTracker, Track
from .pipeline import process_video, TrackerConfig
from .analysis import analyze_tracks
from .arena import detect_arena

__all__ = [
    "build_background",
    "BlobDetector",
    "Detection",
    "MultiObjectTracker",
    "Track",
    "process_video",
    "TrackerConfig",
    "analyze_tracks",
    "detect_arena",
]
