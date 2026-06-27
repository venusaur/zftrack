"""Static background estimation via the per-pixel median of sampled frames.

Because the fish move and the arena does not, the median over a spread of frames
converges to the empty arena: any given pixel shows background in the majority of
frames, so the median is the background even where a fish occasionally passes.
"""

from __future__ import annotations

import cv2
import numpy as np


def build_background(video_path: str, num_samples: int = 80,
                     mode: str = "median") -> np.ndarray:
    """Return a grayscale background image for ``video_path``.

    Parameters
    ----------
    video_path:
        Path to the input video.
    num_samples:
        How many frames, evenly spaced across the whole clip, to estimate from.
    mode:
        ``"median"`` (default) — robust empty-scene estimate, good when subjects
        move around. ``"max"`` — per-pixel brightest value, the right choice for
        dark subjects on a bright field (e.g. larvae in well plates): it recovers
        the bright well floor wherever a fish *ever* vacated a pixel, so even a
        mostly-stationary fish is not baked into the background.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Video reports no frames: {video_path}")

    num_samples = max(5, min(num_samples, total))
    indices = np.linspace(0, total - 1, num_samples).astype(int)

    frames = []
    max_bg = None
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if mode == "max":
            max_bg = gray if max_bg is None else np.maximum(max_bg, gray)
        else:
            frames.append(gray)
    cap.release()

    if mode == "max":
        if max_bg is None:
            raise RuntimeError("Failed to read any frames for background estimation")
        return max_bg
    if not frames:
        raise RuntimeError("Failed to read any frames for background estimation")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
