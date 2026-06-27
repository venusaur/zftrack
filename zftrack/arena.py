"""Estimate the circular arena (centre + radius) for thigmotaxis analysis.

Tries a Hough-circle fit on the static background first (the dish is a strong
circle); if that fails or looks implausible, falls back to a robust estimate
from the fish positions themselves.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_arena(background: np.ndarray,
                 positions: np.ndarray | None = None) -> tuple[float, float, float]:
    """Return ``(cx, cy, radius)`` of the arena in pixels."""
    h, w = background.shape[:2]
    gray = background if background.ndim == 2 else cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h,
        param1=100, param2=50,
        minRadius=int(0.20 * min(h, w)), maxRadius=int(0.55 * min(h, w)),
    )
    if circles is not None:
        cx, cy, r = circles[0][0]
        # Sanity: circle centre should be roughly central.
        if abs(cx - w / 2) < 0.25 * w and abs(cy - h / 2) < 0.25 * h:
            return float(cx), float(cy), float(r)

    if positions is not None and len(positions) > 10:
        cx, cy = np.median(positions, axis=0)
        d = np.hypot(positions[:, 0] - cx, positions[:, 1] - cy)
        return float(cx), float(cy), float(np.percentile(d, 98))

    return w / 2.0, h / 2.0, 0.45 * min(h, w)
