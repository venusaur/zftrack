"""Shared fixtures and synthetic-data helpers for the zftrack test suite.

Everything here builds *synthetic* frames/videos so the tests are fast,
deterministic and need no sample footage. The convention matches the real data:
a bright field (well floor / lit arena) with darker fish on top.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from zftrack.detector import Detection


# ----- synthetic imagery --------------------------------------------------

BG_VALUE = 200      # bright background (well floor / lit arena)
FISH_VALUE = 40     # fish are darker than background


def gray_background(h: int = 120, w: int = 160, value: int = BG_VALUE) -> np.ndarray:
    """Uniform bright grayscale background."""
    return np.full((h, w), value, np.uint8)


def frame_with_fish(centroids, h: int = 120, w: int = 160,
                    axes: tuple[int, int] = (16, 7), angle: float = 0.0,
                    bg_value: int = BG_VALUE, fish_value: int = FISH_VALUE):
    """A BGR frame: bright field with a dark ellipse ('fish') at each centroid.

    ``axes`` are the ellipse half-axes (major, minor) in px, so the blob is
    clearly elongated -> length > width and a well-defined body axis.
    """
    gray = np.full((h, w), bg_value, np.uint8)
    for (cx, cy) in centroids:
        cv2.ellipse(gray, (int(cx), int(cy)), axes, angle, 0, 360, fish_value, -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def synthetic_video(path: str, n_frames: int = 40, h: int = 120, w: int = 160,
                    fps: float = 10.0):
    """Write a short mp4: one dark fish sweeping left->right on a bright field.

    Returns the per-frame centroid list so a test can check the background did
    *not* bake the moving fish in.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    assert writer.isOpened(), "could not open VideoWriter (codec missing?)"
    centroids = []
    for i in range(n_frames):
        cx = int(10 + (w - 20) * i / max(n_frames - 1, 1))
        cy = h // 2
        centroids.append((cx, cy))
        writer.write(frame_with_fish([(cx, cy)], h=h, w=w))
    writer.release()
    return centroids


# ----- synthetic detections (for the tracker, no image needed) -------------

def make_detection(cx: float, cy: float, area: float = 400.0,
                   length: float = 24.0, width: float = 12.0,
                   descriptor=None) -> Detection:
    """A minimal Detection at (cx, cy) with a body axis along x."""
    half_w = max(int(length / 2), 1)
    contour = np.array(
        [[[int(cx - half_w), int(cy)]], [[int(cx), int(cy - 3)]],
         [[int(cx + half_w), int(cy)]], [[int(cx), int(cy + 3)]]],
        dtype=np.int32,
    )
    if descriptor is None:
        descriptor = np.array([area, length, width, FISH_VALUE], dtype=np.float32)
    return Detection(
        centroid=(float(cx), float(cy)),
        bbox=(int(cx - half_w), int(cy - 3), 2 * half_w, 6),
        area=area,
        contour=contour,
        length=length,
        width=width,
        axis_deg=0.0,
        end_a=(cx + half_w, cy),
        end_b=(cx - half_w, cy),
        end_a_intensity=FISH_VALUE,
        end_b_intensity=FISH_VALUE + 30,
        descriptor=descriptor,
    )


@pytest.fixture
def bg():
    return gray_background()
