"""Tests for arena estimation (Hough fit with a positions fallback)."""

from __future__ import annotations

import numpy as np

from zftrack.arena import detect_arena
from conftest import gray_background


def test_arena_falls_back_to_position_cloud():
    # A featureless background defeats Hough; the fish cloud drives the estimate.
    bg = gray_background(200, 200)
    rng = np.random.default_rng(0)
    cx, cy, r = 120.0, 90.0, 40.0
    ang = rng.uniform(0, 2 * np.pi, 500)
    rad = rng.uniform(0, r, 500)
    pos = np.column_stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)])
    ex, ey, er = detect_arena(bg, pos)
    assert abs(ex - cx) < 10 and abs(ey - cy) < 10
    assert r * 0.7 < er < r * 1.3


def test_arena_default_when_no_circle_no_positions():
    bg = gray_background(200, 160)
    cx, cy, r = detect_arena(bg, None)
    assert abs(cx - 80) < 1 and abs(cy - 100) < 1     # image centre
    assert abs(r - 0.45 * 160) < 1                     # 0.45 * min(h, w)
