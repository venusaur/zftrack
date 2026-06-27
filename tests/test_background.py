"""Tests for background estimation (median vs max) from a synthetic video."""

from __future__ import annotations

import numpy as np
import pytest

from zftrack.background import build_background
from conftest import synthetic_video, BG_VALUE, FISH_VALUE


def test_median_background_removes_moving_fish(tmp_path):
    path = str(tmp_path / "clip.mp4")
    synthetic_video(path, n_frames=40)
    bg = build_background(path, mode="median")
    assert bg.ndim == 2
    # The moving fish never occupies one pixel for the majority of frames, so the
    # median is the bright field everywhere.
    assert bg.mean() > BG_VALUE - 20
    assert bg.min() > (BG_VALUE + FISH_VALUE) / 2     # no fish baked in


def test_max_background_is_bright_field(tmp_path):
    path = str(tmp_path / "clip.mp4")
    synthetic_video(path, n_frames=40)
    bg = build_background(path, mode="max")
    # Per-pixel brightest recovers the field wherever the fish ever vacated.
    assert bg.min() >= BG_VALUE - 5


def test_missing_video_raises(tmp_path):
    with pytest.raises((IOError, OSError)):
        build_background(str(tmp_path / "does_not_exist.mp4"))
