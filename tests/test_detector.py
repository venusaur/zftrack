"""Tests for blob detection and body-shape features."""

from __future__ import annotations

import numpy as np

from zftrack.detector import BlobDetector, _patch_intensity, _shape_features
from conftest import frame_with_fish, gray_background, BG_VALUE, FISH_VALUE


def test_patch_intensity_reads_local_mean():
    gray = gray_background(value=200)
    gray[50:60, 50:60] = 10
    assert _patch_intensity(gray, 55, 55, r=3) < 50    # over the dark patch
    assert _patch_intensity(gray, 5, 5, r=3) == 200    # over background
    # Fully out-of-bounds patch returns the sentinel.
    assert _patch_intensity(gray, -100, -100) == 255.0


def test_detect_finds_single_fish():
    bg = gray_background()
    frame = frame_with_fish([(80, 60)])
    dets = BlobDetector(bg).detect(frame)
    assert len(dets) == 1
    cx, cy = dets[0].centroid
    assert abs(cx - 80) < 3 and abs(cy - 60) < 3
    assert dets[0].length > dets[0].width    # ellipse is elongated


def test_detect_finds_multiple_fish():
    bg = gray_background()
    frame = frame_with_fish([(40, 30), (120, 90)])
    dets = BlobDetector(bg).detect(frame)
    assert len(dets) == 2
    centers = sorted(d.centroid for d in dets)
    assert abs(centers[0][0] - 40) < 4
    assert abs(centers[1][0] - 120) < 4


def test_area_filter_rejects_out_of_range_blobs():
    bg = gray_background()
    frame = frame_with_fish([(80, 60)])
    # A min_area above the blob area rejects it; below keeps it.
    assert BlobDetector(bg, min_area=100000.0).detect(frame) == []
    assert len(BlobDetector(bg, min_area=10.0, max_area=100000.0).detect(frame)) == 1


def test_foreground_mask_is_empty_without_fish():
    bg = gray_background()
    empty = np.full_like(bg, BG_VALUE)
    mask = BlobDetector(bg).foreground_mask(empty)
    assert mask.sum() == 0


def test_shape_features_orientation_follows_body_axis():
    # A horizontal bar -> axis ~0 deg, length (x) > width (y).
    gray = gray_background()
    import cv2
    cv2.ellipse(gray, (80, 60), (20, 6), 0, 0, 360, FISH_VALUE, -1)
    contours, _ = cv2.findContours((gray < 128).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    length, width, axis_deg, end_a, end_b = _shape_features(contours[0])
    assert length > width
    assert axis_deg < 15 or axis_deg > 165     # near horizontal (0/180)
