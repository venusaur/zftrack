"""Tests for well-plate mode: pitch estimation, well finding, in-well detection."""

from __future__ import annotations

import numpy as np
import cv2

from zftrack.plate import (
    _nearest_neighbor_pitch, find_wells, detect_in_well, PlateTracker,
)
from conftest import gray_background, FISH_VALUE, BG_VALUE


def test_nearest_neighbor_pitch_recovers_grid_spacing():
    xs, ys = np.meshgrid(np.arange(0, 200, 40), np.arange(0, 200, 40))
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    assert abs(_nearest_neighbor_pitch(pts) - 40.0) < 1e-6


def test_find_wells_locates_grid_of_occupied_wells():
    # 3x3 grid of wells, pitch 45; scatter detections around each centre.
    rng = np.random.default_rng(0)
    centers = [(40 + 45 * c, 40 + 45 * r) for r in range(3) for c in range(3)]
    pts = []
    for (cx, cy) in centers:
        pts.extend(rng.normal([cx, cy], 3.0, size=(40, 2)))
    wells = find_wells(np.array(pts), shape=(170, 170))
    assert len(wells) == 9
    # Every true centre is matched by some detected well.
    for (cx, cy) in centers:
        assert any(np.hypot(wx - cx, wy - cy) < 12 for wx, wy, _ in wells)


def test_find_wells_numbers_in_reading_order():
    centers = [(40 + 45 * c, 40 + 45 * r) for r in range(3) for c in range(3)]
    rng = np.random.default_rng(1)
    pts = []
    for (cx, cy) in centers:
        pts.extend(rng.normal([cx, cy], 2.0, size=(40, 2)))
    wells = find_wells(np.array(pts), shape=(170, 170))
    # Reading order: y ascends in blocks of 3, x ascends within a row.
    ys = [w[1] for w in wells]
    assert ys[0] <= ys[3] <= ys[6]           # row tops increase
    assert wells[0][0] < wells[2][0]         # first row left->right


def test_find_wells_empty_input():
    assert find_wells(np.zeros((0, 2)), shape=(100, 100)) == []


def test_detect_in_well_finds_dark_fish():
    bg = gray_background(80, 80)                       # bright (max) background
    gray = bg.copy()
    cv2.ellipse(gray, (40, 40), (8, 4), 0, 0, 360, FISH_VALUE, -1)
    well = (40.0, 40.0, 25.0)
    det = detect_in_well(gray, bg, well, threshold=15, min_area=8.0, max_area=600.0)
    assert det is not None
    assert np.hypot(det.centroid[0] - 40, det.centroid[1] - 40) < 4


def test_detect_in_well_empty_when_no_fish():
    bg = gray_background(80, 80)
    well = (40.0, 40.0, 25.0)
    det = detect_in_well(bg, bg, well, threshold=15, min_area=8.0, max_area=600.0)
    assert det is None


def test_plate_resting_fish_holds_position_and_sleeps():
    # A fish that darts, then goes undetected (too still to segment) must hold its
    # last position — not coast on its dart velocity — so it accrues sleep.
    well = (40.0, 40.0, 25.0)
    bg = gray_background(80, 80)
    tracker = PlateTracker([well], fps=10.0, background=bg, threshold=15,
                           min_area=8.0, max_area=600.0,
                           sleep_seconds=1.0, activity_px=10.0)
    # A few moving detections to give the Kalman a non-zero velocity.
    for i, x in enumerate([30, 33, 36, 39]):
        gray = bg.copy()
        cv2.ellipse(gray, (x, 40), (6, 4), 0, 0, 360, FISH_VALUE, -1)
        tracker.update_frame(gray, i)
    last = tracker.live_tracks()[0].centroid
    # Now no fish is segmented for a while (resting / faint).
    for i in range(4, 40):
        tracker.update_frame(bg.copy(), i)
    track = tracker.live_tracks()[0]
    assert track.centroid == last        # held, did not drift with velocity
    assert track.sleeping is True


def test_plate_tracker_assigns_well_numbers_as_ids():
    wells = [(25.0, 25.0, 20.0), (75.0, 25.0, 20.0)]
    bg = gray_background(60, 110)
    tracker = PlateTracker(wells, fps=10.0, background=bg,
                           threshold=15, min_area=8.0, max_area=600.0)
    for i in range(5):
        gray = bg.copy()
        cv2.ellipse(gray, (25, 25), (7, 4), 0, 0, 360, FISH_VALUE, -1)
        cv2.ellipse(gray, (75, 25), (7, 4), 0, 0, 360, FISH_VALUE, -1)
        tracker.update_frame(gray, i)
    ids = sorted(t.id for t in tracker.live_tracks())
    assert ids == [1, 2]                      # one persistent track per well
