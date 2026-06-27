"""Tests for multi-object tracking: ID stability, re-ID and sleep scoring."""

from __future__ import annotations

import numpy as np

from zftrack.tracker import MultiObjectTracker, Track
from conftest import make_detection


def _run(tracker, frames):
    """Feed a list of per-frame detection lists; return last confirmed tracks."""
    confirmed = []
    for i, dets in enumerate(frames):
        confirmed = tracker.update(dets, i)
    return confirmed


def test_single_fish_keeps_one_stable_id():
    tracker = MultiObjectTracker(min_hits=2)
    frames = [[make_detection(10 + 5 * i, 60)] for i in range(12)]
    _run(tracker, frames)
    ids = {t.id for t in tracker.tracks}
    assert ids == {1}                       # never renumbered
    assert tracker.tracks[0].hits == 12


def test_two_fish_get_distinct_stable_ids():
    tracker = MultiObjectTracker(min_hits=2)
    frames = [[make_detection(10 + 4 * i, 30),
               make_detection(150 - 4 * i, 90)] for i in range(12)]
    confirmed = _run(tracker, frames)
    assert len({t.id for t in confirmed}) == 2
    assert len(tracker.tracks) == 2          # no spurious extra tracks


def test_crossing_fish_do_not_collapse_to_one_id():
    # Two fish approach, stay separated by > a blob width, then part.
    tracker = MultiObjectTracker(min_hits=2)
    frames = []
    for i in range(20):
        a = make_detection(20 + 5 * i, 60)
        b = make_detection(180 - 5 * i, 64)
        frames.append([a, b])
    _run(tracker, frames)
    assert len({t.id for t in tracker.tracks}) == 2


def test_reid_revives_lost_id_instead_of_spawning_new():
    tracker = MultiObjectTracker(
        min_hits=2, max_disappeared=2, reid=True, reid_frames=20,
        reid_max_distance=200.0,
    )
    # Establish a moving fish.
    for i in range(6):
        tracker.update([make_detection(20 + 6 * i, 60)], i)
    established_id = tracker.tracks[0].id
    # Disappear long enough to retire into the lost buffer.
    for i in range(6, 12):
        tracker.update([], i)
    assert tracker.tracks == [] and len(tracker.lost) == 1
    # Reappear near where it was heading -> same ID revived.
    tracker.update([make_detection(20 + 6 * 12, 60)], 12)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].id == established_id


def test_reid_disabled_spawns_new_id():
    tracker = MultiObjectTracker(min_hits=2, max_disappeared=1, reid=False)
    for i in range(4):
        tracker.update([make_detection(20, 60)], i)
    first_id = tracker.tracks[0].id
    for i in range(4, 8):
        tracker.update([], i)
    tracker.update([make_detection(22, 60)], 8)
    assert tracker.tracks[0].id != first_id


def test_stationary_fish_is_scored_sleeping():
    fps = 10.0
    tracker = MultiObjectTracker(
        min_hits=2, fps=fps, sleep_seconds=1.0, activity_window_s=1.0,
        activity_px=20.0,
    )
    # Hold still well past the 1 s (10-frame) sleep threshold.
    for i in range(30):
        tracker.update([make_detection(80, 60)], i)
    track = tracker.tracks[0]
    assert track.sleeping is True
    assert track.inactive_seconds >= 1.0


def test_sleep_survives_single_frame_position_outliers():
    # A resting fish whose detector emits one wild outlier per second must still
    # be scored asleep (median filter + spread metric reject the spikes).
    fps = 10.0
    tracker = MultiObjectTracker(
        min_hits=2, fps=fps, sleep_seconds=1.0, activity_window_s=1.0,
        activity_px=20.0,
    )
    for i in range(40):
        if i % 10 == 5:
            det = make_detection(80 + 200, 60)     # one-frame jump far away
        else:
            det = make_detection(80, 60)
        tracker.update([det], i)
    assert tracker.tracks[0].sleeping is True


def test_spread_metric_is_frame_rate_independent():
    # The same gentle sub-pixel jitter must not read as "active" just because the
    # frame rate (and hence step count per window) is higher.
    for fps in (10.0, 60.0):
        tracker = MultiObjectTracker(
            min_hits=2, fps=fps, sleep_seconds=1.0, activity_window_s=1.0,
            activity_px=20.0,
        )
        n = int(fps * 3)
        for i in range(n):
            jitter = 0.5 * (i % 2)                  # ±0.5 px wobble, fish at rest
            tracker.update([make_detection(80 + jitter, 60)], i)
        assert tracker.tracks[0].sleeping is True, f"failed at {fps} fps"


def test_moving_fish_is_not_sleeping():
    fps = 10.0
    tracker = MultiObjectTracker(
        min_hits=2, fps=fps, sleep_seconds=1.0, activity_window_s=1.0,
        activity_px=20.0,
    )
    for i in range(30):
        tracker.update([make_detection(20 + 8 * i, 60)], i)
    assert tracker.tracks[0].sleeping is False


def test_expected_count_caps_total_ids():
    # Two real fish plus a persistent spurious third detection far from both.
    tracker = MultiObjectTracker(min_hits=2, expected_count=2)
    for i in range(6):
        tracker.update([make_detection(20 + 3 * i, 30),
                        make_detection(150 - 3 * i, 90)], i)
    for i in range(6, 14):
        tracker.update([make_detection(20 + 3 * i, 30),
                        make_detection(150 - 3 * i, 90),
                        make_detection(80, 150)], i)
    assert len({t.id for t in tracker.tracks}) <= 2
    assert tracker._next_id - 1 <= 2          # a third ID was never allocated


def test_no_cap_spawns_extra_id_for_third_detection():
    tracker = MultiObjectTracker(min_hits=2)   # expected_count=None (default)
    for i in range(6):
        tracker.update([make_detection(20 + 3 * i, 30),
                        make_detection(150 - 3 * i, 90)], i)
    for i in range(6, 14):
        tracker.update([make_detection(20 + 3 * i, 30),
                        make_detection(150 - 3 * i, 90),
                        make_detection(80, 150)], i)
    assert tracker._next_id - 1 >= 3          # the third fish got its own ID


def test_rescue_keeps_id_when_fish_reappears_beyond_max_distance():
    # A fish coasts, then reappears between max_distance and reid_max_distance.
    tracker = MultiObjectTracker(min_hits=2, max_distance=60.0,
                                 reid_max_distance=200.0)
    for i in range(5):
        tracker.update([make_detection(20 + 5 * i, 60)], i)
    established = tracker.tracks[0].id
    tracker.update([], 5)                       # miss one frame (coasts)
    # Reappear ~90 px away: beyond max_distance(60), within reid_max_distance.
    tracker.update([make_detection(135, 60)], 6)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].id == established


def test_appearance_cost_zero_for_identical_none_safe():
    d = np.array([400.0, 24.0, 12.0, 40.0], np.float32)
    assert MultiObjectTracker._appearance_cost(d, d) == 0.0
    assert MultiObjectTracker._appearance_cost(None, d) == 0.0
    far = np.array([4000.0, 2.0, 1.0, 200.0], np.float32)
    assert MultiObjectTracker._appearance_cost(d, far) > 0.3


def test_track_confirmation_requires_min_hits():
    tracker = MultiObjectTracker(min_hits=3)
    tracker.update([make_detection(40, 40)], 0)
    # First frames: confirmed early because frame_count <= min_hits (warm-up).
    for i in range(1, 5):
        tracker.update([make_detection(40 + i, 40)], i)
    assert tracker.tracks[0].confirmed is True
    assert tracker.tracks[0].hits >= 3
