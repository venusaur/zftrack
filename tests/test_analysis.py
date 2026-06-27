"""Tests for the analysis layer: sleep bouts, phase detection, end-to-end CSV."""

from __future__ import annotations

import csv

import numpy as np

from zftrack.analysis import _sleep_bouts, detect_phases, analyze_tracks
from conftest import gray_background


def test_sleep_bouts_counts_runs_and_longest():
    assert _sleep_bouts([0, 0, 0]) == (0, 0)
    assert _sleep_bouts([1, 1, 1]) == (1, 3)
    assert _sleep_bouts([1, 0, 1, 1, 0, 1]) == (3, 2)


def test_detect_phases_constant_lighting_single_phase():
    b = np.full(300, 128.0)
    phases = detect_phases(b, fps=10.0)
    assert len(phases) == 1
    assert phases[0][2] == "constant"


def test_detect_phases_light_then_dark():
    fps = 10.0
    b = np.concatenate([np.full(300, 200.0), np.full(300, 60.0)])
    phases = detect_phases(b, fps=fps)
    labels = [p[2] for p in phases]
    assert "light" in labels and "dark" in labels
    # First span is the bright half.
    assert phases[0][2] == "light"


def _write_csv(path, rows):
    cols = ["frame", "time_s", "id", "cx", "cy", "bbox_x", "bbox_y", "bbox_w",
            "bbox_h", "area", "length", "width", "axis_deg", "heading_deg",
            "speed_px", "inactive_s", "sleeping", "detected"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_analyze_tracks_writes_summary(tmp_path):
    # Fish 1 moves 1 px/frame (awake); fish 2 sits still and is marked sleeping.
    rows = []
    for fr in range(60):
        rows.append(dict(frame=fr, time_s=fr / 10.0, id=1, cx=10 + fr, cy=50,
                         bbox_x=10 + fr, bbox_y=48, bbox_w=4, bbox_h=4, area=400,
                         length=24, width=12, axis_deg=0, heading_deg=0,
                         speed_px=10, inactive_s=0, sleeping=0, detected=1))
        rows.append(dict(frame=fr, time_s=fr / 10.0, id=2, cx=120, cy=80,
                         bbox_x=118, bbox_y=78, bbox_w=4, bbox_h=4, area=400,
                         length=24, width=12, axis_deg=0, heading_deg=0,
                         speed_px=0, inactive_s=fr / 10.0, sleeping=1, detected=1))
    csv_path = str(tmp_path / "tracks.csv")
    _write_csv(csv_path, rows)

    result = analyze_tracks(
        csv_path=csv_path, fps=10.0, out_prefix=str(tmp_path / "tracks"),
        background=gray_background(120, 160), bin_seconds=2.0,
    )
    assert result["n_fish"] == 2
    by_id = {r["id"]: r for r in result["rows"]}
    # Fish 1 travelled ~59 px; fish 2 essentially zero.
    assert by_id[1]["distance_px"] > 50
    assert by_id[2]["distance_px"] < 1
    # Fish 2 is fully asleep; fish 1 fully awake.
    assert by_id[2]["sleep_total_s"] > 5.0
    assert by_id[1]["sleep_total_s"] == 0.0
