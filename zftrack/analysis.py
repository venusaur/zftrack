"""Post-hoc analysis of a trajectory CSV: per-fish metrics, an occupancy
heatmap and a trajectory map. Rendered with OpenCV (no extra dependencies).

Metrics per fish: time tracked, distance travelled, speed, active vs inactive
time, sleep (bouts/total/longest) and thigmotaxis (time near the wall vs centre).
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import cv2
import numpy as np

from .arena import detect_arena
from .tracker import _PALETTE


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def _load_csv(csv_path: str):
    rows = list(csv.DictReader(open(csv_path)))
    tracks: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        tracks[int(r["id"])].append(r)
    for pts in tracks.values():
        pts.sort(key=lambda r: int(r["frame"]))
    return tracks


def _sleep_bouts(sleeping) -> tuple[int, int]:
    """Return (number of sleep bouts, longest bout length in frames)."""
    bouts = longest = run = 0
    for s in sleeping:
        if s:
            run += 1
            longest = max(longest, run)
            if run == 1:
                bouts += 1
        else:
            run = 0
    return bouts, longest


def detect_phases(brightness, fps: float, min_phase_s: float = 20.0):
    """Segment a recording into light/dark phases from its brightness trace.

    Returns a list of ``(start_frame, end_frame, label)``. A recording filmed
    under constant lighting yields a single ``"constant"`` phase.
    """
    b = np.asarray(brightness, dtype=np.float32)
    n = len(b)
    if n == 0:
        return []
    # Robust spread (ignores single-frame glitches): constant lighting if the
    # bulk of the trace barely varies.
    lo, hi = np.percentile(b, 5), np.percentile(b, 95)
    if (hi - lo) < 0.06 * max(np.median(b), 1.0):
        return [(0, n - 1, "constant")]

    # Smooth over a few seconds so flicker doesn't manufacture phases.
    win = max(1, int(fps * 5))
    kernel = np.ones(win) / win
    bs = np.convolve(b, kernel, mode="same")
    thr = 0.5 * (lo + hi)
    labels = np.where(bs >= thr, "light", "dark")

    phases, start = [], 0
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            phases.append([start, i - 1, labels[i - 1]])
            start = i
    phases.append([start, n - 1, labels[-1]])

    # Absorb phases shorter than min_phase_s into the preceding one.
    min_frames = min_phase_s * fps
    merged: list[list] = []
    for p in phases:
        if merged and (p[1] - p[0] + 1) < min_frames:
            merged[-1][1] = p[1]
        else:
            merged.append(p)
    if len(merged) == 1:
        merged[0][2] = "constant"
    return [(s, e, lab) for s, e, lab in merged]


def _phase_label(frame_idx: int, phases) -> str:
    for s, e, lab in phases:
        if s <= frame_idx <= e:
            return lab
    return phases[-1][2] if phases else "all"


def analyze_tracks(
    csv_path: str,
    fps: float,
    out_prefix: str,
    background: np.ndarray | None = None,
    px_per_mm: float | None = None,
    center_radius_frac: float = 0.5,
    min_track_frames: int = 12,
    wells: list | None = None,
    brightness=None,
    bin_seconds: float = 60.0,
    phases=None,
) -> dict:
    """Compute metrics and render figures. Returns a small summary dict.

    Writes the per-fish ``*_summary.csv``, the ``*_heatmap.png`` /
    ``*_trajectories.png`` figures, and the sleep analytics: a per-bin
    ``*_timeseries.csv``, a sleep ``*_actogram.png``, and per-phase /
    group-level ``*_phases.csv``. Light/dark phases are taken from ``phases`` if
    given, else detected from ``brightness``. In plate mode (``wells`` given),
    thigmotaxis is measured per fish against its own well centre.
    """
    tracks = _load_csv(csv_path)

    all_pos = np.array(
        [(float(r["cx"]), float(r["cy"])) for pts in tracks.values() for r in pts]
    )
    # Per-fish "centre" reference: own well in plate mode, shared arena otherwise.
    plate_mode = bool(wells)
    cx, cy, radius = detect_arena(
        background if background is not None else np.zeros((10, 10), np.uint8),
        all_pos,
    )
    center_r = center_radius_frac * radius

    def conv(px):  # px -> mm if calibrated, else px
        return px / px_per_mm if px_per_mm else px

    unit = "mm" if px_per_mm else "px"
    per_fish: dict[int, dict] = {}
    summary_rows = []
    for tid, pts in sorted(tracks.items()):
        if len(pts) < min_track_frames:
            continue
        xs = np.array([float(r["cx"]) for r in pts])
        ys = np.array([float(r["cy"]) for r in pts])
        frames = np.array([int(r["frame"]) for r in pts])
        sleeping = np.array([int(r["sleeping"]) for r in pts])
        per_fish[tid] = {"frames": frames, "xs": xs, "ys": ys, "sleeping": sleeping}

        # Distance / speed over consecutive frames only.
        consec = np.diff(frames) == 1
        steps = np.hypot(np.diff(xs), np.diff(ys))[consec]
        total_dist = float(steps.sum())
        speeds = steps * fps  # px/s
        mean_speed = float(speeds.mean()) if len(speeds) else 0.0
        max_speed = float(speeds.max()) if len(speeds) else 0.0

        bouts, longest = _sleep_bouts(sleeping)

        # Thigmotaxis: fraction of time inside the central zone. In plate mode
        # each fish is scored against its own well centre and radius.
        if plate_mode:
            wcx, wcy = float(np.median(xs)), float(np.median(ys))
            wr = float(np.percentile(np.hypot(xs - wcx, ys - wcy), 95)) or 1.0
            dist_center = np.hypot(xs - wcx, ys - wcy)
            center_frac = float((dist_center <= center_radius_frac * wr).mean())
        else:
            dist_center = np.hypot(xs - cx, ys - cy)
            center_frac = float((dist_center <= center_r).mean())

        summary_rows.append({
            "id": tid,
            "frames_tracked": len(pts),
            "duration_s": round(len(pts) / fps, 2),
            f"distance_{unit}": round(conv(total_dist), 2),
            f"mean_speed_{unit}_s": round(conv(mean_speed), 2),
            f"max_speed_{unit}_s": round(conv(max_speed), 2),
            "active_frac": round(1.0 - sleeping.mean(), 3),
            "sleep_total_s": round(int(sleeping.sum()) / fps, 2),
            "sleep_bouts": bouts,
            "longest_sleep_s": round(longest / fps, 2),
            "center_frac": round(center_frac, 3),
            "periphery_frac": round(1.0 - center_frac, 3),
        })

    summary_csv = f"{out_prefix}_summary.csv"
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    heatmap_png = f"{out_prefix}_heatmap.png"
    traj_png = f"{out_prefix}_trajectories.png"
    if background is not None and len(all_pos):
        _render_heatmap(background, all_pos, heatmap_png, (cx, cy, radius), wells)
        _render_trajectories(background, tracks, traj_png,
                             (cx, cy, radius, center_r), min_track_frames, wells)

    # ----- sleep analytics: phases, time series, actogram, group stats --------
    total_frames = int(max((f["frames"].max() for f in per_fish.values()), default=0)) + 1
    if brightness is not None and len(brightness):
        total_frames = max(total_frames, len(brightness))
    if phases:
        phase_spans = phases
    elif brightness is not None and len(brightness):
        phase_spans = detect_phases(brightness, fps)
    else:
        phase_spans = [(0, total_frames - 1, "all")]

    timeseries_csv = f"{out_prefix}_timeseries.csv"
    actogram_png = f"{out_prefix}_actogram.png"
    phases_csv = f"{out_prefix}_phases.csv"
    group = {}
    if per_fish:
        _write_timeseries(per_fish, fps, bin_seconds, phase_spans, total_frames,
                          timeseries_csv, conv, unit)
        _render_actogram(per_fish, fps, bin_seconds, phase_spans, total_frames,
                         actogram_png)
        group = _write_phase_stats(per_fish, fps, phase_spans, phases_csv, conv, unit)

    return {
        "summary_csv": summary_csv if summary_rows else None,
        "heatmap_png": heatmap_png if background is not None else None,
        "trajectories_png": traj_png if background is not None else None,
        "timeseries_csv": timeseries_csv if per_fish else None,
        "actogram_png": actogram_png if per_fish else None,
        "phases_csv": phases_csv if per_fish else None,
        "phases": [(s, e, lab) for s, e, lab in phase_spans],
        "group": group,
        "arena": (round(cx, 1), round(cy, 1), round(radius, 1)),
        "n_fish": len(summary_rows),
        "rows": summary_rows,
    }


def _to_bgr(background: np.ndarray) -> np.ndarray:
    return background if background.ndim == 3 else cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)


def _draw_zones(vis, arena, wells, center=True):
    if wells:
        for (wx, wy, wr) in wells:
            cv2.circle(vis, (int(wx), int(wy)), int(wr), (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cx, cy, r = arena[:3]
        cv2.circle(vis, (int(cx), int(cy)), int(r), (255, 255, 255), 1, cv2.LINE_AA)
        if center and len(arena) > 3:
            cv2.circle(vis, (int(cx), int(cy)), int(arena[3]), (180, 180, 180), 1, cv2.LINE_AA)


def _render_heatmap(background, positions, path, arena, wells=None):
    h, w = background.shape[:2]
    acc = np.zeros((h, w), np.float32)
    for x, y in positions:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            acc[yi, xi] += 1.0
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=12)
    if acc.max() > 0:
        acc /= acc.max()
    heat = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = _to_bgr(background)
    mask = (acc > 0.04)[:, :, None]
    blended = np.where(mask, cv2.addWeighted(base, 0.45, heat, 0.55, 0), base)
    _draw_zones(blended, arena, wells, center=False)
    cv2.imwrite(path, blended)


def _render_trajectories(background, tracks, path, arena, min_track_frames, wells=None):
    vis = _to_bgr(background).copy()
    _draw_zones(vis, arena, wells, center=True)
    for tid, pts in tracks.items():
        if len(pts) < min_track_frames:
            continue
        color = _color_for_id(tid)
        poly = np.array([[int(float(r_["cx"])), int(float(r_["cy"]))] for r_ in pts], np.int32)
        cv2.polylines(vis, [poly], False, color, 1, cv2.LINE_AA)
        sx, sy = poly[0]
        cv2.putText(vis, str(tid), (sx + 4, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2, cv2.LINE_AA)
    cv2.imwrite(path, vis)


def _bin_edges(total_frames, bin_frames):
    return list(range(0, total_frames, bin_frames))


def _write_timeseries(per_fish, fps, bin_seconds, phases, total_frames,
                      path, conv, unit):
    """Per-fish, per-time-bin sleep / activity / distance."""
    bin_frames = max(1, int(round(bin_seconds * fps)))
    edges = _bin_edges(total_frames, bin_frames)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "bin", "t_start_s", "phase", "sleep_s",
                         "active_s", f"distance_{unit}", f"mean_speed_{unit}_s"])
        for tid, d in per_fish.items():
            frames, xs, ys, sleeping = d["frames"], d["xs"], d["ys"], d["sleeping"]
            for bi, e0 in enumerate(edges):
                e1 = e0 + bin_frames
                sel = (frames >= e0) & (frames < e1)
                if not sel.any():
                    continue
                bframes, bxs, bys = frames[sel], xs[sel], ys[sel]
                bsleep = sleeping[sel]
                consec = np.diff(bframes) == 1
                steps = np.hypot(np.diff(bxs), np.diff(bys))[consec]
                dist = float(steps.sum())
                n = len(bframes)
                sleep_s = float(bsleep.sum()) / fps
                active_s = (n - float(bsleep.sum())) / fps
                mean_speed = conv(float((steps * fps).mean())) if len(steps) else 0.0
                writer.writerow([
                    tid, bi, round(e0 / fps, 1), _phase_label(e0, phases),
                    round(sleep_s, 2), round(active_s, 2),
                    round(conv(dist), 2), round(mean_speed, 2),
                ])


def _render_actogram(per_fish, fps, bin_seconds, phases, total_frames, path):
    """Sleep raster: one row per fish, columns = time bins, blue = asleep."""
    bin_frames = max(1, int(round(bin_seconds * fps)))
    edges = _bin_edges(total_frames, bin_frames)
    ids = sorted(per_fish)
    n_rows, n_cols = len(ids), len(edges)
    if n_rows == 0 or n_cols == 0:
        return

    sleep_frac = np.full((n_rows, n_cols), np.nan, np.float32)
    for ri, tid in enumerate(ids):
        frames, sleeping = per_fish[tid]["frames"], per_fish[tid]["sleeping"]
        for ci, e0 in enumerate(edges):
            sel = (frames >= e0) & (frames < e0 + bin_frames)
            if sel.any():
                sleep_frac[ri, ci] = float(sleeping[sel].mean())

    cell, pad, top = 22, 70, 26
    H = top + n_rows * cell + 10
    W = pad + n_cols * cell + 10
    img = np.full((H, W, 3), 255, np.uint8)

    # Phase strip along the top (yellow = light, dark gray = dark/constant).
    for ci, e0 in enumerate(edges):
        lab = _phase_label(e0, phases)
        col = (150, 230, 255) if lab == "light" else (90, 90, 90) if lab == "dark" else (210, 210, 210)
        cv2.rectangle(img, (pad + ci * cell, 4), (pad + (ci + 1) * cell, top - 6), col, -1)

    for ri, tid in enumerate(ids):
        y = top + ri * cell
        cv2.putText(img, f"id {tid}", (6, y + cell - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)
        for ci in range(n_cols):
            x = pad + ci * cell
            v = sleep_frac[ri, ci]
            if np.isnan(v):
                col = (235, 235, 235)            # no data
            else:
                # white (awake) -> deep blue (asleep)
                col = (255, int(255 - 175 * v), int(255 - 255 * v))
            cv2.rectangle(img, (x, y), (x + cell - 1, y + cell - 1), col, -1)
            cv2.rectangle(img, (x, y), (x + cell - 1, y + cell - 1), (220, 220, 220), 1)

    label = f"Sleep actogram ({bin_seconds:.0f}s bins) - blue = asleep"
    cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def _write_phase_stats(per_fish, fps, phases, path, conv, unit):
    """Per-fish-per-phase sleep stats plus group mean +/- SEM. Returns group dict.

    A phase *label* may occur in several spans (e.g. day/night/day); stats for a
    label are aggregated across all its spans so each fish contributes one row
    per label.
    """
    labels = []
    for _, _, lab in phases:
        if lab not in labels:
            labels.append(lab)

    rows = []
    for tid, d in per_fish.items():
        frames, xs, ys, sleeping = d["frames"], d["xs"], d["ys"], d["sleeping"]
        for lab in labels:
            n = sleep_f = bouts = longest = 0
            dist = 0.0
            seen = False
            for (s, e, plab) in phases:
                if plab != lab:
                    continue
                sel = (frames >= s) & (frames <= e)
                if not sel.any():
                    continue
                seen = True
                pf, pxs, pys, psl = frames[sel], xs[sel], ys[sel], sleeping[sel]
                consec = np.diff(pf) == 1
                dist += float(np.hypot(np.diff(pxs), np.diff(pys))[consec].sum())
                b, lg = _sleep_bouts(psl)
                bouts += b
                longest = max(longest, lg)
                sleep_f += int(psl.sum())
                n += len(pf)
            if not seen:
                continue
            rows.append({
                "id": tid, "phase": lab,
                "duration_s": round(n / fps, 1),
                "sleep_s": round(sleep_f / fps, 2),
                "sleep_pct": round(100.0 * sleep_f / n, 1) if n else 0.0,
                "sleep_bouts": bouts,
                "longest_sleep_s": round(longest / fps, 2),
                f"distance_{unit}": round(conv(dist), 2),
            })

    if rows:
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Group mean +/- SEM per phase.
    group = {}
    by_phase = defaultdict(list)
    for r in rows:
        by_phase[r["phase"]].append(r)
    for lab, prows in by_phase.items():
        def stat(key):
            vals = np.array([r[key] for r in prows], float)
            sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            return round(float(vals.mean()), 2), round(sem, 2)
        group[lab] = {
            "n_fish": len(prows),
            "sleep_pct": stat("sleep_pct"),
            "sleep_s": stat("sleep_s"),
            "sleep_bouts": stat("sleep_bouts"),
            f"distance_{unit}": stat(f"distance_{unit}"),
        }
    return group
