"""Well-plate sleep-assay mode: anchor tracking to wells.

A multi-well plate holds one larva per well and the fish never cross between
wells, so the generic tracker's job collapses to: find the wells once, then keep
exactly one persistent track per well. This caps the ID count at the number of
wells, gives every fish a stable number for the whole recording, and ignores
spurious detections that fall outside any well.

Wells are located from the data itself — the cloud of fish detections over the
recording clusters into the occupied wells — which sidesteps the bright menisci
and rims that defeat circle-fitting on the raw background.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

from .detector import Detection, _shape_features, _patch_intensity
from .tracker import Track


def _nearest_neighbor_pitch(points: np.ndarray) -> float:
    """Median nearest-neighbour distance among a set of points."""
    if len(points) < 2:
        return 0.0
    dists = []
    for i, p in enumerate(points):
        d = np.hypot(points[:, 0] - p[0], points[:, 1] - p[1])
        d[i] = np.inf
        dists.append(d.min())
    return float(np.median(dists))


def find_wells(positions: np.ndarray, shape: tuple[int, int],
               blur_sigma: float | None = None) -> list[tuple[float, float, float]]:
    """Locate occupied wells as ``(cx, cy, radius)`` from detection positions.

    Each occupied well is a hill in the smoothed occupancy map of all detections.
    We pick one peak per well via non-maximum suppression at the well pitch (the
    pitch is estimated from the data so it adapts to different plate formats).
    Scattered noise never forms a hill and is ignored.
    """
    h, w = shape
    if len(positions) == 0:
        return []

    acc = np.zeros((h, w), np.float32)
    for x, y in positions:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            acc[yi, xi] += 1.0
    if acc.max() == 0:
        return []

    sigma = blur_sigma if blur_sigma else max(15.0, min(h, w) / 25.0)
    sm = cv2.GaussianBlur(acc, (0, 0), sigmaX=sigma)
    sm /= sm.max()

    # First pass: rough peaks to estimate the well pitch.
    rough = np.argwhere((sm == maximum_filter(sm, size=int(sigma * 2))) & (sm > 0.05))
    rough_xy = np.array([[x, y] for y, x in rough], dtype=np.float32)
    pitch = _nearest_neighbor_pitch(rough_xy) if len(rough_xy) > 1 else sigma * 4
    pitch = max(pitch, sigma * 2)

    # Second pass: one peak per well, suppressed within ~the well pitch.
    win = max(3, int(pitch))
    cand = np.argwhere((sm == maximum_filter(sm, size=win)) & (sm > 0.05))
    cand = sorted(cand, key=lambda p: -sm[p[0], p[1]])
    kept: list[tuple[int, int]] = []
    min_sep2 = (0.7 * pitch) ** 2
    for y, x in cand:
        if all((x - kx) ** 2 + (y - ky) ** 2 > min_sep2 for kx, ky in kept):
            kept.append((x, y))

    radius = 0.5 * pitch
    wells = [(float(x), float(y), float(radius)) for x, y in kept]
    # Number wells in reading order (top-to-bottom, left-to-right) for stable IDs.
    row_h = max(pitch * 0.6, 1.0)
    wells.sort(key=lambda c: (round(c[1] / row_h), c[0]))
    return wells


def detect_in_well(gray: np.ndarray, background: np.ndarray,
                   well: tuple[float, float, float], threshold: int,
                   min_area: float, max_area: float,
                   inner: float = 1.0) -> Detection | None:
    """Find the fish inside one well as the darkest compact blob.

    Because each well holds exactly one larva and the well floor is bright,
    searching this well's interior reliably finds the fish whether it is darting
    (motion-blurred) or motionless — the key to high per-frame coverage and to
    keeping sleeping fish visible. The full well can be searched safely: against
    a *max* background the dark rim cancels (it is dark in every frame), so it
    never registers as foreground, even for wall-hugging fish.
    """
    h, w = gray.shape
    wx, wy, wr = well
    R = int(math.ceil(wr))
    x0, y0 = max(0, int(wx) - R), max(0, int(wy) - R)
    x1, y1 = min(w, int(wx) + R + 1), min(h, int(wy) + R + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = gray[y0:y1, x0:x1]
    bgc = background[y0:y1, x0:x1]
    diff = cv2.subtract(bgc, crop)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)

    mask = np.zeros(diff.shape, np.uint8)
    cv2.circle(mask, (int(round(wx)) - x0, int(round(wy)) - y0),
               int(wr * inner), 255, -1)
    diff = cv2.bitwise_and(diff, diff, mask=mask)

    _, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea) + np.array([[[x0, y0]]], dtype=np.int32)

    area = cv2.contourArea(contour)
    bx, by, bw, bh = cv2.boundingRect(contour)
    m = cv2.moments(contour)
    if m["m00"] == 0:
        cx, cy = bx + bw / 2.0, by + bh / 2.0
    else:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]

    length, width, axis_deg, end_a, end_b = _shape_features(contour)
    return Detection(
        centroid=(cx, cy), bbox=(bx, by, bw, bh), area=area, contour=contour,
        length=length, width=width, axis_deg=axis_deg, end_a=end_a, end_b=end_b,
        end_a_intensity=_patch_intensity(gray, *end_a),
        end_b_intensity=_patch_intensity(gray, *end_b),
        descriptor=np.array([area, length, width,
                             _patch_intensity(gray, cx, cy, r=4)], dtype=np.float32),
    )


class PlateTracker:
    """Maintains one persistent track per well; never retires a well.

    Detection is done per well (see :func:`detect_in_well`), so every fish is
    located in its own well each frame regardless of motion.
    """

    def __init__(self, wells: list[tuple[float, float, float]], fps: float,
                 background: np.ndarray, threshold: int = 18,
                 min_area: float = 6.0, max_area: float = 1500.0,
                 activity_window_s: float = 1.0, activity_px: float = 15.0,
                 sleep_seconds: float = 60.0) -> None:
        self.wells = wells
        self.fps = fps
        self.background = background
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self._act_window = max(1, int(round(activity_window_s * fps)))
        self.activity_px = activity_px
        self.sleep_frames = max(1, int(round(sleep_seconds * fps)))
        self.tracks: list[Track | None] = [None] * len(wells)

    def update_frame(self, gray: np.ndarray, frame_idx: int) -> list[Track]:
        visible = []
        for wi, well in enumerate(self.wells):
            det = detect_in_well(gray, self.background, well, self.threshold,
                                 self.min_area, self.max_area)
            track = self.tracks[wi]
            if det is not None:
                if track is None:
                    track = Track(wi + 1, det, frame_idx, activity_window=self._act_window)
                    track.confirmed = True
                    self.tracks[wi] = track
                else:
                    track.predict()
                    track.update(det, frame_idx)
                visible.append(track)
            elif track is not None:
                # Undetected = resting larva (too still to segment), not a fish
                # that moved away: hold position so sleep can accrue.
                track.hold(frame_idx)

            if self.tracks[wi] is not None:
                self.tracks[wi].update_activity(
                    frame_idx, self.fps, self.activity_px, self.sleep_frames)
        return visible

    def live_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t is not None]
