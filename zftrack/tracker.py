"""Multi-object tracking with per-track Kalman prediction, Hungarian (global,
optimal) data association, lightweight re-identification, body heading and
online sleep/activity state.

Pipeline per frame:

1. predict every track (constant-velocity Kalman);
2. match detections to predicted positions by minimising total assignment
   distance, gated by ``max_distance``;
3. matched tracks update position, body heading and appearance;
4. unmatched tracks either park on a merged blob (a neighbour they overlapped)
   or coast on prediction toward retirement;
5. unmatched detections first try to *revive a recently-lost track* (re-ID, so a
   fish that reappears keeps its old number) and otherwise start a new track;
6. tracks unseen past ``max_disappeared`` are retired into a lost buffer kept for
   re-ID; tracks lost past ``max_disappeared + reid_frames`` are discarded.

Every confirmed track also maintains how long it has been continuously inactive,
from which a sleep flag is derived (inactivity longer than ``sleep_frames``).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from .detector import Detection


# A palette of visually distinct BGR colors, cycled by track id.
_PALETTE = [
    (66, 66, 244),    # red
    (244, 194, 66),   # blue
    (66, 244, 89),    # green
    (244, 66, 212),   # magenta
    (66, 232, 244),   # yellow
    (244, 130, 66),   # orange
    (180, 66, 244),   # purple
    (66, 244, 200),   # teal
    (200, 200, 200),  # gray
    (120, 180, 80),   # olive
]


def _new_kalman(cx: float, cy: float) -> cv2.KalmanFilter:
    """Constant-velocity Kalman filter initialised at (cx, cy)."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
    return kf


class Track:
    """A single tracked fish across time."""

    HEAD_SPEED_THRESH = 1.5  # px/frame; above this, heading follows motion

    def __init__(self, track_id: int, detection: Detection, frame_idx: int,
                 activity_window: int = 24) -> None:
        self.id = track_id
        self.kf = _new_kalman(*detection.centroid)
        self.centroid = detection.centroid
        self.last_seen = detection.centroid
        self.bbox = detection.bbox
        self.area = detection.area
        self.contour = detection.contour

        # Body shape / heading.
        self.length = detection.length
        self.width = detection.width
        self.axis_deg = detection.axis_deg
        self.head_point: tuple[float, float] | None = None
        self.tail_point: tuple[float, float] | None = None
        self.heading_deg = float("nan")
        self.descriptor = (
            detection.descriptor.copy() if detection.descriptor is not None else None
        )

        # Lifecycle.
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.parked = 0
        self.confirmed = False
        self.color = _PALETTE[track_id % len(_PALETTE)]
        self.history: list[tuple[int, float, float, bool]] = [
            (frame_idx, detection.centroid[0], detection.centroid[1], True)
        ]

        # Activity / sleep.
        self._act_window = activity_window
        self._act_positions: deque[tuple[int, float, float]] = deque()
        self.activity_path = 0.0
        self.is_active = True
        self.inactive_frames = 0
        self.inactive_seconds = 0.0
        self.sleeping = False

        # Lost-buffer bookkeeping (set on retirement).
        self.lost_frame = -1
        self.lost_velocity = (0.0, 0.0)

        self._resolve_heading(detection)

    # ----- geometry / prediction -------------------------------------------

    @property
    def predicted_centroid(self) -> tuple[float, float]:
        state = self.kf.statePre if self.kf.statePre is not None else self.kf.statePost
        return float(state[0, 0]), float(state[1, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])

    def predict(self) -> tuple[float, float]:
        state = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return float(state[0, 0]), float(state[1, 0])

    def _resolve_heading(self, detection: Detection) -> None:
        """Decide which body tip is the head and set the heading angle.

        Uses direction of motion when the fish is moving (fish swim head-first),
        falls back to continuity with the previous head, and finally to the
        darker tip (the eyes are the darkest part of a larva).
        """
        cx, cy = detection.centroid
        a, b = detection.end_a, detection.end_b
        vx, vy = self.velocity
        speed = math.hypot(vx, vy)

        if speed > self.HEAD_SPEED_THRESH:
            da = (a[0] - cx) * vx + (a[1] - cy) * vy
            db = (b[0] - cx) * vx + (b[1] - cy) * vy
            head, tail = (a, b) if da >= db else (b, a)
        elif self.head_point is not None:
            da = math.hypot(a[0] - self.head_point[0], a[1] - self.head_point[1])
            db = math.hypot(b[0] - self.head_point[0], b[1] - self.head_point[1])
            head, tail = (a, b) if da <= db else (b, a)
        else:
            head, tail = (a, b) if detection.end_a_intensity <= detection.end_b_intensity else (b, a)

        self.head_point = head
        self.tail_point = tail
        self.heading_deg = math.degrees(math.atan2(head[1] - cy, head[0] - cx)) % 360.0

    # ----- updates ----------------------------------------------------------

    def update(self, detection: Detection, frame_idx: int) -> None:
        """Correct the filter with a matched detection."""
        measurement = np.array(
            [[np.float32(detection.centroid[0])],
             [np.float32(detection.centroid[1])]]
        )
        self.kf.correct(measurement)
        self.centroid = detection.centroid
        self.last_seen = detection.centroid
        self.bbox = detection.bbox
        self.area = detection.area
        self.contour = detection.contour
        self.length = detection.length
        self.width = detection.width
        self.axis_deg = detection.axis_deg
        self._resolve_heading(detection)
        if detection.descriptor is not None:
            if self.descriptor is None:
                self.descriptor = detection.descriptor.copy()
            else:
                self.descriptor = 0.8 * self.descriptor + 0.2 * detection.descriptor
        self.hits += 1
        self.time_since_update = 0
        self.parked = 0
        self.history.append(
            (frame_idx, detection.centroid[0], detection.centroid[1], True)
        )

    def revive(self, detection: Detection, frame_idx: int) -> None:
        """Re-acquire a lost fish with a new detection, keeping the same ID."""
        self.kf = _new_kalman(*detection.centroid)
        self.time_since_update = 0
        self.parked = 0
        self.update(detection, frame_idx)

    def mark_missed(self, frame_idx: int) -> None:
        cx, cy = self.predicted_centroid
        self.centroid = (cx, cy)
        self.history.append((frame_idx, cx, cy, False))

    def mark_merged(self, blob: Detection, frame_idx: int) -> None:
        """Park this track on a blob it has merged into with another fish."""
        sx, sy = self.last_seen
        self.kf.statePost = np.array([[sx], [sy], [0], [0]], dtype=np.float32)
        self.centroid = (sx, sy)
        self.contour = blob.contour
        self.bbox = blob.bbox
        self.time_since_update = 0
        self.parked += 1
        self.history.append((frame_idx, sx, sy, False))

    def update_activity(self, frame_idx: int, fps: float,
                        activity_px: float, sleep_frames: int) -> None:
        """Update continuous-inactivity state and the sleep flag.

        A fish is inactive on this frame if the path length it travelled over the
        trailing activity window is below ``activity_px``. Continuous inactivity
        longer than ``sleep_frames`` is scored as sleeping.
        """
        self._act_positions.append((frame_idx, self.centroid[0], self.centroid[1]))
        while self._act_positions and self._act_positions[0][0] <= frame_idx - self._act_window:
            self._act_positions.popleft()

        pts = list(self._act_positions)
        path = 0.0
        for i in range(1, len(pts)):
            if pts[i][0] == pts[i - 1][0] + 1:
                path += math.hypot(pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2])
        self.activity_path = path

        span = pts[-1][0] - pts[0][0] if pts else 0
        have_window = span >= self._act_window * 0.5
        inactive = have_window and path < activity_px
        self.inactive_frames = self.inactive_frames + 1 if inactive else 0
        self.inactive_seconds = self.inactive_frames / fps if fps else 0.0
        self.sleeping = self.inactive_frames >= sleep_frames


class MultiObjectTracker:
    """Associates detections to tracks over time, maintaining stable IDs."""

    def __init__(
        self,
        max_distance: float = 120.0,
        max_disappeared: int = 48,
        min_hits: int = 3,
        merge_aware: bool = False,
        merge_area_factor: float = 1.8,
        reid: bool = True,
        reid_frames: int = 48,
        reid_max_distance: float = 160.0,
        reid_appearance_max: float = 0.4,
        fps: float = 24.0,
        activity_window_s: float = 1.0,
        activity_px: float = 30.0,
        sleep_seconds: float = 60.0,
    ) -> None:
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.merge_aware = merge_aware
        self.merge_area_factor = merge_area_factor

        self.reid = reid
        self.reid_frames = reid_frames
        self.reid_max_distance = reid_max_distance
        self.reid_appearance_max = reid_appearance_max

        self.fps = fps
        self._act_window = max(1, int(round(activity_window_s * fps)))
        self.activity_px = activity_px
        self.sleep_frames = max(1, int(round(sleep_seconds * fps)))

        self.tracks: list[Track] = []
        self.lost: list[Track] = []
        self._next_id = 1
        self._frame_count = 0
        self._area_est: float | None = None

    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        """Process one frame of detections; return the confirmed tracks."""
        self._frame_count += 1

        for track in self.tracks:
            track.predict()

        matches, unmatched_tracks, unmatched_dets = self._associate(detections)

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], frame_idx)
        self._update_area_estimate(matches, detections)

        matched_dets = {det_idx for _, det_idx in matches}
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            blob = None
            if self.merge_aware and track.parked < self.max_disappeared:
                blob = self._merged_blob_for(track, detections, matched_dets)
            if blob is not None:
                track.mark_merged(blob, frame_idx)
            else:
                track.mark_missed(frame_idx)

        # New detections: revive a lost track if one fits, else start fresh.
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            revived = self._try_reid(det, frame_idx) if self.reid else None
            if revived is not None:
                revived.revive(det, frame_idx)
                self.tracks.append(revived)
            else:
                self.tracks.append(
                    Track(self._next_id, det, frame_idx, activity_window=self._act_window)
                )
                self._next_id += 1

        self._confirm_and_retire(frame_idx)

        # Update activity/sleep for every live track, then report the visible ones.
        for track in self.tracks:
            track.update_activity(frame_idx, self.fps, self.activity_px, self.sleep_frames)

        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]

    # ----- association helpers ---------------------------------------------

    def _associate(self, detections: list[Detection]):
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        track_points = np.array([t.predicted_centroid for t in self.tracks])
        det_points = np.array([d.centroid for d in detections])
        cost = np.linalg.norm(
            track_points[:, None, :] - det_points[None, :, :], axis=2
        )
        big = self.max_distance * 10.0
        gated = np.where(cost > self.max_distance, big, cost)
        row_idx, col_idx = linear_sum_assignment(gated)

        matches, matched_t, matched_d = [], set(), set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] > self.max_distance:
                continue
            matches.append((r, c))
            matched_t.add(r)
            matched_d.add(c)
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_t]
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_d]
        return matches, unmatched_tracks, unmatched_dets

    def _update_area_estimate(self, matches, detections: list[Detection]) -> None:
        for _, det_idx in matches:
            area = detections[det_idx].area
            if self._area_est is None:
                self._area_est = area
            elif area < self.merge_area_factor * self._area_est:
                self._area_est = 0.9 * self._area_est + 0.1 * area

    def _merged_blob_for(self, track: Track, detections, matched_dets):
        if self._area_est is None:
            return None
        merge_threshold = self.merge_area_factor * self._area_est
        px, py = track.predicted_centroid
        best, best_dist = None, self.max_distance
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets or det.area < merge_threshold:
                continue
            dist = math.hypot(det.centroid[0] - px, det.centroid[1] - py)
            if dist < best_dist:
                best_dist, best = dist, det
        return best

    @staticmethod
    def _appearance_cost(a: np.ndarray | None, b: np.ndarray | None) -> float:
        """Mean relative difference between two appearance descriptors (0=same)."""
        if a is None or b is None:
            return 0.0
        denom = np.abs(a) + np.abs(b) + 1e-6
        return float(np.mean(np.abs(a - b) / denom))

    def _try_reid(self, det: Detection, frame_idx: int):
        """Return a lost track that best matches ``det``, or None."""
        best, best_cost = None, float("inf")
        for track in self.lost:
            dt = frame_idx - track.lost_frame
            ex = track.last_seen[0] + track.lost_velocity[0] * dt
            ey = track.last_seen[1] + track.lost_velocity[1] * dt
            dist = math.hypot(det.centroid[0] - ex, det.centroid[1] - ey)
            if dist > self.reid_max_distance:
                continue
            appear = self._appearance_cost(track.descriptor, det.descriptor)
            if appear > self.reid_appearance_max:
                continue
            cost = dist + 200.0 * appear
            if cost < best_cost:
                best_cost, best = cost, track
        if best is not None:
            self.lost.remove(best)
        return best

    def live_tracks(self) -> list[Track]:
        """All currently-active tracks (for drawing, incl. coasting ones)."""
        return self.tracks

    def _confirm_and_retire(self, frame_idx: int) -> None:
        for track in self.tracks:
            if not track.confirmed and (
                track.hits >= self.min_hits or self._frame_count <= self.min_hits
            ):
                track.confirmed = True

        keep = []
        for track in self.tracks:
            if track.time_since_update <= self.max_disappeared:
                keep.append(track)
            elif self.reid and track.confirmed:
                track.lost_frame = frame_idx
                track.lost_velocity = track.velocity
                self.lost.append(track)
        self.tracks = keep
        self.lost = [t for t in self.lost
                     if frame_idx - t.lost_frame <= self.reid_frames]
