"""End-to-end pipeline: video in -> annotated video + trajectory CSV out."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .background import build_background
from .detector import BlobDetector
from .tracker import MultiObjectTracker, Track
from .plate import find_wells, PlateTracker


@dataclass
class TrackerConfig:
    """All tunable parameters for one tracking run."""

    # Mode
    mode: str = "arena"  # "arena" (open dish) or "plate" (multi-well sleep assay)
    # Background
    bg_samples: int = 80
    bg_mode: str = "median"  # "median" (arena) or "max" (well-plate / dark-on-bright)
    # Detector
    threshold: int = 13
    min_area: float = 200.0
    max_area: float = 6000.0
    blur: int = 5
    close_ksize: int = 7
    open_ksize: int = 3
    # Tracker
    max_distance: float = 120.0
    max_disappeared: int = 48
    min_hits: int = 3
    merge_aware: bool = False
    merge_area_factor: float = 1.8
    # Re-identification (revive lost IDs instead of spawning new ones)
    reid: bool = True
    reid_frames: int = 48
    reid_max_distance: float = 160.0
    reid_appearance_max: float = 0.4
    # Activity / sleep
    activity_window_s: float = 1.0
    activity_px: float = 30.0
    sleep_seconds: float = 60.0
    # Annotation
    draw_trail: bool = True
    trail_length: int = 40
    draw_bbox: bool = False
    draw_heading: bool = True


_SLEEP_COLOR = (255, 230, 120)  # light cyan/blue for sleeping fish


def _collect_positions(input_path: str, detector: BlobDetector,
                       num_samples: int = 150) -> np.ndarray:
    """Sample frames and return all detected centroids (for well finding)."""
    cap = cv2.VideoCapture(input_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), min(num_samples, max(total, 1))).astype(int)
    points = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            points.extend(d.centroid for d in detector.detect(frame))
    cap.release()
    return np.array(points) if points else np.zeros((0, 2))


def _draw_annotations(
    frame: np.ndarray,
    confirmed: list[Track],
    all_tracks: list[Track],
    config: TrackerConfig,
    frame_idx: int,
    fps: float,
    wells: list | None = None,
) -> np.ndarray:
    """Render outlines, IDs, headings, sleep state, trails and a HUD."""
    vis = frame.copy()

    # Well outlines (plate mode), drawn faintly underneath everything.
    if wells:
        for (wx, wy, wr) in wells:
            cv2.circle(vis, (int(wx), int(wy)), int(wr), (120, 120, 120), 1, cv2.LINE_AA)

    if config.draw_trail:
        for track in confirmed:
            pts = [(int(x), int(y)) for (_, x, y, _) in track.history[-config.trail_length:]]
            for i in range(1, len(pts)):
                cv2.line(vis, pts[i - 1], pts[i], track.color, 1, cv2.LINE_AA)

    n_sleeping = 0
    for track in confirmed:
        color = _SLEEP_COLOR if track.sleeping else track.color
        if track.sleeping:
            n_sleeping += 1
        if track.contour is not None:
            cv2.drawContours(vis, [track.contour], -1, color, 2)
        if config.draw_bbox:
            x, y, w, h = track.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)

        cx, cy = track.centroid
        # Heading arrow from centroid toward the head tip.
        if config.draw_heading and track.head_point is not None and not track.sleeping:
            hx, hy = track.head_point
            cv2.arrowedLine(vis, (int(cx), int(cy)), (int(hx), int(hy)),
                            color, 2, cv2.LINE_AA, tipLength=0.4)

        label = f"{track.id}"
        if track.sleeping:
            label += " Zzz"
        lx, ly = int(cx) + 8, int(cy) - 8
        cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color, 2, cv2.LINE_AA)

    # Coasting tracks: faint marker so the user sees an ID is being maintained.
    for track in all_tracks:
        if track.confirmed and track.time_since_update > 0:
            cx, cy = track.centroid
            cv2.circle(vis, (int(cx), int(cy)), 6, track.color, 1, cv2.LINE_AA)
            cv2.putText(vis, f"{track.id}?", (int(cx) + 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, track.color, 1, cv2.LINE_AA)

    t = frame_idx / fps if fps else 0.0
    hud = (f"frame {frame_idx}  t={t:5.2f}s  fish: {len(confirmed)}"
           f"  sleeping: {n_sleeping}")
    cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def process_video(
    input_path: str,
    output_video: str | None = None,
    output_csv: str | None = None,
    config: TrackerConfig | None = None,
    progress: bool = True,
) -> dict:
    """Track fish through ``input_path``; return a summary dict."""
    config = config or TrackerConfig()

    if progress:
        print("Estimating background...", flush=True)
    background = build_background(input_path, num_samples=config.bg_samples,
                                  mode=config.bg_mode)

    detector = BlobDetector(
        background=background,
        threshold=config.threshold,
        min_area=config.min_area,
        max_area=config.max_area,
        blur=config.blur,
        close_ksize=config.close_ksize,
        open_ksize=config.open_ksize,
    )

    # Plate mode: locate the wells once from where the fish actually appear.
    wells = None
    if config.mode == "plate":
        if progress:
            print("Locating wells...", flush=True)
        positions = _collect_positions(input_path, detector, num_samples=150)
        wells = find_wells(positions, background.shape[:2])
        if progress:
            print(f"  found {len(wells)} occupied wells", flush=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if config.mode == "plate":
        tracker = PlateTracker(
            wells, fps=fps, background=background,
            threshold=max(8, config.threshold - 10),
            min_area=max(4.0, config.min_area * 0.4),
            max_area=max(config.max_area, 1500.0),
            activity_window_s=config.activity_window_s,
            activity_px=config.activity_px,
            sleep_seconds=config.sleep_seconds,
        )
    else:
        tracker = MultiObjectTracker(
            max_distance=config.max_distance,
            max_disappeared=config.max_disappeared,
            min_hits=config.min_hits,
            merge_aware=config.merge_aware,
            merge_area_factor=config.merge_area_factor,
            reid=config.reid,
            reid_frames=config.reid_frames,
            reid_max_distance=config.reid_max_distance,
            reid_appearance_max=config.reid_appearance_max,
            fps=fps,
            activity_window_s=config.activity_window_s,
            activity_px=config.activity_px,
            sleep_seconds=config.sleep_seconds,
        )

    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise IOError(f"Could not open video writer: {output_video}")

    csv_file = csv_writer = None
    if output_csv:
        csv_file = open(output_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["frame", "time_s", "id", "cx", "cy",
             "bbox_x", "bbox_y", "bbox_w", "bbox_h", "area",
             "length", "width", "axis_deg", "heading_deg",
             "speed_px", "inactive_s", "sleeping", "detected"]
        )

    distinct_ids: set[int] = set()
    brightness_trace: list[float] = []
    start = time.time()
    frame_idx = 0
    is_plate = config.mode == "plate"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mean brightness (downsampled) for light/dark phase detection.
        brightness_trace.append(float(frame[::4, ::4].mean()))

        if is_plate:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            confirmed = tracker.update_frame(gray, frame_idx)
        else:
            detections = detector.detect(frame)
            confirmed = tracker.update(detections, frame_idx)
        for trk in confirmed:
            distinct_ids.add(trk.id)

        if csv_writer is not None:
            t_s = frame_idx / fps if fps else 0.0
            for track in confirmed:
                x, y, w, h = track.bbox
                cx, cy = track.centroid
                vx, vy = track.velocity
                heading = "" if math.isnan(track.heading_deg) else f"{track.heading_deg:.1f}"
                csv_writer.writerow(
                    [frame_idx, f"{t_s:.4f}", track.id,
                     f"{cx:.2f}", f"{cy:.2f}", x, y, w, h, f"{track.area:.1f}",
                     f"{track.length:.1f}", f"{track.width:.1f}",
                     f"{track.axis_deg:.1f}", heading,
                     f"{math.hypot(vx, vy):.2f}", f"{track.inactive_seconds:.2f}",
                     int(track.sleeping), 1]
                )

        if writer is not None:
            vis = _draw_annotations(frame, confirmed, tracker.live_tracks(),
                                    config, frame_idx, fps, wells)
            writer.write(vis)

        frame_idx += 1
        if progress and total and frame_idx % 60 == 0:
            print(f"  {frame_idx}/{total} frames ({100.0 * frame_idx / total:4.1f}%)", flush=True)

    cap.release()
    if writer is not None:
        writer.release()
    if csv_file is not None:
        csv_file.close()

    elapsed = time.time() - start
    summary = {
        "frames": frame_idx,
        "distinct_ids": len(distinct_ids),
        "elapsed_s": round(elapsed, 2),
        "fps_processed": round(frame_idx / elapsed, 1) if elapsed else 0.0,
        "output_video": output_video,
        "output_csv": output_csv,
        "fps": fps,
        "background": background,
        "width": width,
        "height": height,
        "wells": wells,
        "brightness": brightness_trace,
    }
    if progress:
        print(
            f"Done: {summary['frames']} frames, {summary['distinct_ids']} IDs, "
            f"{summary['elapsed_s']}s ({summary['fps_processed']} fps)",
            flush=True,
        )
    return summary
