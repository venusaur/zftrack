#!/usr/bin/env python3
"""Command-line entry point for the zebrafish tracker.

Examples
--------
Annotated video + CSV + analysis next to the input::

    python track.py path/to/video.mp4

Demonstrate sleep labelling with a short threshold (the sample clip is only
~32 s, so the scientific 60 s default can never trigger)::

    python track.py video.mp4 --sleep-seconds 5
"""

from __future__ import annotations

import argparse
import os
import sys

from zftrack import process_video, TrackerConfig, analyze_tracks


# Detection presets. "arena" = one open dish, large fish, median background.
# "plate" = multi-well sleep assay: small larvae, dark-on-bright, max background
# (robust to stationary/sleeping fish), tighter blob sizes.
PRESETS = {
    "arena": dict(bg_mode="median", threshold=13, min_area=200.0,
                  max_area=6000.0, blur=5, activity_px=30.0),
    "plate": dict(bg_mode="max", threshold=25, min_area=12.0,
                  max_area=600.0, blur=3, activity_px=15.0),
}


def _default_output(input_path: str, suffix: str, ext: str) -> str:
    base, _ = os.path.splitext(input_path)
    return f"{base}{suffix}{ext}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Track zebrafish in a top-down arena video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Input video file")
    p.add_argument("-o", "--output", default=None,
                   help="Annotated output video (default: <input>_tracked.mp4)")
    p.add_argument("-c", "--csv", default=None,
                   help="Trajectory CSV (default: <input>_tracks.csv)")
    p.add_argument("--no-video", action="store_true", help="Skip the annotated video")
    p.add_argument("--no-csv", action="store_true", help="Skip the trajectory CSV")
    p.add_argument("--no-analyze", action="store_true",
                   help="Skip the analysis layer (summary + heatmaps)")
    p.add_argument("--mode", choices=("arena", "plate"), default="arena",
                   help="Detection preset: 'arena' (open dish) or 'plate' "
                        "(multi-well sleep assay, robust to stationary fish)")

    g = p.add_argument_group("detection (defaults follow --mode)")
    g.add_argument("--bg-samples", type=int, default=80,
                   help="Frames sampled to build the background")
    g.add_argument("--bg-mode", choices=("median", "max"), default=None,
                   help="Background model (default: per --mode)")
    g.add_argument("--threshold", type=int, default=None,
                   help="Darker-than-background threshold (0-255)")
    g.add_argument("--min-area", type=float, default=None, help="Min blob area (px)")
    g.add_argument("--max-area", type=float, default=None, help="Max blob area (px)")
    g.add_argument("--blur", type=int, default=None, help="Difference-image blur (odd)")

    g = p.add_argument_group("tracking")
    g.add_argument("--max-distance", type=float, default=120.0,
                   help="Max px a fish may move between frames to keep its ID")
    g.add_argument("--max-disappeared", type=int, default=48,
                   help="Frames a fish may be unseen before retirement to re-ID buffer")
    g.add_argument("--min-hits", type=int, default=3,
                   help="Detections before a track is confirmed (noise filter)")
    g.add_argument("--no-reid", action="store_true",
                   help="Disable reviving lost IDs (reduces ID switches when on)")
    g.add_argument("--no-merge-aware", action="store_true",
                   help="Disable parking IDs on merged blobs when fish overlap")
    g.add_argument("--merge-area-factor", type=float, default=1.8,
                   help="A blob this many times the single-fish area is a merge")

    g = p.add_argument_group("sleep / activity")
    g.add_argument("--sleep-seconds", type=float, default=60.0,
                   help="Continuous inactivity (s) scored as sleeping")
    g.add_argument("--activity-px", type=float, default=None,
                   help="Path length (px) per window below which a fish is inactive")
    g.add_argument("--activity-window", type=float, default=1.0,
                   help="Activity window length in seconds")

    g = p.add_argument_group("analysis")
    g.add_argument("--px-per-mm", type=float, default=None,
                   help="Pixels per mm for real-world units (default: pixel units)")
    g.add_argument("--center-frac", type=float, default=0.5,
                   help="Central-zone radius as a fraction of arena radius")

    g = p.add_argument_group("annotation")
    g.add_argument("--no-trail", action="store_true", help="Do not draw motion trails")
    g.add_argument("--trail-length", type=int, default=40, help="Trail length (points)")
    g.add_argument("--no-heading", action="store_true", help="Do not draw heading arrows")
    g.add_argument("--bbox", action="store_true", help="Also draw bounding boxes")
    g.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1

    output_video = None if args.no_video else (
        args.output or _default_output(args.input, "_tracked", ".mp4"))
    output_csv = None if args.no_csv else (
        args.csv or _default_output(args.input, "_tracks", ".csv"))
    if output_video is None and output_csv is None:
        print("error: nothing to do (both --no-video and --no-csv set)", file=sys.stderr)
        return 1

    # Resolve preset-dependent defaults (explicit flags win over the preset).
    preset = PRESETS[args.mode]
    def pick(value, key):
        return value if value is not None else preset[key]

    config = TrackerConfig(
        mode=args.mode,
        bg_samples=args.bg_samples,
        bg_mode=pick(args.bg_mode, "bg_mode"),
        threshold=pick(args.threshold, "threshold"),
        min_area=pick(args.min_area, "min_area"),
        max_area=pick(args.max_area, "max_area"),
        blur=pick(args.blur, "blur"),
        max_distance=args.max_distance,
        max_disappeared=args.max_disappeared,
        min_hits=args.min_hits,
        reid=not args.no_reid,
        merge_aware=not args.no_merge_aware,
        merge_area_factor=args.merge_area_factor,
        activity_window_s=args.activity_window,
        activity_px=pick(args.activity_px, "activity_px"),
        sleep_seconds=args.sleep_seconds,
        draw_trail=not args.no_trail,
        trail_length=args.trail_length,
        draw_heading=not args.no_heading,
        draw_bbox=args.bbox,
    )

    summary = process_video(
        input_path=args.input,
        output_video=output_video,
        output_csv=output_csv,
        config=config,
        progress=not args.quiet,
    )

    if output_video:
        print(f"  video: {output_video}")
    if output_csv:
        print(f"  csv:   {output_csv}")

    if output_csv and not args.no_analyze:
        if not args.quiet:
            print("Analyzing trajectories...", flush=True)
        prefix, _ = os.path.splitext(output_csv)
        result = analyze_tracks(
            csv_path=output_csv,
            fps=summary["fps"],
            out_prefix=prefix,
            background=summary["background"],
            px_per_mm=args.px_per_mm,
            center_radius_frac=args.center_frac,
            wells=summary.get("wells"),
        )
        if result["summary_csv"]:
            print(f"  summary: {result['summary_csv']}  ({result['n_fish']} fish)")
        if result["heatmap_png"]:
            print(f"  heatmap: {result['heatmap_png']}")
        if result["trajectories_png"]:
            print(f"  paths:   {result['trajectories_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
