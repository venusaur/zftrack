# zftrack — zebrafish video tracker

Tracks zebrafish in a top-down arena recording, draws a colored **outline** and a
stable **ID number** on each fish, and exports per-frame trajectories.

It uses classical computer vision (no GPU, no training, no labels), which suits
lab footage with a **static camera and static arena**:

```
median background  →  darker-than-background blob detection  →  Kalman + Hungarian
data association  →  annotated MP4 + trajectory CSV
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python track.py /path/to/video.mp4
```

Writes `video_tracked.mp4` (annotated) and `video_tracks.csv` (trajectories)
next to the input. To choose paths explicitly:

```bash
python track.py video.mp4 -o out/tracked.mp4 -c out/tracks.csv
```

Useful flags (see `python track.py -h` for all):

| Flag | Default | Meaning |
|------|---------|---------|
| `--threshold` | 13 | Darker-than-background sensitivity (0–255). Lower = more sensitive. |
| `--min-area` / `--max-area` | 200 / 6000 | Accepted blob size in px² (rejects debris and over-large artefacts). |
| `--max-distance` | 120 | Max px a fish may move between frames and keep its ID. |
| `--max-disappeared` | 48 | Frames a fish may be unseen before its ID is retired. |
| `--min-hits` | 3 | Detections before a track is confirmed (noise filter). |
| `--no-trail` / `--bbox` | — | Drop motion trails / add bounding boxes. |
| `--no-merge-aware` | — | Disable parking IDs on a shared blob when two fish overlap. |
| `--no-video` / `--no-csv` | — | Skip one of the two outputs. |

## Outputs

**Annotated video** — per fish: colored contour outline, ID number, and a fading
motion trail. A fish that is momentarily lost (occlusion / low contrast over the
dark rim) is shown as a small circle with `ID?` while the tracker coasts on its
predicted position. A HUD shows frame number, time, and live fish count.

**Trajectory CSV** — one row per fish per frame:

```
frame, time_s, id, cx, cy, bbox_x, bbox_y, bbox_w, bbox_h, area, detected
```

`cx,cy` is the centroid; `detected` is 1 (this is straightforward to load with
pandas for speed, dwell-time, or thigmotaxis analysis).

## How it works

1. **Background** (`background.py`) — per-pixel median of ~80 frames sampled
   across the clip. Because fish move and the arena does not, the median is the
   empty arena; the dish, holders and vignette cancel out.
2. **Detection** (`detector.py`) — `background − frame` (fish are darker than the
   background) → blur → threshold → morphological close/open → contours filtered
   by area. The signed difference ignores brighter artefacts like glints.
3. **Tracking** (`tracker.py`) — each fish carries a constant-velocity Kalman
   filter; detections are matched to predicted positions by the Hungarian
   algorithm (globally optimal, distance-gated). Unmatched detections start new
   tracks (confirmed after `min_hits`); tracks unseen for `max_disappeared`
   frames are retired. Optionally, a track whose blob merged with a neighbour's
   is *parked* on that blob so its ID survives the overlap.

## Known limitations

* **Overlapping fish.** When two fish swim as one blob, they cannot be told
  apart by shape alone, so an ID may switch or restart when they separate. This
  is the fundamental limit of markerless tracking; resolving it reliably needs an
  appearance/identity model (e.g. idtracker.ai-style CNN fingerprinting) or a
  known, enforced fish count.
* **Static-scene assumption.** A moving camera or shifting arena breaks the
  median background; re-shoot on a tripod or add stabilization first.
* Tuned defaults are for the sample arena (1280×960, dark larvae on a light
  dish). Other setups may need `--threshold` / `--min-area` adjustment.
