# zftrack — zebrafish video tracker

Tracks zebrafish from top-down video, draws a colored **outline** and a stable
**ID number** on each fish, resolves **heading**, scores **sleep**, and exports
per-frame trajectories plus a metrics/figure analysis. Classical computer vision
— no GPU, no training, no labels.

Two modes:

| Mode | For | Detector |
|------|-----|----------|
| `arena` (default) | one open dish, a few fish | median background, motion segmentation |
| `plate` | multi-well sleep assay, one larva per well | **max** background + per-well anchoring (robust to motionless/sleeping fish) |

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Open-arena video
python track.py video.mp4

# Multi-well sleep assay (10-min plate recording)
python track.py fish.mp4 --mode plate

# Open arena with a known number of fish (caps IDs, avoids overlap inflation)
python track.py video.mp4 --num-fish 4
```

Each run writes, next to the input (or use `-o` / `-c`):

* `*_tracked.mp4` — annotated video (outline, ID, heading arrow, motion trail,
  `Zzz` on sleeping fish, well circles in plate mode, HUD with live counts)
* `*_tracks.csv` — per-frame trajectories
* `*_summary.csv` — one row per fish (distance, speed, sleep, thigmotaxis)
* `*_heatmap.png`, `*_trajectories.png` — occupancy and path figures
* `*_timeseries.csv` — per-fish, per-time-bin sleep / activity / distance
* `*_actogram.png` — sleep raster (rows = fish, columns = time bins, blue = asleep)
* `*_phases.csv` — per-fish sleep stats per light/dark phase + group mean ± SEM

### Sleep detection & analytics

A fish is scored **sleeping** after **`--sleep-seconds` (default 60) of
continuous inactivity** — it stays within an `--activity-px` radius over the
trailing 1 s window. Inactivity is measured as the *spatial spread* of the
fish's (median-filtered) positions, not cumulative path length: path length sums
every step, so it accumulates sub-pixel detection jitter and grows with frame
rate (at 30 fps a motionless fish accrues ~15 px of fake "movement" from noise),
which would suppress real sleep. Spread is frame-rate independent and robust to
isolated detection outliers. Written per frame (`sleeping` column) and
summarised per fish (`sleep_total_s`, `sleep_bouts`, `longest_sleep_s`).

The analysis also bins the recording (`--bin-seconds`, default 60) into the
time-series and actogram, and splits it into **light/dark phases** — auto-detected
from frame brightness, or set explicitly with
`--phases "light:0-300,dark:300-600"` (seconds). Per-phase sleep is reported per
fish and as a group **mean ± SEM**.

> The default 60 s threshold needs a recording longer than 60 s. The short
> `video.mp4` sample (~32 s) can never trigger it — use a longer clip, or
> `--sleep-seconds 5` to demo the labelling.

Key flags (`python track.py -h` for all): `--threshold`, `--min-area/--max-area`,
`--max-disappeared`, `--num-fish`, `--sleep-seconds`, `--activity-px`, `--bin-seconds`,
`--phases`, `--px-per-mm` (real-world units), `--no-reid`,
`--no-video/--no-csv/--no-analyze`.

## Outputs — trajectory CSV columns

```
frame, time_s, id, cx, cy, bbox_x, bbox_y, bbox_w, bbox_h, area,
length, width, axis_deg, heading_deg, speed_px, inactive_s, sleeping, detected
```

`heading_deg` points toward the head (resolved from motion direction, falling
back to the darker — eyed — body tip). In `plate` mode `id` is the well number.

## How it works

1. **Background** (`background.py`) — `median` (arena) or `max` (plate). Max takes
   the brightest value per pixel, recovering the bright well floor wherever a
   fish ever moved, so a motionless larva is **not** baked into the background
   (the failure that breaks sleep detection with a median).
2. **Detection** (`detector.py`) — `background − frame` (fish are darker) →
   threshold → morphology → contours filtered by area; each blob also yields a
   body axis, the two tips and an appearance descriptor.
3. **Tracking**
   * arena (`tracker.py`) — constant-velocity Kalman per fish + Hungarian data
     association; merge-aware coasting and a re-ID buffer that revives a
     briefly-lost fish under its old ID instead of spawning a new one.
   * plate (`plate.py`) — wells are found from the cloud of detections (one
     occupancy peak per well), then exactly one persistent track is anchored to
     each well (IDs capped at the well count). Each well is then searched
     individually every frame for its single fish — robust to motion blur and
     motionless fish, giving ~99 % per-frame coverage. The dark rim cancels
     against the max background, so the whole well can be searched without false
     positives.
4. **Sleep / activity** — per-track continuous-inactivity timer → sleep flag.
5. **Analysis** (`analysis.py`) — per-fish metrics, occupancy heatmap and path
   map. Thigmotaxis is scored against the shared arena (arena mode) or each
   fish's own well (plate mode).

## Tests

Unit tests cover detection, tracking (ID stability, re-ID, sleep scoring), well
finding, background estimation, arena fitting and the analysis layer. They use
synthetic frames/videos, so no sample footage is required.

```bash
source .venv/bin/activate
pip install pytest
pytest
```

## Hardware (CAD models)

The `models/` folder holds the CAD for the physical imaging rig that produces the
footage this tracker consumes.

**`models/current/`** — the rig in use:

| File | Format | What it is |
|------|--------|------------|
| `Zebra Fish Box Assembly` | `.step`, `.stl` | The imaging enclosure — top-down camera box that holds the plate and controls lighting for the recordings. |
| `ZebraBox Mesh Plate` | `.step`, `.stl` | The multi-well mesh plate that seats the larvae (one fish per well for the `plate` sleep assay). |

Each model is provided as a `.step` (parametric, for editing in CAD) and a `.stl`
(mesh, for 3D printing / viewing).

**`models/future/`** — parts for a planned motorized stage (SolidWorks `.SLDPRT` /
`.SLDASM`): a HIWIN MGN9H linear slider, 20×20 extrusion frame, bed, interface
plate, and brackets — for automating positioning in a later revision.

## Known limitations

* **Overlapping fish (arena).** Two fish merging into one blob can swap or
  restart an ID when they separate — the limit of markerless tracking without an
  identity/appearance model. Two mitigations are built in: a re-ID *rescue* pass
  revives a coasting track when its fish reappears just beyond the per-frame
  match radius, and **`--num-fish N`** caps the total IDs at the known fish count
  (extra detections revive an existing ID instead of spawning a new one). On the
  4-fish sample clip this takes the ID count from ~14 down to 4. It cannot
  *separate* two fish that stay merged — if a pair overlaps for most of the clip,
  the second fish is only tracked while it is individually visible.
* **Never-moving fish (plate).** A larva that does not move *at all* for the
  whole recording stays in the max background and cannot be segmented (it is also,
  trivially, asleep). Indistinguishable from a fixed speck of debris.
* **Detection coverage.** Fast darts (motion blur) and edge/partial wells reduce
  per-frame detection rate, which can underestimate distance for very active
  fish. Tune `--threshold` / `--max-area` per setup.
* Defaults are tuned for the sample footage; other rigs may need adjustment.
