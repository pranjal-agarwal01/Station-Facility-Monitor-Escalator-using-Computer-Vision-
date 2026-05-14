# Escalator State Monitor

> Real-time computer vision system that monitors escalator operational status from CCTV footage and detects faults.

Combines YOLO11 person detection with dense optical flow to classify escalators into three states — **WORKING**, **STOPPED / FAULT**, or **IDLE** — and produces a structured event log suitable for automated maintenance workflows.

---

## Demo

| State | Description | Visual |
|---|---|---|
| 🟢 **WORKING** | Escalator surface moving consistently | Green status badge, motion-region overlay in green |
| 🔴 **STOPPED / FAULT** | People present but escalator is still | Red status badge, fault snapshot saved to disk |
| 🟡 **IDLE** | No people detected | Yellow status badge, no fault diagnosis |

The system writes an annotated MP4 alongside the original video and a CSV event log with frame-level timestamps and supporting metrics.

---

## Why this exists

CCTV cameras already cover escalators in railway stations, malls, airports, and metros. But a human operator is needed to interpret that footage, and faults often go unreported until passengers complain. This project adds an automated interpretation layer:

- **No new hardware** — works with existing CCTV streams
- **Real-time** — 15–30 FPS on a standard laptop, faster with GPU
- **Explainable** — every status decision is backed by measurable metrics shown live in the debug overlay
- **Structured output** — CSV event log integrates directly into maintenance ticketing systems

---

## How it works

```
┌──────────────┐
│ Video Frame  │
└──────┬───────┘
       │
       ├────────────────────────────┐
       ▼                            ▼
┌──────────────────┐      ┌──────────────────┐
│  YOLO11n         │      │  Convert to      │
│  Person          │      │  Grayscale       │
│  Detection       │      └────────┬─────────┘
└──────┬───────────┘               │
       │ person boxes              │
       ▼                           ▼
┌────────────────────────────────────────────┐
│  Build Region Masks                        │
│  • Handrail strips (left + right edges)    │
│  • Steps area (center)                     │
│  • People regions excluded with padding    │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│  Dense Optical Flow (DIS / Farneback)      │
│  Compute motion vectors per region         │
└──────────────┬─────────────────────────────┘
               │ magnitude + direction
               ▼
┌────────────────────────────────────────────┐
│  Directional Sanity Checks                 │
│  • Motion must be vertical (escalator)     │
│  • Left + right handrails must agree       │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│  Smooth Scoring + Fusion → confidence      │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│  Hysteresis State Machine                  │
│  WORKING ↔ STOPPED ↔ IDLE                  │
│  (45-frame rolling window vote)            │
└──────────────┬─────────────────────────────┘
               │
               ▼
   Status + Annotated MP4 + CSV Event Log
```

**Key design choices:**

- **YOLO11-nano** for person detection — pretrained on COCO, fast enough for CPU, no custom training needed
- **DIS optical flow** (with Farneback fallback) — measures motion vectors with both magnitude and direction
- **Dual-region analysis** — handrails and steps analyzed separately and fused, so the system stays robust whether the escalator is empty or busy
- **People masking** — person bounding boxes are zeroed out before motion analysis, so we measure escalator surface motion, not pedestrian motion
- **Directional checks** — escalator motion must be vertical and consistent across both handrails, eliminating false positives from people walking on stopped escalators
- **Hysteresis state machine** — different thresholds for entering and leaving states (0.45 vs 0.25) prevent status flicker

---

## Quick Start

### Requirements

- Python 3.10 or higher
- ~500 MB free disk space (for the YOLO model and output files)
- A webcam, IP camera, or recorded video file
- GPU optional but recommended for higher FPS

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/escalator-monitor.git
cd escalator-monitor

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate         # Linux / macOS
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Default config (uses settings baked into semi.py)
python semi.py

# Use a YAML config to override settings per camera/video
python semi.py --config mall3.yaml

# Override input video on the fly
python semi.py --input input/your_video.mp4
```

**First run:** the YOLO11-nano model (`yolo11n.pt`, ~5.5 MB) is downloaded automatically by Ultralytics on first use. After that it's cached locally.

**ROI selection:** on the first run with a new video, click the four corners of the escalator in this order: **Top-Left → Top-Right → Bottom-Right → Bottom-Left**, then press `ENTER`. The selection is saved as `output/roi_<videoname>.json` and reused on subsequent runs.

### Controls

| Key | Action |
|---|---|
| `q` | Quit (output is flushed and saved) |
| `p` | Pause / resume |
| `r` | Re-select the ROI |
| `s` | Save a snapshot of the current frame |
| `f` | Toggle fullscreen |

---

## Project Structure

```
escalator-monitor/
│
├── semi.py                 # Main script
├── mall3.yaml              # Example per-camera config (optional)
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── input/                  # Place your videos here
│   └── your_video.mp4
│
├── output/                 # Auto-created on first run
│   ├── result.mp4              # Annotated video
│   ├── events.csv              # State transition log
│   ├── roi_<videoname>.json    # Saved ROI coordinates
│   └── snapshots/              # Fault-event JPGs
│
└── yolo11n.pt              # YOLO weights, auto-downloaded on first run
```

You only commit the first four files to git. The `input/`, `output/`, and `yolo11n.pt` are listed in `.gitignore`.

---

## Configuration

All tunable parameters live in two places:

1. **`Config` dataclass at the top of `semi.py`** — the defaults
2. **YAML config file** (e.g. `mall3.yaml`) — overrides for a specific camera or video

You only override the fields you want to change. Anything not listed in your YAML falls back to the default.

### Key parameters

| Parameter | Default | Purpose |
|---|---|---|
| `yolo_model` | `yolo11n.pt` | Detector. Use `yolo11s.pt` / `yolo11m.pt` for higher accuracy at the cost of speed. |
| `person_conf_threshold` | `0.35` | Minimum confidence to count a YOLO detection as a person. |
| `detect_every_n_frames` | `3` | Run YOLO every Nth frame; IoU-track in between (≈3× speedup). |
| `handrail_mag_gate` | `0.10` | Minimum flow magnitude on handrail strips. Calibrated for distant CCTV. |
| `steps_mag_gate` | `0.15` | Same, for central step area. |
| `consistency_gate` | `0.55` | Minimum directional alignment to count as motion. |
| `require_vertical_motion` | `true` | Escalator motion must be primarily vertical, not horizontal. |
| `require_handrail_agreement` | `true` | Left and right handrails must move in the same direction. |
| `window_size` | `45` | Rolling window length (frames) for temporal smoothing. |
| `enter_working_ratio` | `0.45` | Fraction of recent frames that must be "moving" to enter WORKING. |
| `exit_working_ratio` | `0.25` | Drop below this to exit WORKING (hysteresis gap). |
| `save_fault_snapshots` | `true` | Save a JPG when a STOPPED / FAULT event is detected. |
| `webhook_url` | `""` | POST event JSON here on every state transition (optional). |

### Example YAML

```yaml
input_video: input/stair2.mp4
output_video: output/result.mp4

yolo_model: yolo11n.pt
person_conf_threshold: 0.35

# Lower these for far/zoomed-out CCTV
handrail_mag_gate: 0.08
steps_mag_gate: 0.12

# Stricter direction check for noisy environments
vertical_ratio_min: 1.8

webhook_url: "https://hooks.slack.com/services/..."
save_fault_snapshots: true
```

---

## Output Files

### `output/result.mp4`
Full-resolution annotated video with:
- Status badge at the top (color-coded by state)
- ROI quadrilateral overlay
- Person bounding boxes with YOLO confidence scores
- Semi-transparent motion-region overlays (green = moving, orange = still)
- Live debug panel with all metrics
- Progress bar along the bottom

### `output/events.csv`

| Column | Description |
|---|---|
| `frame` | Frame number when transition occurred |
| `timestamp_wall` | Real-world ISO timestamp |
| `video_time` | Position in video (HH:MM:SS) |
| `event` | Transition (e.g. `WORKING -> STOPPED / FAULT`) |
| `details` | Supporting metrics that triggered the transition |
| `snapshot_path` | Path to saved fault snapshot (if applicable) |

Sample row:
```
450,2025-12-05T14:30:22,0:00:18.000,WORKING -> STOPPED / FAULT,people=3 hr_mag=0.05 conf=0.08,output/snapshots/20251205_143022_f450_STOPPED_FAULT.jpg
```

### `output/snapshots/`
JPG snapshots of each STOPPED / FAULT event. Useful evidence for maintenance teams.

### `output/roi_<videoname>.json`
Saved ROI coordinates so you don't have to re-draw on every run.

---

## Performance

Measured on a typical mid-range laptop (Intel i5-11th gen, no GPU) with 1080p input:

| Configuration | FPS |
|---|---|
| YOLO11n + DIS flow, detect every 3 frames | 22–28 |
| YOLO11n + Farneback, detect every 3 frames | 14–18 |
| YOLO11s + DIS flow, detect every 3 frames | 12–16 |
| YOLO11n + DIS flow + GPU (CUDA) | 50+ |

For real-time CCTV processing on CPU, the default configuration (YOLO11n + DIS + detect_every_n_frames=3) is recommended.

---

## Limitations

Honest about what this system cannot do:

- **Fixed camera required.** Camera movement (pan, tilt, shake) introduces motion that cannot be cleanly distinguished from escalator motion using 2D optical flow alone.
- **Manual ROI selection on first run.** Automatic escalator localization is not implemented.
- **Binary fault detection only.** The system detects "stopped" but cannot diagnose specific causes (motor failure, jammed step, emergency stop activation).
- **Lighting sensitivity.** Very low light, motion blur, heavy reflections, or extreme angles reduce both detection and flow reliability.
- **Per-camera tuning may be needed.** The 20+ configuration parameters can be adjusted for cameras with significantly different distance, angle, or frame rate.

---

## Roadmap

- [ ] Camera motion compensation via ORB feature matching + RANSAC
- [ ] Automatic ROI detection using a custom-trained YOLO class for "escalator"
- [ ] Direction-reversal detection (flagging escalators running the wrong way)
- [ ] Web dashboard for live monitoring across multiple cameras
- [ ] Native deployment on NVIDIA Jetson and similar edge devices
- [ ] Extension to other facilities (elevators, ticket machines, washroom queues) using the same framework

---

## Tech Stack

| Component | Used For |
|---|---|
| Python 3.10+ | Core implementation |
| Ultralytics YOLO11 | Person detection (pretrained on COCO) |
| OpenCV 4.8+ | Video I/O, optical flow, drawing |
| OpenCV-contrib | DIS optical flow algorithm |
| PyTorch | YOLO inference backend |
| NumPy | Array operations on flow vectors |
| PyYAML | Per-camera config files |

---

## References

1. Redmon et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR.
2. Farneback, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion.* SCIA.
3. Kroeger et al. (2016). *Fast Optical Flow using Dense Inverse Search.* ECCV.
4. Ultralytics YOLO11 Documentation — [docs.ultralytics.com](https://docs.ultralytics.com/)
5. OpenCV Documentation — [docs.opencv.org](https://docs.opencv.org/)

---

## License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

---

## Contributing

Issues and pull requests welcome. If you've used this on a different kind of footage and want to share tuning advice, please open a discussion thread with a sample frame and your YAML config — others will benefit.
