# Escalator Monitor

[![CI](https://github.com/pranjal-agarwal01/Station-Facility-Monitor-Escalator-using-Computer-Vision-/actions/workflows/ci.yml/badge.svg)](https://github.com/pranjal-agarwal01/Station-Facility-Monitor-Escalator-using-Computer-Vision-/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-Hugging%20Face%20Spaces-ffcc4d)](https://huggingface.co/spaces/pranjal-agarwal01/escalator-monitor)
![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Escalators in stations and malls stop more often than people think, and a
stopped one often goes unreported until passengers complain. The CCTV cameras pointed at them are already there. This
project watches that footage and reports, in real time, whether each escalator
is **working**, **stopped with people on it** (a fault) or **idle**, together
with an event log, fault snapshots and webhook alerts.

It needs no new hardware and no training data. It combines a pretrained person
detector (YOLO11) with dense optical flow, and every decision can be traced
back to numbers shown on screen.

<p align="center">
  <img src="docs/demo.gif" alt="Annotated output: the escalator stops while people are on it and the status switches to STOPPED / FAULT" width="480">
  <br><sub>Annotated output on the built-in synthetic test scene: the belt stops with people on it, the monitor raises STOPPED / FAULT, then returns to WORKING when it restarts.</sub>
</p>

## Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Evaluation and benchmarks](#evaluation-and-benchmarks)
- [Put it online](#put-it-online)
- [Project layout](#project-layout)
- [Limitations and roadmap](#limitations-and-roadmap)

## How it works

```mermaid
flowchart LR
    A[Frame] --> B[YOLO11 person detection<br/>on the ROI crop, every Nth frame]
    B --> C[IoU tracker<br/>keeps boxes between runs]
    A --> D[Dense optical flow, DIS<br/>on the ROI crop]
    C -->|mask people out| E[Region statistics<br/>left rail, right rail, steps]
    D --> E
    E --> F[Direction checks<br/>vertical, rails agree]
    F --> G[Move confidence]
    G --> H[Hysteresis state machine<br/>45-frame window]
    C -->|occupancy| H
    H --> I[WORKING / STOPPED / IDLE<br/>events, snapshots, webhooks]
```

1. **ROI.** You click the escalator's four corners once per camera. The quad is
   split into two handrail strips and the steps area.
2. **People.** YOLO11 runs on the ROI crop (which also helps with small,
   distant people), a light IoU tracker keeps their boxes between detector
   runs, and those boxes are masked out of the motion analysis. A person
   walking up a stopped escalator must not look like a moving escalator.
3. **Surface motion.** DIS optical flow on the ROI crop gives magnitude and
   direction per pixel. For each region we measure mean speed and
   *consistency* (how aligned the vectors are). Motion only counts if it is
   mostly vertical and both handrails agree on the direction. That rejects
   people crossing the frame, lighting changes and most sensor noise.
4. **Decision.** Handrail and step scores are fused into a move confidence.
   A rolling window feeds a state machine with separate enter and exit
   thresholds, so the status does not flicker:

| Surface | People in ROI | Move confidence | Status |
|---|---|---|---|
| moving | yes | high | **WORKING** |
| moving | no | high | **WORKING** (running empty) |
| still | yes | low | **STOPPED / FAULT** |
| still | no | low | **IDLE** (e.g. energy-saving stop) |

When the escalator stops, occupancy is judged on the most recent frames only,
so riders stepping off just before an energy-saving stop are not reported as a
fault. Optional extras: camera-shake compensation (subtracts the background's
median flow) and a wrong-direction alert.

## Quick start

```bash
git clone https://github.com/pranjal-agarwal01/Station-Facility-Monitor-Escalator-using-Computer-Vision-.git
cd Station-Facility-Monitor-Escalator-using-Computer-Vision-
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[detect,video]"
```

YOLO11n weights (`yolo11n.pt`, 5.4 MB) download automatically on first use.

**A video file, with a preview window**

```bash
escalator-monitor run --input input/station_cam.mp4
```

On the first run, click the escalator corners (top-left, top-right,
bottom-right, bottom-left) and press Enter. They are saved to
`output/roi_<video>.json` and reused next time. Keys: `q` quit, `p` pause,
`r` re-select ROI, `s` snapshot, `f` fullscreen.

**A live RTSP camera on a server, with alerts**

```bash
escalator-monitor run --input rtsp://user:pass@10.0.0.21:554/stream1 \
    --roi output/roi_cam21.json --headless --output "" \
    --webhook https://hooks.slack.com/services/XXX
```

Live streams are read on a background thread that always keeps the newest
frame, so processing never falls behind the camera.

**Try it without footage**

```bash
escalator-monitor synth demo.mp4                       # synthetic clip + ground truth
escalator-monitor run --input demo.mp4 --roi demo.roi.json --detector replay:demo.boxes.json --headless
escalator-monitor evaluate output/timeline.csv demo.gt.csv
```

**Web app or Docker**

```bash
pip install -e ".[web]" && python app.py               # http://localhost:7860
docker build -t escalator-monitor . && docker run -p 7860:7860 escalator-monitor
```

Older invocations such as `python -m escalator_monitor --config cam.yaml`
still work (`run` is the default command).

## Outputs

| File | Contents |
|---|---|
| `output/result.mp4` | Annotated H.264 video: status badge, ROI and regions tinted by motion, people, live metrics |
| `output/events.csv` | One row per state change with video time, wall time, the metrics behind it and the snapshot path |
| `output/timeline.csv` | Per-frame state, move confidence, people count and window ratios, ready for plotting or `evaluate` |
| `output/report.json` | Time in each state, availability, fault episodes with start/end, processing speed, config used |
| `output/snapshots/` | A JPEG of every STOPPED / FAULT transition, useful as evidence for maintenance |

`availability_pct` is working time divided by working + stopped time. Idle
time is left out because an empty, stopped escalator is not a failure.

Webhook payload (POSTed from a background thread on every state change):

```json
{"event": "WORKING -> STOPPED / FAULT", "from_state": "WORKING", "to_state": "STOPPED / FAULT",
 "source": "station_cam", "frame": 184, "video_time": "0:00:07.320", "wall_time": "2026-09-25T08:15:55",
 "people": 3, "move_confidence": 0.0, "snapshot": "output/snapshots/20260925_081555_f184_STOPPED_FAULT.jpg"}
```

![Timeline of state, move confidence and occupancy](docs/timeline.png)

## Configuration

Defaults live in [`escalator_monitor/config.py`](escalator_monitor/config.py).
A per-camera YAML file only lists what differs
([example](configs/example.yaml)); unknown keys are reported and values are
validated on load.

```bash
escalator-monitor run --config configs/example.yaml
```

| Parameter | Default | What it does |
|---|---|---|
| `yolo_model` | `yolo11n.pt` | Detector; `yolo11s.pt`/`yolo11m.pt` are more accurate but slower |
| `detect_every_n_frames` | `3` | Run YOLO every Nth frame, track in between |
| `handrail_mag_gate`, `steps_mag_gate` | `0.10`, `0.15` | Minimum flow (px/frame at half resolution) to count as motion; lower for distant cameras |
| `consistency_gate` | `0.55` | Minimum alignment of the flow vectors |
| `move_confidence_min` | `0.35` | Confidence needed to call a frame "moving" |
| `window_size` | `45` | Frames in the voting window: longer is steadier but slower to react |
| `enter_working_ratio`, `exit_working_ratio` | `0.45`, `0.25` | Hysteresis thresholds |
| `compensate_camera_motion` | `false` | Cancel camera shake using the background around the ROI |
| `expected_direction` | `any` | `up`/`down` (image space) raises a WRONG DIRECTION alert |
| `process_max_width` | `0` | Downscale large frames before analysis (0 = native) |
| `webhook_url` | `""` | POST every state change here |

## Evaluation and benchmarks

**Accuracy.** Real labelled escalator footage is hard to come by, so the
repository ships a scene generator (`escalator_monitor/synthetic.py`). It renders
a perspective escalator with scrolling steps and handrails, riders, people
walking on a stopped escalator, sensor noise, flicker and camera shake, and
records the exact state and person boxes for every frame. Each scenario below
runs through *working, fault, working, idle* (27 s, 3 seeds), with simulated
person boxes, so it measures the motion analysis and the state machine, not
the detector:

| Scenario | Frame accuracy | Fault detected after | Restart detected after | Idle detected after | False fault alarms | Missed changes |
|---|---|---|---|---|---|---|
| clean | 87.0% | 1.32 s | 0.80 s | 1.32 s | 0 | 0 |
| sensor noise (σ = 8) | 87.0% | 1.32 s | 0.80 s | 1.32 s | 0 | 0 |
| brightness flicker ±10 % | 87.0% | 1.32 s | 0.80 s | 1.32 s | 0 | 0 |
| crowded (5-6 people) | 87.0% | 1.32 s | 0.80 s | 1.32 s | 0 | 0 |
| running down | 87.0% | 1.32 s | 0.80 s | 1.32 s | 0 | 0 |
| slow belt (half speed) | 86.9% | 1.32 s | 0.83 s | 1.32 s | 0 | 0 |
| camera shake ±1.5 px | 51.3% | 4.66 s | 0.56 s | 5.52 s | 0 | 3 |
| camera shake + compensation | 86.9% | 1.32 s | 0.81 s | 1.33 s | 0 | 0 |

The ~13 % of frames counted as wrong is the confirmation delay. The state
machine only accepts a stop once three quarters of its 45-frame window
(1.8 s at 25 fps) is still, about 1.3 s, and after that every frame is correct. The shake rows show
why compensation exists: without it, a shaking camera looks like a moving
escalator and stops are missed. Synthetic scenes are easier than real CCTV,
so treat these numbers as a regression baseline and measure your own cameras
as described below.

**On your own footage.** Label the true states in a CSV (a spreadsheet works):

```csv
start_s,end_s,state
0,42.5,WORKING
42.5,95,STOPPED
95,130,IDLE
```

Then run `escalator-monitor evaluate output/timeline.csv labels.csv` to get
frame accuracy, per-state precision and recall, time to detect each change,
missed changes and false fault alarms.

**Speed.** Measured on a 4-vCPU cloud VM with no GPU, mean milliseconds per
frame (`escalator-monitor benchmark --yolo yolo11n.pt` reproduces it on your
machine):

| Resolution | Motion analysis | Overlay | YOLO11n, per run | FPS, YOLO every 3rd frame | FPS, motion only |
|---|---|---|---|---|---|
| 640x360 | 6.2 | 1.3 | 98 | 25 | 143 |
| 1280x720 | 9.3 | 3.6 | 94 | 23 | 86 |
| 1920x1080 | 20.0 | 7.1 | 78 | 19 | 48 |

Video decoding and encoding are excluded. End to end in the web app (decode,
detect, analyse, draw, H.264 encode), an 800x600 clip processed at 17 FPS on
the same machine. A GPU makes detection nearly free. YOLO11n's cost was
measured with its network architecture (`yolo11n.yaml`); CPU inference time
does not depend on the weight values.

## Put it online

**Free live demo on Hugging Face Spaces (automatic):**

1. Create a [Hugging Face](https://huggingface.co/join) account and an
   [access token](https://huggingface.co/settings/tokens) with *write* permission.
2. In this GitHub repository open *Settings → Secrets and variables → Actions*
   and add a secret named `HF_TOKEN` with that token.
3. Optionally add a *variable* `HF_SPACE` such as `your-hf-name/escalator-monitor`
   (that name is the default).
4. Push to `main`, or run the **Deploy demo** workflow from the Actions tab.
   The Space is created on the first run and redeployed on every push. The
   first build takes a few minutes; after that it is live at
   `https://huggingface.co/spaces/<your-hf-name>/escalator-monitor`.

The free CPU tier is enough: the app downsizes frames to 960 px and caps
clips at 60 s (`PROCESS_WIDTH` and `MAX_SECONDS` environment variables). The
Space also exposes an API:

```python
from gradio_client import Client, handle_file

client = Client("your-hf-name/escalator-monitor")
video, summary, *_ = client.predict(
    handle_file("clip.mp4"),
    "812,140,1105,140,1290,1040,640,1040",  # escalator corners in video pixels
    0.35,  # move confidence needed
    3,  # run the detector every N frames
    False,  # fast mode
    False,  # camera-shake compensation
    "any",  # expected direction
    30,  # analyse at most N seconds
    api_name="/analyze",
)
```

**Anywhere else:** the Docker image serves the same app on port 7860, so it
runs on Render, Railway, Fly.io, Google Cloud Run or a station server.

**For real cameras:** run the CLI next to the cameras (an edge PC or a Jetson)
with `--headless` and a webhook, one process per camera. Alerts go to Slack,
Teams or a ticketing system, and `events.csv`/`report.json` feed dashboards.

## Project layout

```
escalator_monitor/
  config.py       all tunable parameters, YAML loading and validation
  geometry.py     ROI quad, corner ordering, handrail/steps masks
  flow.py         DIS optical flow and per-region statistics
  motion.py       people masking, direction checks, scoring, fusion
  tracking.py     detections and the IoU tracker
  detector.py     YOLO wrapper, plus replay/null detectors for tests
  state.py        hysteresis state machine
  pipeline.py     per-frame orchestration (no I/O)
  runner.py       video in, outputs out: CSV, JSON, snapshots, webhooks
  render.py       annotated overlay
  video.py        file/RTSP/webcam input, H.264 output
  gui.py          desktop preview and ROI picker
  synthetic.py    synthetic scenes with ground truth
  evaluate.py     metrics against labelled intervals
  benchmark.py    accuracy and speed suites
  cli.py          `escalator-monitor` command
app.py            Gradio web app (also the Hugging Face Space)
tests/            pytest suite, runs without model weights or GPU
configs/          example per-camera config
deploy/           Hugging Face Space files
```

Run the checks locally with `pip install -e ".[dev]" && ruff check . && pytest`.

## Limitations and roadmap

- **Fixed cameras.** Shake compensation handles small translations, not pan,
  tilt or zoom.
- **Manual ROI.** Four clicks per camera, saved for reuse.
- **It sees symptoms, not causes.** It reports "stopped while occupied", not
  whether that was a motor fault or an emergency stop.
- **Hard conditions.** Very low light, glare and heavy occlusion reduce both
  detection and flow quality. Distant cameras may need lower magnitude gates.
- **Synthetic accuracy numbers.** They validate the logic. Real-world accuracy
  should be measured per site with `evaluate`.

Done: camera-shake compensation, wrong-direction alerts, web demo, evaluation tooling.

Next:
- [ ] Automatic escalator localisation (segmentation or a custom YOLO class)
- [ ] Multi-camera service with a live dashboard and Prometheus metrics
- [ ] A labelled real-footage benchmark
- [ ] ONNX / TensorRT export for Jetson-class devices
- [ ] Other station assets: elevator doors, ticket gates

## Tech stack

Python 3.10+, OpenCV (DIS optical flow, video I/O), Ultralytics YOLO11
(PyTorch), NumPy, Gradio, imageio-ffmpeg, PyYAML, pytest, ruff, GitHub
Actions, Docker, Hugging Face Spaces.

## References

1. Kroeger et al., *Fast Optical Flow using Dense Inverse Search*, ECCV 2016.
2. Farnebäck, *Two-Frame Motion Estimation Based on Polynomial Expansion*, SCIA 2003.
3. Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016.
4. [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/)

## License

[MIT](LICENSE)
