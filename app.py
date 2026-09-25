"""Web demo (Gradio): upload an escalator clip, mark the escalator, get the analysis.

Runs locally with `python app.py` and as a Hugging Face Space.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from matplotlib.figure import Figure

from escalator_monitor.config import Config
from escalator_monitor.detector import ReplayDetector, YoloPersonDetector
from escalator_monitor.evaluate import evaluate, load_intervals, load_timeline
from escalator_monitor.geometry import Quad
from escalator_monitor.runner import run
from escalator_monitor.synthetic import DEMO_SEGMENTS, SceneSpec, Segment, write_clip

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("app")

MAX_SECONDS = int(os.environ.get("MAX_SECONDS", "60"))  # longest clip analysed per request
PROCESS_WIDTH = int(os.environ.get("PROCESS_WIDTH", "960"))  # frames are downscaled to this width
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolo11n.pt")
EXAMPLES_DIR = Path(os.environ.get("EXAMPLES_DIR", "examples"))  # optional real clips (+ .roi.json)
WORK_DIR = Path(tempfile.gettempdir()) / "escalator_monitor"
REPO_URL = "https://github.com/pranjal-agarwal01/Station-Facility-Monitor-Escalator-using-Computer-Vision-"

STATE_COLORS = {"WORKING": "#0ca30c", "STOPPED / FAULT": "#d03b3b", "IDLE": "#fab219", "INITIALIZING": "#898781"}
STATE_LABELS = {"WORKING": "Working", "STOPPED / FAULT": "Stopped / fault", "IDLE": "Idle", "INITIALIZING": ""}
CORNER_LABELS = ("TL", "TR", "BR", "BL")

_detector_lock = threading.Lock()
_yolo: YoloPersonDetector | None = None


def get_yolo() -> YoloPersonDetector:
    global _yolo
    with _detector_lock:
        if _yolo is None:
            _yolo = YoloPersonDetector(YOLO_MODEL)
        return _yolo


def file_hash(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- examples -----------------------------------------------------------------
# Synthetic clips come with simulated person boxes and ground truth, so the demo
# works without any footage; when one of them is analysed the recorded boxes are
# used instead of YOLO (which does not recognise the rendered figures).
SYNTHETIC: dict[str, dict] = {}  # md5 -> sidecar paths
LABELS = {
    "synthetic_fault_cycle.mp4": "Synthetic: running, fault, running, idle",
    "synthetic_crowded_shaky.mp4": "Synthetic: crowded, shaky camera",
}


def prepare_examples() -> tuple[list[list[str]], list[str]]:
    cache = WORK_DIR / "examples"
    cache.mkdir(parents=True, exist_ok=True)
    scenes = {
        "synthetic_fault_cycle.mp4": SceneSpec(DEMO_SEGMENTS, width=640, height=480, seed=5),
        "synthetic_crowded_shaky.mp4": SceneSpec(
            [Segment(6, True, 6), Segment(8, False, 5), Segment(6, True, 4)], jitter=1.0, noise=5.0, seed=9
        ),
    }
    examples, labels = [], []
    for name, spec in scenes.items():
        video = cache / name
        sidecars = {
            k: str(video.with_suffix("")) + s
            for k, s in (("boxes", ".boxes.json"), ("ground_truth", ".gt.csv"), ("roi", ".roi.json"))
        }
        if not all(Path(p).exists() for p in [video, *sidecars.values()]):
            log.info("Rendering example %s", name)
            sidecars = write_clip(spec, video)
        SYNTHETIC[file_hash(video)] = sidecars
        examples.append([str(video)])
        labels.append(LABELS[name])
    if EXAMPLES_DIR.is_dir():
        for clip in sorted(EXAMPLES_DIR.glob("*.mp4")):
            examples.append([str(clip)])
            labels.append(clip.stem.replace("_", " "))
    return examples, labels


def roi_sidecar(video: str) -> Path | None:
    """ROI stored next to a clip: synthetic examples, or examples/<name>.roi.json."""
    info = SYNTHETIC.get(file_hash(video))
    if info:
        return Path(info["roi"])
    for folder in (Path(video).parent, EXAMPLES_DIR):
        candidate = folder / f"{Path(video).stem}.roi.json"
        if candidate.exists():
            return candidate
    return None


# --- ROI picking ------------------------------------------------------------------
def read_first_frame(video: str) -> tuple[np.ndarray, float, float, int]:
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if not ok:
        raise gr.Error("Could not read that video. Try an MP4 (H.264) file.")
    h, w = frame.shape[:2]
    scale = min(1.0, PROCESS_WIDTH / w)
    if scale < 1.0:
        frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), scale, fps, count


def draw_roi(frame: np.ndarray | None, points: list[list[float]]) -> np.ndarray | None:
    if frame is None:
        return None
    img = frame.copy()
    ui = max(1.0, min(img.shape[:2]) / 480)
    if len(points) == 4:
        quad = Quad.from_points(points)
        overlay = img.copy()
        for name, poly in quad.regions(0.10).items():
            color = (70, 170, 255) if name == "steps" else (255, 190, 60)
            cv2.fillPoly(overlay, [np.round(poly).astype(np.int32)], color)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        points = quad.as_array().tolist()
        cv2.polylines(img, [np.round(points).astype(np.int32)], True, (40, 220, 90), max(2, int(2 * ui)), cv2.LINE_AA)
    elif len(points) >= 2:
        cv2.polylines(img, [np.round(points).astype(np.int32)], False, (255, 220, 0), max(2, int(2 * ui)), cv2.LINE_AA)
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), int(6 * ui), (40, 220, 90), -1, cv2.LINE_AA)
        label = CORNER_LABELS[i] if len(points) == 4 else str(i + 1)
        cv2.putText(
            img,
            label,
            (int(x + 8 * ui), int(y - 8 * ui)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * ui,
            (255, 255, 255),
            max(1, int(2 * ui)),
            cv2.LINE_AA,
        )
    return img


def points_to_text(points: list[list[float]], scale: float) -> str:
    if len(points) != 4:
        return ""
    quad = Quad.from_points(points).scaled(1 / scale)
    return ", ".join(f"{x},{y}" for x, y in quad.to_list())


def on_video(video: str | None):
    if not video:
        return None, None, 1.0, [], "", "Upload a clip or pick an example."
    frame, scale, fps, count = read_first_frame(video)
    points: list[list[float]] = []
    sidecar = roi_sidecar(video)
    if sidecar:
        points = Quad.load(sidecar).scaled(scale).as_array().tolist()
    duration = f"{count / fps:.1f} s" if count else "unknown length"
    hint = (
        "Escalator region loaded from the example (orange = handrails, blue = steps). Click to redraw it."
        if points
        else "Click the four corners of the escalator, including both handrails."
    )
    info = f"{frame.shape[1]}x{frame.shape[0]} preview, {fps:.0f} fps, {duration}. {hint}"
    return draw_roi(frame, points), frame, scale, points, points_to_text(points, scale), info


def on_click(frame, scale, points, evt: gr.SelectData):
    if frame is None:
        raise gr.Error("Load a video first.")
    points = [] if len(points) >= 4 else list(points)
    points.append([float(evt.index[0]), float(evt.index[1])])
    if len(points) == 4:
        try:
            Quad.from_points(points)
        except ValueError:
            gr.Warning("Those corners are too close together. Start again.")
            return draw_roi(frame, []), [], ""
    return draw_roi(frame, points), points, points_to_text(points, scale)


def on_undo(frame, scale, points):
    points = list(points)[:-1] if len(points) < 4 else []
    return draw_roi(frame, points), points, points_to_text(points, scale)


def on_roi_text(frame, scale, text):
    if frame is None or not text.strip():
        return draw_roi(frame, []), []
    try:
        points = Quad.parse(text).scaled(scale).as_array().tolist()
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return draw_roi(frame, points), points


# --- analysis ---------------------------------------------------------------------
def cleanup_old_runs(max_age_s: float = 3 * 3600) -> None:
    runs = WORK_DIR / "runs"
    if not runs.is_dir():
        return
    for d in runs.iterdir():
        if d.is_dir() and time.time() - d.stat().st_mtime > max_age_s:
            shutil.rmtree(d, ignore_errors=True)


def timeline_figure(timeline_csv: str, cfg: Config) -> Figure | None:
    """State band, move confidence and people count as three panels on one time axis."""
    rows = []
    with open(timeline_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["time_s"]), r["state"], float(r["move_confidence"]), int(r["people"])))
    if not rows:
        return None
    t = np.array([r[0] for r in rows])
    conf = np.array([r[2] for r in rows])
    people = np.array([r[3] for r in rows])
    ink, muted, grid, series = "#52514e", "#898781", "#e1e0d9", "#2a78d6"
    fig = Figure(figsize=(9, 4.2), layout="constrained")
    ax_state, ax_conf, ax_people = fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [0.9, 3, 1.6]})
    step = np.median(np.diff(t)) if len(t) > 1 else 0.04
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i][1] != rows[start][1]:
            state = rows[start][1]
            ax_state.axvspan(t[start], t[i - 1] + step, color=STATE_COLORS.get(state, muted), lw=0)
            label = STATE_LABELS.get(state, state)
            if t[i - 1] + step - t[start] > 0.12 * (t[-1] - t[0] + step):
                ax_state.text(t[start] + step, 0.5, label, va="center", fontsize=8.5, fontweight="bold",
                              color="#1a1a19" if state == "IDLE" else "white")  # fmt: skip
            start = i
    ax_state.set_yticks([])
    ax_state.set_title("State", loc="left", fontsize=9, color=ink)

    ax_conf.plot(t, conf, color=series, lw=2, solid_joinstyle="round")
    ax_conf.axhline(cfg.move_confidence_min, color=muted, lw=1)
    ax_conf.text(t[-1], cfg.move_confidence_min + 0.03, f"moving threshold {cfg.move_confidence_min:.2f}",
                 ha="right", fontsize=8, color=muted)  # fmt: skip
    ax_conf.set_ylim(0, 1)
    ax_conf.set_yticks([0, 0.5, 1])
    ax_conf.set_title("Move confidence", loc="left", fontsize=9, color=ink)

    ax_people.step(t, people, where="post", color=series, lw=2)
    top = max(2, int(people.max()))
    ax_people.set_ylim(0, top + 0.5)
    ax_people.set_yticks([0, top])
    ax_people.set_title("People in ROI", loc="left", fontsize=9, color=ink)
    ax_people.set_xlabel("video time (s)", color=muted, fontsize=8.5)

    for ax in (ax_state, ax_conf, ax_people):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors=muted, labelsize=8, length=0)
        if ax is not ax_state:
            ax.grid(axis="y", color=grid, lw=1)
            ax.set_axisbelow(True)
    return fig


def summary_markdown(summary: dict, note: str, truth_report=None) -> str:
    final = summary.get("final_state") or "-"
    tis = summary["time_in_state_s"]
    pct = summary["time_in_state_pct"]

    def row(state: str, label: str) -> str:
        return f"| {label} | {tis.get(state, 0):.1f} s | {pct.get(state, 0):.0f} % |"

    lines = [
        f"### Final state: **{final}**",
        "",
        "| | time | share |",
        "|---|---|---|",
        row("WORKING", "Working"),
        row("STOPPED / FAULT", "Stopped / fault"),
        row("IDLE", "Idle"),
        "",
        f"- **Fault episodes:** {summary['fault_count']}",
        f"- **Availability** (working vs. stopped time, idle excluded): "
        f"{summary['availability_pct'] if summary['availability_pct'] is not None else '-'} %",
        f"- Analysed {summary['frames_analyzed']} frames ({summary['duration_s']} s of video) at "
        f"{summary['processing_fps']} FPS on this server, {summary['frame_size'][0]}x{summary['frame_size'][1]}.",
    ]
    if summary.get("wrong_direction_alerts"):
        lines.append(f"- **Wrong-direction alerts:** {summary['wrong_direction_alerts']}")
    if truth_report is not None:
        lat = truth_report.mean_latency_s
        lines += [
            "",
            "**Against the clip's ground truth:** "
            f"{truth_report.accuracy:.1%} of frames correct, fault detected after "
            f"{lat.get('STOPPED / FAULT', float('nan')):.2f} s, {truth_report.false_fault_alarms} false alarms "
            "(the delay is the state machine's deliberate confirmation window).",
        ]
    if note:
        lines += ["", f"_{note}_"]
    return "\n".join(lines)


def analyze(
    video,
    roi_text,
    sensitivity,
    detect_every,
    fast_mode,
    compensate,
    direction,
    max_seconds,
    progress=gr.Progress(),  # noqa: B008 - Gradio injects the progress tracker here
):
    if not video:
        raise gr.Error("Upload a video first.")
    cleanup_old_runs()
    out = Path(tempfile.mkdtemp(prefix="run_", dir=WORK_DIR / "runs"))
    _, scale, fps, count = read_first_frame(video)

    notes = []
    roi_points = None
    if roi_text and roi_text.strip():
        try:
            roi_points = Quad.parse(roi_text).to_list()
        except ValueError as exc:
            raise gr.Error(f"Escalator corners: {exc}") from exc
    else:
        notes.append("No escalator region was marked, so a central default region was used.")

    stride = 2 if fast_mode else 1
    seconds = min(float(max_seconds), MAX_SECONDS)
    if count and count / fps > seconds:
        notes.append(f"Only the first {seconds:.0f} s were analysed.")
    cfg = Config(
        input_video=video,
        output_video=str(out / "annotated.mp4"),
        events_csv=str(out / "events.csv"),
        timeline_csv=str(out / "timeline.csv"),
        report_json=str(out / "report.json"),
        snapshot_dir=str(out / "snapshots"),
        roi_file=str(out / "roi.json"),
        roi_points=roi_points,
        process_max_width=PROCESS_WIDTH,
        frame_stride=stride,
        max_frames=max(1, math.ceil(seconds * fps / stride)),
        move_confidence_min=float(sensitivity),
        detect_every_n_frames=int(detect_every),
        compensate_camera_motion=bool(compensate),
        expected_direction=direction,
    )

    synthetic = SYNTHETIC.get(file_hash(video))
    if synthetic:
        detector = ReplayDetector.from_json(synthetic["boxes"])
        notes.append(
            "Synthetic clip: person boxes come from the simulator (YOLO does not recognise the drawn figures)."
        )
    else:
        progress(0, desc="Loading the person detector")
        detector = get_yolo()

    total = cfg.max_frames
    last = [0.0]

    def on_progress(index: int, _total: int) -> None:
        now = time.monotonic()
        if now - last[0] > 0.25:
            last[0] = now
            progress(min(1.0, index / stride / total), desc="Analysing")

    try:
        result = run(cfg, detector, progress=on_progress)
    except (OSError, RuntimeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc

    truth = None
    if synthetic:
        truth = evaluate(load_timeline(cfg.timeline_csv), load_intervals(synthetic["ground_truth"]))
        (out / "ground_truth_eval.json").write_text(json.dumps(truth.to_dict(), indent=2))

    events = []
    with open(cfg.events_csv, newline="") as f:
        for r in csv.DictReader(f):
            events.append([r["video_time"], r["event"], r["people"], r["move_confidence"]])
    snapshots = sorted(str(p) for p in (out / "snapshots").glob("*.jpg"))
    files = [cfg.events_csv, cfg.timeline_csv, cfg.report_json]
    return (
        cfg.output_video,
        summary_markdown(result.summary, " ".join(notes), truth),
        timeline_figure(cfg.timeline_csv, cfg),
        events or [["-", "no state changes", "", ""]],
        snapshots,
        files,
    )


# --- layout -------------------------------------------------------------------------
INTRO = f"""
# Escalator Monitor
Detects whether an escalator is **working**, **stopped with people on it (fault)** or **idle**
from ordinary CCTV footage. YOLO finds the people, dense optical flow measures whether the
steps and handrails move once people are masked out, and a hysteresis state machine turns
that into a stable status with an event log.
[Source code and docs]({REPO_URL})

**How to use:** upload a clip from a fixed camera (or pick an example), click the four corners
of the escalator (including both handrails), then press **Analyse**.
"""


def build_demo() -> gr.Blocks:
    examples, labels = prepare_examples()
    with gr.Blocks(title="Escalator Monitor", delete_cache=(3600, 3 * 3600)) as demo:
        gr.Markdown(INTRO)
        frame_state = gr.State(None)
        scale_state = gr.State(1.0)
        points_state = gr.State([])
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                video = gr.Video(label="1. Escalator video (fixed camera)", sources=["upload"], height=320)
                gr.Examples(examples, inputs=[video], label="Examples", example_labels=labels)
                with gr.Accordion("Settings", open=False):
                    sensitivity = gr.Slider(
                        0.15, 0.6, value=0.35, step=0.05, label="Move confidence needed (lower = more sensitive)"
                    )
                    detect_every = gr.Slider(1, 6, value=3, step=1, label="Run the person detector every N frames")
                    fast_mode = gr.Checkbox(False, label="Fast mode (analyse every 2nd frame)")
                    compensate = gr.Checkbox(False, label="Compensate camera shake")
                    direction = gr.Radio(
                        ["any", "up", "down"], value="any", label="Expected direction (alert if it runs the other way)"
                    )
                    max_seconds = gr.Slider(
                        5, MAX_SECONDS, value=min(30, MAX_SECONDS), step=5, label="Analyse at most (seconds)"
                    )
            with gr.Column(scale=1):
                roi_image = gr.Image(
                    label="2. Click the escalator's 4 corners", interactive=False, type="numpy", height=420
                )
                info = gr.Markdown("Upload a clip or pick an example.")
                with gr.Row():
                    undo = gr.Button("Undo point", size="sm")
                    roi_text = gr.Textbox(
                        label="Corners in video pixels (x,y x4)",
                        scale=3,
                        placeholder="filled in by clicking; you can also paste coordinates",
                    )
                analyse = gr.Button("3. Analyse", variant="primary")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                out_video = gr.Video(label="Annotated result", height=420, autoplay=True)
            with gr.Column(scale=2):
                summary = gr.Markdown()
        plot = gr.Plot(label="Timeline")
        with gr.Row(equal_height=False):
            events = gr.Dataframe(
                headers=["video time", "event", "people", "confidence"],
                label="State changes",
                interactive=False,
                wrap=True,
            )
            snapshots = gr.Gallery(label="Fault snapshots", columns=2, height=260)
        files = gr.File(label="Downloads (events, per-frame timeline, JSON report)", file_count="multiple")

        video.change(on_video, [video], [roi_image, frame_state, scale_state, points_state, roi_text, info])
        roi_image.select(on_click, [frame_state, scale_state, points_state], [roi_image, points_state, roi_text])
        undo.click(on_undo, [frame_state, scale_state, points_state], [roi_image, points_state, roi_text])
        roi_text.blur(on_roi_text, [frame_state, scale_state, roi_text], [roi_image, points_state])
        analyse.click(
            analyze,
            [video, roi_text, sensitivity, detect_every, fast_mode, compensate, direction, max_seconds],
            [out_video, summary, plot, events, snapshots, files],
            api_name="analyze",
        )
    return demo


(WORK_DIR / "runs").mkdir(parents=True, exist_ok=True)
demo = build_demo()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=16).launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="emerald"),
    )
