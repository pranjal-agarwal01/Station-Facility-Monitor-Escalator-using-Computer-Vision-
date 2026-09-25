"""Reproducible accuracy and speed benchmark on synthetic scenes.

Person boxes come from the simulator (or YOLO, if requested), so the accuracy
numbers measure the motion analysis and state machine, not the detector.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from .config import Config
from .detector import PersonDetector, ReplayDetector, YoloPersonDetector
from .evaluate import EvalReport, evaluate
from .pipeline import EscalatorMonitor
from .render import Renderer
from .synthetic import DEMO_SEGMENTS, SceneSpec, Segment, SyntheticEscalator

BUSY_SEGMENTS = [Segment(6, True, 6), Segment(8, False, 6), Segment(6, True, 5), Segment(7, False, 0)]

SCENARIOS: dict[str, tuple[SceneSpec, dict]] = {
    "clean": (SceneSpec(DEMO_SEGMENTS, noise=2.0), {}),
    "sensor noise": (SceneSpec(DEMO_SEGMENTS, noise=8.0), {}),
    "flicker": (SceneSpec(DEMO_SEGMENTS, flicker=0.10), {}),
    "crowded": (SceneSpec(BUSY_SEGMENTS), {}),
    "moving down": (SceneSpec(DEMO_SEGMENTS, direction="down"), {}),
    "slow belt": (SceneSpec(DEMO_SEGMENTS, speed=0.8), {}),
    "camera shake": (SceneSpec(DEMO_SEGMENTS, jitter=1.5), {}),
    "camera shake + comp.": (SceneSpec(DEMO_SEGMENTS, jitter=1.5), {"compensate_camera_motion": True}),
}


@dataclass
class ScenarioResult:
    name: str
    report: EvalReport
    frames: int
    stage_ms: dict[str, float]
    fps: float


def run_scenario(
    name: str, spec: SceneSpec, cfg: Config, detector: PersonDetector | None = None, render: bool = False
) -> ScenarioResult:
    scene = SyntheticEscalator(spec)
    truth_boxes: dict[int, list] = {}  # filled as frames are rendered
    detector = detector or ReplayDetector(truth_boxes)
    size = (spec.width, spec.height)
    monitor = EscalatorMonitor(cfg, scene.quad, size, detector, fps=spec.fps)
    renderer = Renderer(cfg, scene.quad, size) if render else None
    timeline = []
    busy = 0.0
    for f in scene.frames():
        if isinstance(detector, ReplayDetector):
            detector.boxes[f.index] = f.boxes
        t0 = time.perf_counter()
        r = monitor.process(f.image, f.index, f.time)
        if renderer is not None:
            t1 = time.perf_counter()
            renderer.draw(f.image, r, spec.total_frames)
            monitor.timer.add("render", time.perf_counter() - t1)
        busy += time.perf_counter() - t0
        timeline.append((r.time, r.state))
    report = evaluate(timeline, scene.ground_truth())
    return ScenarioResult(name, report, len(timeline), monitor.timer.mean_ms(), len(timeline) / busy)


def accuracy_suite(cfg: Config | None = None, seeds: tuple[int, ...] = (0, 1, 2)) -> list[ScenarioResult]:
    cfg = cfg or Config()
    results = []
    for name, (spec, overrides) in SCENARIOS.items():
        runs = [run_scenario(name, replace(spec, seed=s), replace(cfg, **overrides)) for s in seeds]
        results.append(_merge(name, runs))
    return results


def _merge(name: str, runs: list[ScenarioResult]) -> ScenarioResult:
    """Average several seeds of one scenario into a single row."""
    first = runs[0].report
    merged = EvalReport(
        frames=sum(r.report.frames for r in runs),
        accuracy=sum(r.report.accuracy * r.report.frames for r in runs) / max(1, sum(r.report.frames for r in runs)),
        per_class=first.per_class,
        missed_transitions=sum(r.report.missed_transitions for r in runs),
        false_fault_alarms=sum(r.report.false_fault_alarms for r in runs),
    )
    lat: dict[str, list[float]] = {}
    for r in runs:
        for k, v in r.report.mean_latency_s.items():
            lat.setdefault(k, []).append(v)
    merged.mean_latency_s = {k: round(sum(v) / len(v), 2) for k, v in lat.items()}
    stage = {k: sum(r.stage_ms.get(k, 0) for r in runs) / len(runs) for k in runs[0].stage_ms}
    return ScenarioResult(name, merged, sum(r.frames for r in runs), stage, sum(r.fps for r in runs) / len(runs))


def speed_suite(
    cfg: Config,
    yolo_model: str | None = None,
    sizes: tuple[tuple[int, int], ...] = ((640, 360), (1280, 720), (1920, 1080)),
) -> list[dict]:
    """Per-stage latency at common CCTV resolutions (optionally with real YOLO inference)."""
    detector = (
        YoloPersonDetector(yolo_model, cfg.person_conf_threshold, cfg.yolo_imgsz, cfg.device) if yolo_model else None
    )
    rows = []
    for w, h in sizes:
        spec = SceneSpec([Segment(4, True, 3), Segment(4, False, 3)], width=w, height=h)
        res = run_scenario(f"{w}x{h}", spec, cfg, detector=detector, render=True)
        per_frame = sum(
            res.stage_ms.get(k, 0) * (1 / cfg.detect_every_n_frames if k == "detect" else 1) for k in res.stage_ms
        )
        rows.append(
            {
                "resolution": f"{w}x{h}",
                **{f"{k}_ms": round(v, 2) for k, v in res.stage_ms.items()},
                "pipeline_fps": round(1000 / per_frame, 1) if per_frame else None,
            }
        )
    return rows


def format_accuracy(results: list[ScenarioResult]) -> str:
    lines = [
        "| Scenario | Frame accuracy | Fault detected after | Restart detected after | Idle detected after "
        "| False fault alarms | Missed changes |",
        "|---|---|---|---|---|---|---|",
    ]

    def fmt(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f} s"

    for r in results:
        lat = r.report.mean_latency_s
        lines.append(
            f"| {r.name} | {r.report.accuracy:.1%} | {fmt(lat.get('STOPPED / FAULT'))} | {fmt(lat.get('WORKING'))} | "
            f"{fmt(lat.get('IDLE'))} | {r.report.false_fault_alarms} | {r.report.missed_transitions} |"
        )
    return "\n".join(lines)


def format_speed(rows: list[dict]) -> str:
    keys = [k for k in rows[0] if k != "resolution"]
    lines = ["| Resolution | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for row in rows:
        lines.append(f"| {row['resolution']} | " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)
