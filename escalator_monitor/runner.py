"""Runs the monitor over a video source and writes every output file."""

from __future__ import annotations

import itertools
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np

from .config import Config, source_name
from .detector import PersonDetector
from .events import (
    EVENT_COLUMNS,
    TIMELINE_COLUMNS,
    CsvLog,
    SessionStats,
    WebhookNotifier,
    event_details,
    format_time,
    timeline_row,
)
from .geometry import Quad
from .pipeline import EscalatorMonitor, FrameResult
from .render import Renderer
from .state import State
from .video import Frame, VideoSource, VideoWriter

log = logging.getLogger(__name__)

HookAction = Union[str, Quad, None]  # "quit", a new ROI, or nothing
FrameHook = Callable[[Frame, "np.ndarray | None", FrameResult], HookAction]
RoiPicker = Callable[[np.ndarray], "Quad | None"]


@dataclass
class RunResult:
    summary: dict[str, Any]
    quad: Quad
    output_video: str = ""
    events_csv: str = ""
    timeline_csv: str = ""
    report_json: str = ""
    snapshots: list[str] = field(default_factory=list)


def save_snapshot(image: np.ndarray, directory: str, label: str, frame_idx: int) -> str:
    Path(directory).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    path = str(Path(directory) / f"{stamp}_f{frame_idx}_{safe}.jpg")
    cv2.imwrite(path, image)
    return path


def resolve_roi(cfg: Config, frame: np.ndarray, scale: float, picker: RoiPicker | None = None) -> Quad:
    """Pick the ROI for a run, in processing coordinates.

    Order: ``roi_points`` from the config, then the saved ROI file, then the
    interactive picker (if a GUI is available), then a central fallback.
    Saved and configured points are in source-video pixels.
    """
    h, w = frame.shape[:2]
    if cfg.roi_points:
        return Quad.from_points(cfg.roi_points).scaled(scale)
    roi_file = cfg.resolved_roi_file()
    if Path(roi_file).exists():
        try:
            quad = Quad.load(roi_file)
            log.info("Loaded ROI from %s", roi_file)
            return quad.scaled(scale)
        except (ValueError, KeyError, OSError) as exc:
            log.warning("Could not load %s (%s); selecting the ROI again", roi_file, exc)
    quad = picker(frame) if picker else None
    if quad is None:
        log.warning("No ROI configured; using a central fallback region. Set roi_points or pick one interactively.")
        return Quad.default_for(w, h)
    quad.scaled(1 / scale).save(roi_file)
    log.info("Saved ROI to %s", roi_file)
    return quad


def run(
    cfg: Config,
    detector: PersonDetector,
    roi: Quad | None = None,
    picker: RoiPicker | None = None,
    hook: FrameHook | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> RunResult:
    """Process ``cfg.input_video`` end to end.

    ``roi`` is in processing coordinates (after ``process_max_width``). ``hook``
    sees every annotated frame and can stop the run or replace the ROI.
    """
    source = VideoSource(cfg.input_video, cfg.process_max_width, cfg.frame_stride)
    frames = iter(source)
    first = next(frames, None)
    if first is None:
        source.close()
        raise RuntimeError(f"No frames could be read from {cfg.input_video}")

    size = (source.width, source.height)
    quad = roi or resolve_roi(cfg, first.image, source.scale, picker)
    log.info(
        "Source %s: %dx%d @ %.2f fps%s",
        cfg.input_video,
        *size,
        source.fps,
        f", {source.total_frames} frames" if source.total_frames else " (live)",
    )

    monitor = EscalatorMonitor(cfg, quad, size, detector, fps=source.fps)
    renderer = Renderer(cfg, quad, size)
    writer = VideoWriter(cfg.output_video, source.output_fps, size) if cfg.output_video else None
    events = CsvLog(cfg.events_csv, EVENT_COLUMNS)
    timeline = CsvLog(cfg.timeline_csv, TIMELINE_COLUMNS)
    webhook = WebhookNotifier(cfg.webhook_url, cfg.webhook_timeout)
    stats = SessionStats()
    snapshots: list[str] = []
    total = source.total_frames
    state_since = first.time
    wrong_active = False
    t_start = time.perf_counter()

    def emit(r: FrameResult, event: str, details: str, snapshot: str, from_state: str, to_state: str) -> None:
        wall = datetime.now().isoformat(timespec="seconds")
        events.write(
            [
                r.index,
                wall,
                format_time(r.time),
                event,
                details,
                snapshot,
                from_state,
                to_state,
                f"{r.motion.confidence:.3f}",
                len(r.people),
            ]
        )
        log.info("[%s] %s  (conf=%.2f, people=%d)", format_time(r.time), event, r.motion.confidence, len(r.people))
        webhook.send(
            {
                "event": event,
                "from_state": from_state,
                "to_state": to_state,
                "source": source_name(cfg.input_video),
                "frame": r.index,
                "video_time": format_time(r.time),
                "wall_time": wall,
                "people": len(r.people),
                "move_confidence": round(r.motion.confidence, 3),
                "snapshot": snapshot,
            }
        )

    try:
        for frame in itertools.chain([first], frames):
            result = monitor.process(frame.image, frame.index, frame.time)
            elapsed = time.perf_counter() - t_start
            proc_fps = (stats.frames + 1) / elapsed if elapsed > 0 else 0.0
            draw_now = writer is not None or hook is not None
            annotated = renderer.draw(frame.image, result, total, proc_fps) if draw_now else None

            snapshot = ""
            if result.changed:
                prev_duration = result.time - state_since
                state_since = result.time
                if cfg.save_fault_snapshots and result.state == State.STOPPED:
                    evidence = annotated if annotated is not None else renderer.draw(frame.image, result, total)
                    snapshot = save_snapshot(evidence, cfg.snapshot_dir, result.state.value, result.index)
                    snapshots.append(snapshot)
                emit(
                    result,
                    f"{result.previous.value} -> {result.state.value}",
                    event_details(result, prev_duration),
                    snapshot,
                    result.previous.value,
                    result.state.value,
                )
            if result.wrong_direction and not wrong_active:
                stats.wrong_direction_alerts += 1
                emit(
                    result,
                    "WRONG DIRECTION",
                    f"expected={cfg.expected_direction} observed={result.direction}",
                    "",
                    result.state.value,
                    result.state.value,
                )
            wrong_active = result.wrong_direction

            stats.update(result, snapshot)
            timeline.write(timeline_row(result), flush=result.index % 50 == 0)
            if writer is not None:
                writer.write(annotated)
            if progress is not None:
                progress(result.index, total)
            if hook is not None:
                action = hook(frame, annotated, result)
                if action == "quit":
                    break
                if isinstance(action, Quad):
                    quad = action
                    monitor.set_roi(quad)
                    renderer.set_roi(quad)
                    quad.scaled(1 / source.scale).save(cfg.resolved_roi_file())
            if cfg.max_frames and stats.frames >= cfg.max_frames:
                break
    except KeyboardInterrupt:
        log.info("Interrupted; writing outputs")
    finally:
        source.close()
        if writer is not None:
            writer.close()
        events.close()
        timeline.close()
        webhook.close()

    elapsed = time.perf_counter() - t_start
    stats.finish(frame_period=source.stride / source.fps)
    summary = {
        "source": str(cfg.input_video),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **stats.summary(),
        "processing_fps": round(stats.frames / elapsed, 2) if elapsed > 0 else None,
        "stage_ms": {k: round(v, 2) for k, v in monitor.timer.mean_ms().items()},
        "flow_algorithm": monitor.motion.flow.name,
        "frame_size": list(size),
        "roi": quad.to_list(),
        "config": cfg.to_dict(),
    }
    if cfg.report_json:
        Path(cfg.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.report_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return RunResult(summary, quad, cfg.output_video, cfg.events_csv, cfg.timeline_csv, cfg.report_json, snapshots)
