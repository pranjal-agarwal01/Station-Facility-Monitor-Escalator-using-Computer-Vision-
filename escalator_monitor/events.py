"""Outputs other than video: event log, per-frame timeline, webhook alerts, run summary."""

from __future__ import annotations

import csv
import json
import logging
import queue
import threading
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from .pipeline import FrameResult
from .state import State

log = logging.getLogger(__name__)

EVENT_COLUMNS = [
    "frame",
    "timestamp_wall",
    "video_time",
    "event",
    "details",
    "snapshot_path",
    "from_state",
    "to_state",
    "move_confidence",
    "people",
]
TIMELINE_COLUMNS = [
    "frame",
    "time_s",
    "state",
    "move_confidence",
    "is_moving",
    "people",
    "handrail_score",
    "steps_score",
    "moving_ratio",
    "people_ratio",
    "direction",
]


def format_time(seconds: float) -> str:
    """``H:MM:SS.mmm``"""
    ms = int(round(max(0.0, seconds) * 1000))
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    return f"{hours}:{minutes:02d}:{rest // 1000:02d}.{rest % 1000:03d}"


class CsvLog:
    """CSV file with a header, flushed after every row so it can be tailed live."""

    def __init__(self, path: str | Path | None, columns: list[str]):
        self.path = str(path) if path else ""
        self._file = None
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "w", newline="", encoding="utf-8")  # noqa: SIM115 - kept open for the run
            self._writer = csv.writer(self._file)
            self._writer.writerow(columns)

    def write(self, row: list[Any], flush: bool = True) -> None:
        if self._file is None:
            return
        self._writer.writerow(row)
        if flush:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def timeline_row(r: FrameResult) -> list[Any]:
    m = r.motion
    return [
        r.index,
        f"{r.time:.3f}",
        r.state.value,
        f"{m.confidence:.3f}",
        int(m.is_moving),
        len(r.people),
        f"{m.handrail_score:.3f}",
        f"{m.steps_score:.3f}",
        f"{r.window.moving_ratio:.3f}",
        f"{r.window.people_ratio:.3f}",
        r.direction,
    ]


def event_details(r: FrameResult, prev_duration: float) -> str:
    m = r.motion
    return (
        f"people={len(r.people)} hr_mag={m.handrail_mag:.2f} hr_score={m.handrail_score:.2f} "
        f"st_score={m.steps_score:.2f} dir={m.direction_gate:.2f} agree={m.handrail_agreement:.2f} "
        f"conf={m.confidence:.2f} moving={r.window.moving_ratio:.2f} prev_dur={prev_duration:.1f}s"
    )


class WebhookNotifier:
    """POSTs JSON events from a background thread so a slow endpoint never
    stalls video processing."""

    def __init__(self, url: str, timeout: float = 3.0, max_pending: int = 100):
        self.url = url
        self.timeout = timeout
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self._thread = threading.Thread(target=self._run, daemon=True) if url else None
        if self._thread:
            self._thread.start()

    def send(self, payload: dict[str, Any]) -> None:
        if not self._thread:
            return
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            log.warning("Webhook queue full; dropping event %s", payload.get("event"))

    def _run(self) -> None:
        while True:
            payload = self._queue.get()
            if payload is None:
                return
            try:
                req = urllib.request.Request(
                    self.url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp.read()
            except Exception as exc:
                log.warning("Webhook failed: %s", exc)

    def close(self, timeout: float = 5.0) -> None:
        if self._thread:
            self._queue.put(None)
            self._thread.join(timeout)
            self._thread = None


class SessionStats:
    """Time spent in each state, fault episodes and transitions over a run."""

    def __init__(self) -> None:
        self.durations: dict[str, float] = defaultdict(float)
        self.transitions = 0
        self.faults: list[dict[str, Any]] = []
        self.wrong_direction_alerts = 0
        self.frames = 0
        self._last: FrameResult | None = None

    def update(self, r: FrameResult, snapshot: str = "") -> None:
        self.frames += 1
        if self._last is not None:
            self.durations[self._last.state.value] += max(0.0, r.time - self._last.time)
        if r.changed:
            self.transitions += 1
            if r.previous == State.STOPPED and self.faults and self.faults[-1]["end_s"] is None:
                self._close_fault(r.time)
            if r.state == State.STOPPED:
                self.faults.append({"start_s": round(r.time, 3), "end_s": None, "frame": r.index, "snapshot": snapshot})
        self._last = r

    def _close_fault(self, t: float) -> None:
        fault = self.faults[-1]
        fault["end_s"] = round(t, 3)
        fault["duration_s"] = round(t - fault["start_s"], 3)

    def finish(self, frame_period: float) -> None:
        if self._last is None:
            return
        end = self._last.time + frame_period
        self.durations[self._last.state.value] += frame_period
        if self.faults and self.faults[-1]["end_s"] is None:
            self._close_fault(end)
            self.faults[-1]["ongoing"] = True

    def summary(self) -> dict[str, Any]:
        total = sum(self.durations.values())
        working = self.durations.get(State.WORKING.value, 0.0)
        stopped = self.durations.get(State.STOPPED.value, 0.0)
        return {
            "frames_analyzed": self.frames,
            "duration_s": round(total, 2),
            "time_in_state_s": {k: round(v, 2) for k, v in self.durations.items()},
            "time_in_state_pct": {k: round(100 * v / total, 1) for k, v in self.durations.items()} if total else {},
            # Availability ignores IDLE time: an empty, stopped escalator is not a failure.
            "availability_pct": round(100 * working / (working + stopped), 1) if working + stopped > 0 else None,
            "fault_count": len(self.faults),
            "faults": self.faults,
            "transitions": self.transitions,
            "wrong_direction_alerts": self.wrong_direction_alerts,
            "final_state": self._last.state.value if self._last else None,
        }
