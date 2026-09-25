"""Frame-by-frame escalator monitoring: detection -> motion -> state. No I/O, no GUI."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Config
from .detector import PersonDetector
from .geometry import Quad
from .motion import MotionAnalyzer, MotionReading
from .state import EscalatorStateMachine, State, WindowStats
from .tracking import IoUTracker, Track

DIRECTION_EPS = 0.05  # px/frame; slower median surface speed = direction unknown


@dataclass
class FrameResult:
    index: int  # frame number in the source (1-based)
    time: float  # seconds since the start of the source
    state: State
    previous: State | None  # the state before this frame, if it changed
    people: list[Track]
    motion: MotionReading
    window: WindowStats
    direction: str  # "up" / "down" (image space) / "" when unknown
    wrong_direction: bool = False

    @property
    def changed(self) -> bool:
        return self.previous is not None


@dataclass
class StageTimer:
    """Accumulated wall time per pipeline stage, for benchmarking."""

    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, stage: str, seconds: float) -> None:
        self.totals[stage] = self.totals.get(stage, 0.0) + seconds
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def mean_ms(self) -> dict[str, float]:
        return {k: 1000 * v / self.counts[k] for k, v in self.totals.items()}


class EscalatorMonitor:
    def __init__(
        self, cfg: Config, quad: Quad, frame_size: tuple[int, int], detector: PersonDetector, fps: float = 25.0
    ):
        self.cfg = cfg
        self.fps = fps
        self.frame_w, self.frame_h = frame_size
        self.detector = detector
        self.tracker = IoUTracker(iou_threshold=0.3, max_misses=2)
        self.motion = MotionAnalyzer(cfg, quad, frame_size)
        self.states = EscalatorStateMachine(cfg)
        self.timer = StageTimer()
        self._vy: deque[float] = deque(maxlen=cfg.window_size)
        self._processed = 0
        self.set_roi(quad)

    def set_roi(self, quad: Quad) -> None:
        self.quad = quad
        self.motion.set_roi(quad)
        self.tracker.reset()
        margin = self.cfg.detect_roi_margin
        self._detect_region = quad.bbox(margin, self.frame_w, self.frame_h) if margin >= 0 else None

    def _in_roi(self, det) -> bool:
        return self.quad.contains(*det.center) or self.quad.contains(*det.foot)

    def _direction(self, reading: MotionReading) -> str:
        if reading.is_moving:
            self._vy.append(reading.vy)
        if len(self._vy) < 5:
            return ""
        median = float(np.median(self._vy))
        if abs(median) < DIRECTION_EPS:
            return ""
        return "up" if median < 0 else "down"

    def process(self, frame: np.ndarray, index: int | None = None, timestamp: float | None = None) -> FrameResult:
        cfg = self.cfg
        self._processed += 1
        index = self._processed if index is None else index

        t0 = time.perf_counter()
        if (self._processed - 1) % cfg.detect_every_n_frames == 0:
            detections = self.detector.detect(frame, index, self._detect_region)
            tracks = self.tracker.update([d for d in detections if self._in_roi(d)])
            self.timer.add("detect", time.perf_counter() - t0)
        else:
            tracks = self.tracker.update(None)

        t1 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        reading = self.motion.update(gray, [t.det.box for t in tracks], stride=cfg.frame_stride)
        self.timer.add("motion", time.perf_counter() - t1)

        before = self.states.state
        state = self.states.update(reading.is_moving, bool(tracks), reading.confidence)
        direction = self._direction(reading)
        if state != State.WORKING:
            direction = ""
        wrong = (
            cfg.expected_direction != "any"
            and state == State.WORKING
            and direction != ""
            and direction != cfg.expected_direction
        )
        return FrameResult(
            index=index,
            time=timestamp if timestamp is not None else index / self.fps,
            state=state,
            previous=before if state != before else None,
            people=tracks,
            motion=reading,
            window=self.states.stats(),
            direction=direction,
            wrong_direction=wrong,
        )
