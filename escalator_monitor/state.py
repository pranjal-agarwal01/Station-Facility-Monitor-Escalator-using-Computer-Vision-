"""Hysteresis state machine: WORKING / STOPPED / IDLE from a rolling window of evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from .config import Config


class State(str, Enum):
    INITIALIZING = "INITIALIZING"
    WORKING = "WORKING"
    STOPPED = "STOPPED / FAULT"
    IDLE = "IDLE"

    @classmethod
    def parse(cls, text: str) -> State:
        key = text.strip().upper().replace("_", " ")
        aliases = {
            "WORKING": cls.WORKING,
            "MOVING": cls.WORKING,
            "RUNNING": cls.WORKING,
            "STOPPED / FAULT": cls.STOPPED,
            "STOPPED": cls.STOPPED,
            "FAULT": cls.STOPPED,
            "IDLE": cls.IDLE,
            "INITIALIZING": cls.INITIALIZING,
        }
        if key not in aliases:
            raise ValueError(f"Unknown state {text!r}")
        return aliases[key]

    @property
    def short(self) -> str:
        return {"STOPPED / FAULT": "STOPPED"}.get(self.value, self.value)


@dataclass(frozen=True)
class WindowStats:
    moving_ratio: float = 0.0  # share of recent frames judged "moving"
    people_ratio: float = 0.0  # share of recent frames with someone in the ROI
    mean_score: float = 0.0  # mean move confidence over the window
    frames: int = 0
    recent_people_ratio: float = 0.0  # people ratio over the newest third of the window


class EscalatorStateMachine:
    """Decides the escalator state from per-frame motion and occupancy.

    * WORKING - the surface moves consistently (with or without passengers).
    * STOPPED / FAULT - people are on it but the surface is still.
    * IDLE - nothing moves and nobody is there (e.g. stopped to save energy).

    Entering and leaving WORKING use different thresholds so the state does not
    flicker around a single cut-off.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = State.INITIALIZING
        self._moving: deque[bool] = deque(maxlen=cfg.window_size)
        self._people: deque[bool] = deque(maxlen=cfg.window_size)
        self._scores: deque[float] = deque(maxlen=cfg.window_size)

    def reset(self) -> None:
        self.state = State.INITIALIZING
        self._moving.clear()
        self._people.clear()
        self._scores.clear()

    def stats(self) -> WindowStats:
        n = len(self._moving)
        if n == 0:
            return WindowStats()
        k = min(n, max(1, self.cfg.window_size // 3))
        recent = list(self._people)[-k:]
        return WindowStats(sum(self._moving) / n, sum(self._people) / n, sum(self._scores) / n, n, sum(recent) / k)

    def update(self, is_moving: bool, people_present: bool, score: float) -> State:
        self._moving.append(bool(is_moving))
        self._people.append(bool(people_present))
        self._scores.append(float(score))
        if len(self._moving) < max(1, self.cfg.window_size // 3):
            return self.state

        cfg = self.cfg
        w = self.stats()
        mr, pr, avg = w.moving_ratio, w.people_ratio, w.mean_score
        state = self.state
        new = state

        if state in (State.INITIALIZING, State.IDLE):
            if mr >= cfg.enter_working_ratio or avg > cfg.enter_working_score:
                new = State.WORKING
            elif (
                pr >= cfg.stopped_people_ratio
                and w.recent_people_ratio >= cfg.stopped_people_ratio  # people there now, not just earlier
                and (1.0 - mr) >= cfg.enter_stopped_ratio
                and avg < cfg.stopped_score_max
            ):
                new = State.STOPPED
            elif pr < cfg.idle_people_ratio and state == State.INITIALIZING:
                new = State.IDLE
        elif state == State.WORKING:
            if mr < cfg.exit_working_ratio and avg < cfg.fault_score_max:
                # Judge occupancy on recent frames only: riders who stepped off
                # just before the stop should not turn it into a fault.
                new = State.STOPPED if w.recent_people_ratio >= cfg.idle_people_ratio else State.IDLE
        elif state == State.STOPPED:
            if mr >= cfg.enter_working_ratio or avg > cfg.resume_working_score:
                new = State.WORKING
            elif pr < cfg.idle_people_ratio:
                new = State.IDLE

        if pr < cfg.idle_people_ratio and mr < cfg.exit_working_ratio and avg < cfg.fault_score_max:
            new = State.IDLE

        self.state = new
        return new
