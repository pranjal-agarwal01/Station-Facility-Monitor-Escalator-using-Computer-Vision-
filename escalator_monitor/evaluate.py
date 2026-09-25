"""Score a run's per-frame timeline against hand-labelled ground truth.

Ground truth is a CSV of intervals::

    start_s,end_s,state
    0,42.5,WORKING
    42.5,95,STOPPED
    95,130,IDLE

Label your own footage this way (a spreadsheet is fine) to measure accuracy,
fault-detection latency and false alarms on the cameras you care about.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .state import State

CLASSES = (State.WORKING, State.STOPPED, State.IDLE)
Interval = tuple[float, float, State]


def load_intervals(path: str | Path) -> list[Interval]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    intervals = []
    for row in rows:
        start = float(row.get("start_s", row.get("start", 0)))
        end = float(row.get("end_s", row.get("end", 0)))
        if end <= start:
            raise ValueError(f"Interval ends before it starts: {row}")
        intervals.append((start, end, State.parse(row["state"])))
    return sorted(intervals, key=lambda i: i[0])


def load_timeline(path: str | Path) -> list[tuple[float, State]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [(float(r["time_s"]), State.parse(r["state"])) for r in csv.DictReader(f)]


def truth_at(intervals: list[Interval], t: float) -> State | None:
    for start, end, state in intervals:
        if start <= t < end:
            return state
    return None


@dataclass
class EvalReport:
    frames: int = 0
    accuracy: float = 0.0
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    mean_latency_s: dict[str, float] = field(default_factory=dict)
    missed_transitions: int = 0
    false_fault_alarms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def evaluate(timeline: list[tuple[float, State]], intervals: list[Interval]) -> EvalReport:
    """Frame-level metrics plus how quickly each true state change was picked up.

    Frames still INITIALIZING or outside every labelled interval are ignored.
    """
    report = EvalReport()
    names = [c.value for c in CLASSES]
    confusion = {t: {p: 0 for p in names} for t in names}
    for t, pred in timeline:
        truth = truth_at(intervals, t)
        if truth is None or pred == State.INITIALIZING:
            continue
        confusion[truth.value][pred.value] += 1
    report.confusion = confusion
    report.frames = sum(sum(row.values()) for row in confusion.values())
    if report.frames:
        report.accuracy = sum(confusion[n][n] for n in names) / report.frames
    for n in names:
        tp = confusion[n][n]
        predicted = sum(confusion[t][n] for t in names)
        actual = sum(confusion[n].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report.per_class[n] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}

    # Latency: time from each true change until the prediction first agrees.
    latencies: dict[str, list[float]] = {}
    for i, (start, end, state) in enumerate(intervals):
        if i == 0 or intervals[i - 1][2] == state:
            continue
        hit = next((t for t, pred in timeline if start <= t < end and pred == state), None)
        entry = {"at_s": start, "to": state.value, "latency_s": None if hit is None else round(hit - start, 3)}
        report.transitions.append(entry)
        if hit is None:
            report.missed_transitions += 1
        else:
            latencies.setdefault(state.value, []).append(hit - start)
    report.mean_latency_s = {k: round(sum(v) / len(v), 3) for k, v in latencies.items()}

    # False alarms: predicted fault episodes that never overlap a true fault.
    episode_start = None
    for t, pred in timeline + [(float("inf"), State.INITIALIZING)]:
        if pred == State.STOPPED and episode_start is None:
            episode_start = t
        elif pred != State.STOPPED and episode_start is not None:
            overlaps = any(s < t and e > episode_start and st == State.STOPPED for s, e, st in intervals)
            report.false_fault_alarms += 0 if overlaps else 1
            episode_start = None
    return report


def format_report(report: EvalReport) -> str:
    lines = [
        f"Frames scored: {report.frames}",
        f"Accuracy:      {report.accuracy:.1%}",
        "",
        f"{'state':<18}{'precision':>10}{'recall':>10}{'f1':>8}{'frames':>9}",
    ]
    for name, m in report.per_class.items():
        lines.append(f"{name:<18}{m['precision']:>10.1%}{m['recall']:>10.1%}{m['f1']:>8.2f}{m['support']:>9}")
    lines.append("")
    for name, latency in report.mean_latency_s.items():
        lines.append(f"Mean time to detect {name}: {latency:.2f}s")
    lines.append(f"Missed state changes: {report.missed_transitions}")
    lines.append(f"False fault alarms:   {report.false_fault_alarms}")
    return "\n".join(lines)
