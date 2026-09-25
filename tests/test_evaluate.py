import pytest

from escalator_monitor.evaluate import evaluate, load_intervals
from escalator_monitor.state import State

W, S, IDLE = State.WORKING, State.STOPPED, State.IDLE
TRUTH = [(0.0, 10.0, W), (10.0, 20.0, S), (20.0, 30.0, IDLE)]


def timeline(fn, fps=10):
    return [(i / fps, fn(i / fps)) for i in range(300)]


def test_perfect_prediction():
    report = evaluate(timeline(lambda t: W if t < 10 else S if t < 20 else IDLE), TRUTH)
    assert report.accuracy == pytest.approx(1.0)
    assert report.mean_latency_s == {"STOPPED / FAULT": 0.0, "IDLE": 0.0}
    assert report.false_fault_alarms == 0


def test_delayed_prediction_reports_latency():
    report = evaluate(timeline(lambda t: W if t < 11.5 else S if t < 21 else IDLE), TRUTH)
    assert report.mean_latency_s["STOPPED / FAULT"] == pytest.approx(1.5)
    assert report.mean_latency_s["IDLE"] == pytest.approx(1.0)
    assert report.accuracy == pytest.approx(1 - 25 / 300)


def test_false_alarm_and_missed_change():
    pred = timeline(lambda t: S if 3 <= t < 4 else W)
    report = evaluate(pred, TRUTH)
    assert report.false_fault_alarms == 1
    assert report.missed_transitions == 2


def test_initializing_frames_are_ignored():
    report = evaluate([(0.0, State.INITIALIZING), (1.0, W)], TRUTH)
    assert report.frames == 1


def test_load_intervals_accepts_aliases(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text("start_s,end_s,state\n0,5,running\n5,9,fault\n")
    assert load_intervals(path) == [(0.0, 5.0, W), (5.0, 9.0, S)]
