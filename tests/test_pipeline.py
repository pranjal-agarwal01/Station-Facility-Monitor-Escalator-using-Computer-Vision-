from dataclasses import replace

from conftest import run_scene

from escalator_monitor.config import Config
from escalator_monitor.detector import NullDetector
from escalator_monitor.evaluate import evaluate
from escalator_monitor.state import State
from escalator_monitor.synthetic import SceneSpec, Segment, SyntheticEscalator


def transitions(results):
    return [r.state for r in results if r.changed and r.state != State.INITIALIZING]


def test_demo_cycle_is_recognised(short_spec):
    frames, results = run_scene(short_spec)
    assert transitions(results) == [State.WORKING, State.STOPPED, State.WORKING, State.IDLE]
    report = evaluate([(r.time, r.state) for r in results], SyntheticEscalator(short_spec).ground_truth())
    assert report.accuracy > 0.7  # short segments: the ~1.3 s confirmation delay dominates
    assert report.false_fault_alarms == 0
    assert report.missed_transitions == 0
    assert report.mean_latency_s["STOPPED / FAULT"] < 2.0


def test_noisy_flickering_scene(short_spec):
    _, results = run_scene(replace(short_spec, noise=8.0, flicker=0.1, seed=11))
    assert transitions(results) == [State.WORKING, State.STOPPED, State.WORKING, State.IDLE]


def test_direction_and_wrong_direction_alert():
    spec = SceneSpec([Segment(3, True, 1)], width=480, height=360, direction="down", seed=5)
    _, results = run_scene(spec, Config(expected_direction="up"))
    assert results[-1].direction == "down"
    assert results[-1].wrong_direction
    _, ok = run_scene(spec, Config(expected_direction="down"))
    assert not any(r.wrong_direction for r in ok)


def test_motion_only_mode_cannot_see_faults():
    spec = SceneSpec([Segment(2, True, 2), Segment(3, False, 3)], width=480, height=360, seed=6)
    _, results = run_scene(spec, detector=NullDetector())
    assert transitions(results) == [State.WORKING, State.IDLE]
