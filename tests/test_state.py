import pytest

from escalator_monitor.config import Config
from escalator_monitor.state import EscalatorStateMachine, State


def feed(sm, n, moving, people, score=None):
    score = (0.7 if moving else 0.0) if score is None else score
    for _ in range(n):
        state = sm.update(moving, people, score)
    return state


def test_full_cycle():
    sm = EscalatorStateMachine(Config())
    assert feed(sm, 5, True, True) == State.INITIALIZING  # still warming up
    assert feed(sm, 40, True, True) == State.WORKING
    assert feed(sm, 45, False, True) == State.STOPPED
    assert feed(sm, 45, True, True) == State.WORKING
    assert feed(sm, 60, False, False) == State.IDLE


def test_empty_moving_escalator_is_working():
    sm = EscalatorStateMachine(Config())
    assert feed(sm, 45, True, False) == State.WORKING


def test_hysteresis_keeps_working_between_thresholds():
    sm = EscalatorStateMachine(Config())
    feed(sm, 45, True, True)
    # ~33 % moving frames: below enter (45 %) but above exit (25 %) -> stays WORKING
    for i in range(300):
        sm.update(i % 3 == 0, True, 0.25 if i % 3 == 0 else 0.1)
        assert sm.state == State.WORKING


def test_riders_leaving_before_stop_is_not_a_fault():
    sm = EscalatorStateMachine(Config())
    feed(sm, 45, True, True)
    feed(sm, 10, True, False)  # last riders step off
    history = [sm.update(False, False, 0.0) for _ in range(45)]
    assert State.STOPPED not in history
    assert history[-1] == State.IDLE


def test_stopped_from_idle_needs_people():
    sm = EscalatorStateMachine(Config())
    assert feed(sm, 45, False, False) == State.IDLE
    assert feed(sm, 45, False, True) == State.STOPPED


@pytest.mark.parametrize(
    "text,state",
    [("working", State.WORKING), ("STOPPED / FAULT", State.STOPPED), ("fault", State.STOPPED), ("Idle", State.IDLE)],
)
def test_state_parse(text, state):
    assert State.parse(text) == state
