import cv2
import numpy as np
import pytest

from escalator_monitor.config import Config
from escalator_monitor.flow import RegionFlow, background_motion, region_flow
from escalator_monitor.motion import MotionAnalyzer, direction_penalty, handrail_agreement, smooth_score
from escalator_monitor.synthetic import SceneSpec, Segment, SyntheticEscalator


def test_smooth_score_gates_and_saturates():
    args = dict(mag_gate=0.1, mag_norm=0.3, cons_gate=0.55, cons_norm=0.85)
    assert smooth_score(0.05, 0.9, **args) == 0.0  # below magnitude gate
    assert smooth_score(0.5, 0.4, **args) == 0.0  # below consistency gate
    low, high = smooth_score(0.2, 0.9, **args), smooth_score(2.0, 0.9, **args)
    assert 0 < low < high < 1


def test_direction_penalty_prefers_vertical_motion():
    cfg = Config()
    assert direction_penalty(0.0, 1.0, cfg) == 1.0
    assert direction_penalty(1.0, 0.2, cfg) == 0.0
    assert direction_penalty(1.0, 0.2, Config(require_vertical_motion=False)) == 1.0


def test_handrails_must_agree():
    cfg = Config()
    up = RegionFlow(1, 1, 0.0, -1.0, 100)
    down = RegionFlow(1, 1, 0.0, 1.0, 100)
    assert handrail_agreement(up, up, cfg) == pytest.approx(1.0)
    assert handrail_agreement(up, down, cfg) == 0.0


def test_region_flow_statistics():
    flow = np.zeros((20, 20, 2), np.float32)
    flow[..., 1] = -0.8
    mask = np.ones((20, 20), np.uint8)
    r = region_flow(flow, mask)
    assert r.magnitude == pytest.approx(0.8)
    assert r.consistency == pytest.approx(1.0, abs=1e-3)
    assert (r.vx, r.vy) == pytest.approx((0.0, -0.8))

    rng = np.random.default_rng(0)
    noisy = rng.normal(0, 1, (20, 20, 2)).astype(np.float32)
    assert region_flow(noisy, mask).consistency < 0.3
    assert not region_flow(flow, np.zeros_like(mask)).visible


def test_background_motion_is_median():
    flow = np.zeros((30, 30, 2), np.float32)
    flow[..., 0] = 1.5
    flow[:5, :5] = 40  # outliers
    assert background_motion(flow, np.ones((30, 30), np.uint8)) == pytest.approx((1.5, 0.0))


def _moving_fraction(spec: SceneSpec, cfg: Config, boxes=None) -> float:
    scene = SyntheticEscalator(spec)
    analyzer = MotionAnalyzer(cfg, scene.quad, (spec.width, spec.height))
    flags = []
    for f in scene.frames():
        r = analyzer.update(cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY), boxes if boxes is not None else f.boxes)
        if r.valid:
            flags.append(r.is_moving)
    return float(np.mean(flags))


@pytest.mark.parametrize("direction", ["up", "down"])
def test_moving_escalator_is_detected(direction):
    spec = SceneSpec([Segment(1.5, True, 2)], width=480, height=360, direction=direction, seed=1)
    assert _moving_fraction(spec, Config()) > 0.95


def test_stopped_escalator_with_walking_people_is_still():
    spec = SceneSpec([Segment(2, False, 4)], width=480, height=360, seed=2)
    assert _moving_fraction(spec, Config()) < 0.05


def test_person_masking_hides_the_whole_escalator():
    spec = SceneSpec([Segment(1, True, 0)], width=480, height=360, seed=3)
    everything = [(0, 0, 480, 360)]
    assert _moving_fraction(spec, Config(), boxes=everything) == 0.0


def test_camera_shake_is_compensated():
    spec = SceneSpec([Segment(2, False, 0)], width=640, height=480, jitter=1.5, seed=4)
    assert _moving_fraction(spec, Config()) > 0.1  # shake alone looks like motion...
    assert _moving_fraction(spec, Config(compensate_camera_motion=True)) < 0.03  # ...until compensated
    moving = SceneSpec([Segment(2, True, 0)], width=640, height=480, jitter=1.5, seed=4)
    assert _moving_fraction(moving, Config(compensate_camera_motion=True)) > 0.9
