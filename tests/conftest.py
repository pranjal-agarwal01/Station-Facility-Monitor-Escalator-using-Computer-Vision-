from __future__ import annotations

import pytest

from escalator_monitor.config import Config
from escalator_monitor.detector import ReplayDetector
from escalator_monitor.pipeline import EscalatorMonitor
from escalator_monitor.synthetic import SceneSpec, Segment, SyntheticEscalator

# Short version of the demo: run, fault, run again, empty and stopped.
SHORT_SEGMENTS = [
    Segment(3, moving=True, people=3),
    Segment(4, moving=False, people=3),
    Segment(3, moving=True, people=2),
    Segment(4, moving=False, people=0),
]


def run_scene(spec: SceneSpec, cfg: Config | None = None, detector=None):
    """Run the monitor over a synthetic scene; returns (frames, results)."""
    cfg = cfg or Config()
    scene = SyntheticEscalator(spec)
    boxes: dict[int, list] = {}
    detector = detector if detector is not None else ReplayDetector(boxes)
    monitor = EscalatorMonitor(cfg, scene.quad, (spec.width, spec.height), detector, fps=spec.fps)
    frames, results = [], []
    for f in scene.frames():
        if isinstance(detector, ReplayDetector):
            detector.boxes[f.index] = f.boxes
        results.append(monitor.process(f.image, f.index, f.time))
        frames.append(f)
    return frames, results


@pytest.fixture
def short_spec() -> SceneSpec:
    return SceneSpec(SHORT_SEGMENTS, width=480, height=360, seed=7)
