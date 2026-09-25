"""Person detectors. YOLO is the default; the others exist for testing and replay."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Config
from .geometry import Box
from .tracking import Detection

log = logging.getLogger(__name__)


class PersonDetector(Protocol):
    def detect(self, frame: np.ndarray, frame_idx: int, region: Box | None = None) -> list[Detection]:
        """Return people in ``frame`` (full-frame coordinates).

        ``region`` is a hint: only people inside it matter, so a detector may
        crop to it (which also helps with small, distant people).
        """
        ...


class NullDetector:
    """Detects nobody. Motion-only mode: STOPPED and IDLE can't be told apart."""

    def detect(self, frame, frame_idx, region=None) -> list[Detection]:
        return []


class ReplayDetector:
    """Replays boxes recorded per frame, e.g. from the synthetic clip generator."""

    def __init__(self, boxes_by_frame: dict[int, list]):
        self.boxes = {int(k): v for k, v in boxes_by_frame.items()}

    @classmethod
    def from_json(cls, path: str | Path) -> ReplayDetector:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def detect(self, frame, frame_idx, region=None) -> list[Detection]:
        return [
            Detection(*map(int, b[:4]), conf=float(b[4]) if len(b) > 4 else 1.0) for b in self.boxes.get(frame_idx, [])
        ]


class YoloPersonDetector:
    """Ultralytics YOLO restricted to the COCO ``person`` class."""

    def __init__(self, model: str = "yolo11n.pt", conf: float = 0.35, imgsz: int = 640, device: str = ""):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Person detection needs Ultralytics: pip install ultralytics  (or run with --detector none)"
            ) from exc
        log.info("Loading %s", model)
        self.model = YOLO(model)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or None

    def detect(self, frame: np.ndarray, frame_idx: int, region: Box | None = None) -> list[Detection]:
        x0 = y0 = 0
        image = frame
        if region is not None:
            x0, y0, x1, y1 = region
            image = frame[y0:y1, x0:x1]
        results = self.model.predict(
            image, classes=[0], conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )
        people = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for (bx1, by1, bx2, by2), score in zip(xyxy, confs):
                people.append(Detection(int(bx1) + x0, int(by1) + y0, int(bx2) + x0, int(by2) + y0, float(score)))
        return people


def build_detector(spec: str, cfg: Config) -> PersonDetector:
    """``yolo`` (default), ``none``, or ``replay:<boxes.json>``."""
    spec = (spec or "yolo").strip()
    if spec == "yolo":
        return YoloPersonDetector(cfg.yolo_model, cfg.person_conf_threshold, cfg.yolo_imgsz, cfg.device)
    if spec == "none":
        return NullDetector()
    if spec.startswith("replay:"):
        return ReplayDetector.from_json(spec.split(":", 1)[1])
    raise ValueError(f"Unknown detector {spec!r}; use 'yolo', 'none' or 'replay:<boxes.json>'")
