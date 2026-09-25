"""Procedurally rendered escalator clips with exact ground truth.

Used by the test-suite, the benchmark and the demo. The scene is a textured
escalator in perspective: steps and handrails scroll when it runs, people ride
it or walk on it when it is stopped, and the camera can add sensor noise,
brightness flicker and shake. Every frame comes with the true state and the
true person boxes, so the motion + state logic can be measured in isolation
from the person detector.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry import Quad
from .state import State

BELT_W, BELT_H, BELT_LEN = 240, 480, 3840  # belt texture (px); BELT_LEN is a multiple of the step pitch
STEP_PITCH = 32
RAIL_W = 24


@dataclass
class Segment:
    seconds: float
    moving: bool
    people: int = 0

    @property
    def expected(self) -> State:
        if self.moving:
            return State.WORKING
        return State.STOPPED if self.people else State.IDLE


@dataclass
class SceneSpec:
    segments: list[Segment]
    width: int = 640
    height: int = 480
    fps: float = 25.0
    speed: float = 1.6  # surface speed in belt px/frame
    direction: str = "up"  # surface motion in the image: "up" or "down"
    noise: float = 3.0  # sensor noise, grey levels (std)
    flicker: float = 0.0  # brightness modulation amplitude (0.1 = +/-10 %)
    jitter: float = 0.0  # camera shake amplitude in px
    seed: int = 0
    corners: tuple = ((0.40, 0.08), (0.60, 0.08), (0.72, 0.97), (0.28, 0.97))  # TL TR BR BL, frame fractions

    @property
    def total_frames(self) -> int:
        return sum(round(s.seconds * self.fps) for s in self.segments)


@dataclass
class SyntheticFrame:
    index: int  # 1-based
    time: float
    image: np.ndarray
    boxes: list[list[int]]
    state: State


@dataclass
class _Person:
    u: float  # across the belt, 0..1
    v: float  # along the visible belt, px from the top edge
    walk: float  # own speed on the surface, belt px/frame (< 0 walks towards the top)
    size: float  # height relative to the local escalator width
    color: tuple[int, int, int]
    phase: float = field(default=0.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


class SyntheticEscalator:
    def __init__(self, spec: SceneSpec):
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)
        w, h = spec.width, spec.height
        c = np.array(spec.corners, dtype=np.float64) * [w, h]
        self.corners = c
        tl, tr, br, bl = c
        # Outer 8 % of each side is handrail, then a static balustrade, then the steps.
        self.rail_left = np.array([tl, _lerp(tl, tr, 0.08), _lerp(bl, br, 0.08), bl])
        self.rail_right = np.array([_lerp(tr, tl, 0.08), tr, br, _lerp(br, bl, 0.08)])
        self.steps_poly = np.array([_lerp(tl, tr, 0.11), _lerp(tr, tl, 0.11), _lerp(br, bl, 0.11), _lerp(bl, br, 0.11)])
        self._steps_tex = self._make_steps_texture()
        self._rail_tex = self._make_rail_texture()
        self._background = self._make_background()
        self._steps_H = self._homography(BELT_W, self.steps_poly)
        self._rail_H = (self._homography(RAIL_W, self.rail_left), self._homography(RAIL_W, self.rail_right))
        self._masks = [self._poly_mask(p) for p in (self.steps_poly, self.rail_left, self.rail_right)]

    @property
    def quad(self) -> Quad:
        """The ROI a user would draw: the whole escalator including both handrails."""
        return Quad.from_points(self.corners)

    def ground_truth(self) -> list[tuple[float, float, State]]:
        intervals, t = [], 0.0
        for seg in self.spec.segments:
            n = round(seg.seconds * self.spec.fps)
            intervals.append((t, t + n / self.spec.fps, seg.expected))
            t += n / self.spec.fps
        return intervals

    # --- textures -------------------------------------------------------------
    def _make_steps_texture(self) -> np.ndarray:
        fine = cv2.GaussianBlur(self.rng.normal(0, 24, (BELT_LEN, BELT_W)).astype(np.float32), (0, 0), 1.2)
        base = np.full((BELT_LEN, BELT_W), 128, np.float32)
        rows = np.arange(BELT_LEN) % STEP_PITCH
        base[rows < 3] = 60  # gap between steps
        base[(rows >= 3) & (rows < 6)] = 185  # step nose
        base += np.where(np.arange(BELT_W) % 6 < 2, -20.0, 0.0)[None, :]  # cleats
        return np.clip(base + fine, 0, 255).astype(np.uint8)

    def _make_rail_texture(self) -> np.ndarray:
        rail = self.rng.normal(38, 10, (BELT_LEN, RAIL_W)).astype(np.float32)
        rail[self.rng.random((BELT_LEN, RAIL_W)) < 0.03] = 150  # scuffs give the flow something to track
        return np.clip(cv2.GaussianBlur(rail, (0, 0), 1.0), 0, 255).astype(np.uint8)

    def _make_background(self) -> np.ndarray:
        w, h = self.spec.width, self.spec.height
        low = cv2.resize(
            self.rng.normal(0, 1, (h // 24 + 2, w // 24 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC
        )
        grain = cv2.GaussianBlur(self.rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.5)
        grey = 125 + 22 * low + 10 * grain + np.linspace(-15, 20, h, dtype=np.float32)[:, None]
        img = np.dstack([grey * 0.95, grey, grey * 1.05])
        for _ in range(7):  # signs, doors, panels
            x, y = int(self.rng.integers(0, w - 40)), int(self.rng.integers(0, h - 40))
            bw, bh = int(self.rng.integers(20, 90)), int(self.rng.integers(15, 70))
            img[y : y + bh, x : x + bw] = self.rng.integers(40, 230, 3)
        balustrade = self._poly_mask(self.corners).astype(bool)
        img[balustrade] = img[balustrade] * 0.4 + 105
        return np.clip(img, 0, 255).astype(np.uint8)

    def _poly_mask(self, poly: np.ndarray) -> np.ndarray:
        mask = np.zeros((self.spec.height, self.spec.width), np.uint8)
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
        return mask

    @staticmethod
    def _homography(belt_w: int, poly: np.ndarray) -> np.ndarray:
        src = np.float32([[0, 0], [belt_w, 0], [belt_w, BELT_H], [0, BELT_H]])
        return cv2.getPerspectiveTransform(src, poly.astype(np.float32))

    def _belt_window(self, tex: np.ndarray, offset: float) -> tuple[np.ndarray, float]:
        start = int(np.floor(offset))
        rows = (np.arange(BELT_H + 2) + start) % BELT_LEN
        return tex[rows], offset - start

    def _warp(self, tex: np.ndarray, H: np.ndarray, offset: float) -> np.ndarray:
        window, frac = self._belt_window(tex, offset)
        shift = np.array([[1, 0, 0], [0, 1, -frac], [0, 0, 1]], dtype=np.float64)
        return cv2.warpPerspective(window, H @ shift, (self.spec.width, self.spec.height), flags=cv2.INTER_LINEAR)

    # --- people ------------------------------------------------------------------
    def _spawn(self, anywhere: bool, moving: bool) -> _Person:
        rng = self.rng
        # Riders stand or walk with the surface; on a stopped escalator people walk either way.
        walk = float(rng.choice([0.0, 0.0, 0.6, 0.9])) * (self._sign if moving else float(rng.choice([1, -1])))
        velocity = self._sign * (self.spec.speed if moving else 0.0) + walk
        if anywhere or velocity == 0:
            v = rng.uniform(20, BELT_H - 10)
        else:  # enter from the end the person is walking away from
            v = BELT_H + rng.uniform(0, 60) if velocity < 0 else -rng.uniform(0, 60)
        grey = int(rng.integers(20, 110))
        color = (grey + int(rng.integers(0, 60)), grey + int(rng.integers(0, 40)), grey + int(rng.integers(0, 60)))
        return _Person(
            float(rng.uniform(0.25, 0.75)),
            float(v),
            float(walk),
            float(rng.uniform(1.1, 1.4)),
            color,
            float(rng.uniform(0, 6.28)),
        )

    def _foot_and_scale(self, p: _Person) -> tuple[np.ndarray, float]:
        pts = np.float32([[[p.u * BELT_W, p.v]], [[0, p.v]], [[BELT_W, p.v]]])
        foot, left, right = cv2.perspectiveTransform(pts, self._steps_H)[:, 0]
        return foot, float(np.linalg.norm(right - left))

    def _draw_person(self, img: np.ndarray, p: _Person, t: int) -> list[int] | None:
        (fx, fy), width = self._foot_and_scale(p)
        h = p.size * width
        w = 0.36 * h
        fx += 1.2 * np.sin(p.phase + t * 0.15)  # sway
        x1, y1, x2, y2 = int(fx - w / 2), int(fy - h), int(fx + w / 2), int(fy)
        if x2 <= 0 or y2 <= 0 or x1 >= img.shape[1] or y1 >= img.shape[0]:
            return None
        c = tuple(int(v) for v in p.color)
        legs = (int(fx - w * 0.28), int(fy - h * 0.45), int(fx + w * 0.28), y2)
        cv2.rectangle(img, legs[:2], legs[2:], tuple(max(0, v - 25) for v in c), -1)
        cv2.ellipse(img, (int(fx), int(fy - h * 0.62)), (int(w / 2), int(h * 0.22)), 0, 0, 360, c, -1)
        cv2.circle(img, (int(fx), int(fy - h * 0.9)), max(2, int(h * 0.1)), (60, 80, 120), -1)
        return [max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)]

    # --- rendering -------------------------------------------------------------
    @property
    def _sign(self) -> int:
        return -1 if self.spec.direction == "up" else 1  # image-space direction of the surface

    def frames(self) -> Iterator[SyntheticFrame]:
        spec = self.spec
        cv2.setRNGSeed(spec.seed)
        offset = 0.0
        index = 0
        people: list[_Person] = []
        noise = np.empty((spec.height, spec.width, 3), np.int16)
        for seg in spec.segments:
            people = [self._spawn(anywhere=True, moving=seg.moving) for _ in range(seg.people)]
            for _ in range(round(seg.seconds * spec.fps)):
                index += 1
                surface = spec.speed if seg.moving else 0.0
                offset -= self._sign * surface  # window offset moves opposite to the content
                img = self._background.copy()
                for tex, H, mask in (
                    (self._steps_tex, self._steps_H, self._masks[0]),
                    (self._rail_tex, self._rail_H[0], self._masks[1]),
                    (self._rail_tex, self._rail_H[1], self._masks[2]),
                ):
                    warped = self._warp(tex, H, offset)
                    img[mask > 0] = warped[mask > 0][:, None]

                boxes = []
                for i, p in enumerate(people):
                    p.v += self._sign * surface + p.walk
                    if p.v < -80 or p.v > BELT_H + 80:
                        people[i] = p = self._spawn(anywhere=False, moving=seg.moving)
                for p in sorted(people, key=lambda q: q.v):
                    if -10 <= p.v <= BELT_H + 10:
                        box = self._draw_person(img, p, index)
                        if box is not None:
                            boxes.append(box)

                if spec.flicker:
                    gain = 1.0 + spec.flicker * np.sin(2 * np.pi * index / (1.7 * spec.fps))
                    img = cv2.convertScaleAbs(img, alpha=gain)
                if spec.jitter:
                    dx, dy = self.rng.uniform(-spec.jitter, spec.jitter, 2)
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    img = cv2.warpAffine(img, M, (spec.width, spec.height), borderMode=cv2.BORDER_REFLECT)
                    boxes = [[int(b[0] + dx), int(b[1] + dy), int(b[2] + dx), int(b[3] + dy)] for b in boxes]
                if spec.noise:
                    cv2.randn(noise, 0, spec.noise)
                    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                yield SyntheticFrame(index, (index - 1) / spec.fps, img, boxes, seg.expected)


def write_clip(spec: SceneSpec, path: str | Path) -> dict[str, str]:
    """Render ``spec`` to ``path`` plus sidecars: ``.boxes.json`` (person boxes per
    frame), ``.gt.csv`` (true state intervals) and ``.roi.json``."""
    from .video import VideoWriter

    path = Path(path)
    scene = SyntheticEscalator(spec)
    writer = VideoWriter(path, spec.fps, (spec.width, spec.height))
    boxes: dict[int, list[list[int]]] = {}
    try:
        for f in scene.frames():
            writer.write(f.image)
            if f.boxes:
                boxes[f.index] = f.boxes
    finally:
        writer.close()
    stem = path.with_suffix("")
    out = {
        "video": str(path),
        "boxes": f"{stem}.boxes.json",
        "ground_truth": f"{stem}.gt.csv",
        "roi": f"{stem}.roi.json",
    }
    Path(out["boxes"]).write_text(json.dumps(boxes), encoding="utf-8")
    with open(out["ground_truth"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "state"])
        for start, end, state in scene.ground_truth():
            w.writerow([f"{start:.3f}", f"{end:.3f}", state.value])
    scene.quad.save(out["roi"])
    return out


DEMO_SEGMENTS = [
    Segment(6, moving=True, people=3),
    Segment(8, moving=False, people=3),
    Segment(6, moving=True, people=2),
    Segment(7, moving=False, people=0),
]
