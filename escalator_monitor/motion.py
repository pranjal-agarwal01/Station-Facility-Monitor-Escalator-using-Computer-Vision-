"""Escalator surface motion: optical flow on the ROI turned into a move confidence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .flow import FlowEngine, RegionFlow, background_motion, region_flow
from .geometry import Box, Quad, build_region_masks

MIN_SCORED_PIXELS = 200  # a region needs this many visible pixels to be scored


def smooth_score(
    mag: float, cons: float, mag_gate: float, mag_norm: float, cons_gate: float, cons_norm: float
) -> float:
    """Map (magnitude, consistency) to [0, 1). Zero below either gate, saturating above."""
    if mag < mag_gate or cons < cons_gate:
        return 0.0
    mag_term = mag / (mag + mag_norm)
    cons_term = cons / (cons + (1.0 - cons_norm))
    return float(math.sqrt(mag_term * min(cons_term, 1.0)))


def direction_penalty(vx: float, vy: float, cfg: Config) -> float:
    """1 for clearly vertical motion, 0 for sideways motion (people crossing, pans)."""
    if not cfg.require_vertical_motion:
        return 1.0
    if abs(vy) < 1e-3:
        return 0.0
    ratio = abs(vy) / (abs(vx) + 1e-3)
    if ratio < cfg.vertical_ratio_min:
        return 0.0
    return float(min(1.0, ratio / (cfg.vertical_ratio_min * 2)))


def handrail_agreement(left: RegionFlow, right: RegionFlow, cfg: Config) -> float:
    """Cosine similarity of the two handrail velocities (both rails move together)."""
    if not cfg.require_handrail_agreement:
        return 1.0
    ln = math.hypot(left.vx, left.vy)
    rn = math.hypot(right.vx, right.vy)
    if ln < 1e-3 or rn < 1e-3:
        return 0.0
    dot = (left.vx * right.vx + left.vy * right.vy) / (ln * rn)
    if dot < cfg.direction_dot_min:
        return 0.0
    return float(max(0.0, dot))


@dataclass(frozen=True)
class MotionReading:
    valid: bool = False  # False until two frames have been seen
    handrail_mag: float = 0.0
    handrail_cons: float = 0.0
    steps_mag: float = 0.0
    steps_cons: float = 0.0
    handrail_score: float = 0.0
    steps_score: float = 0.0
    direction_gate: float = 0.0
    handrail_agreement: float = 0.0
    confidence: float = 0.0
    is_moving: bool = False
    vy: float = 0.0  # signed surface speed along the image y axis; < 0 = towards the top
    camera_shift: tuple[float, float] = (0.0, 0.0)


class MotionAnalyzer:
    """Measures escalator surface motion between consecutive frames.

    Flow is computed only on the ROI's bounding box (plus a margin), downscaled
    by ``flow_downscale``. People are masked out so that only the escalator
    surface contributes; handrails and steps are scored separately and fused.
    """

    def __init__(self, cfg: Config, quad: Quad, frame_size: tuple[int, int]):
        self.cfg = cfg
        self.frame_w, self.frame_h = frame_size
        self.flow = FlowEngine(cfg.use_dis_flow)
        self.set_roi(quad)

    def set_roi(self, quad: Quad) -> None:
        self.quad = quad
        self.window = quad.bbox(self.cfg.flow_roi_margin, self.frame_w, self.frame_h)
        x0, y0, x1, y1 = self.window
        s = self.cfg.flow_downscale
        self.size = (max(8, round((x1 - x0) * s)), max(8, round((y1 - y0) * s)))  # (w, h)
        self.scale = (self.size[0] / (x1 - x0), self.size[1] / (y1 - y0))
        self.flow.adapt_to(*self.size)
        shape = (self.size[1], self.size[0])
        self.masks = build_region_masks(quad, self.window, self.scale, shape, self.cfg.handrail_width_frac)
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def _prepare(self, gray: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.window
        crop = gray[y0:y1, x0:x1]
        if (crop.shape[1], crop.shape[0]) != self.size:
            crop = cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA)
        return crop

    def _usable_pixels(self, person_boxes: Sequence[Box]) -> np.ndarray:
        """1 where the escalator surface is visible, 0 under (padded) people."""
        keep = np.ones((self.size[1], self.size[0]), dtype=np.uint8)
        pad = self.cfg.person_mask_padding
        x0, y0 = self.window[:2]
        sx, sy = self.scale
        for bx1, by1, bx2, by2 in person_boxes:
            c1 = max(0, int((bx1 - pad - x0) * sx))
            r1 = max(0, int((by1 - pad - y0) * sy))
            c2 = min(self.size[0], int(math.ceil((bx2 + pad - x0) * sx)))
            r2 = min(self.size[1], int(math.ceil((by2 + pad - y0) * sy)))
            if c2 > c1 and r2 > r1:
                keep[r1:r2, c1:c2] = 0
        return keep

    def update(self, gray: np.ndarray, person_boxes: Sequence[Box] = (), stride: int = 1) -> MotionReading:
        curr = self._prepare(gray)
        prev, self._prev = self._prev, curr
        if prev is None:
            return MotionReading()

        cfg = self.cfg
        flow = self.flow.compute(prev, curr)
        if stride > 1:
            flow /= stride  # keep magnitudes in px per source frame

        keep = self._usable_pixels(person_boxes)
        shift = (0.0, 0.0)
        if cfg.compensate_camera_motion:
            shift = background_motion(flow, self.masks.background & keep)
            if shift != (0.0, 0.0):
                flow -= np.array(shift, dtype=np.float32)

        left_mask = self.masks.left & keep
        right_mask = self.masks.right & keep
        steps_mask = self.masks.steps & keep
        left = region_flow(flow, left_mask)
        right = region_flow(flow, right_mask)
        steps = region_flow(flow, steps_mask)

        # Handrails: average both sides; if one is hidden (crowd), trust the other.
        rails = [r for r in (left, right) if r.visible]
        if len(rails) == 2:
            agreement = handrail_agreement(left, right, cfg)
        else:
            agreement = 1.0 if rails else 0.0
        if rails:
            hr_mag = sum(r.magnitude for r in rails) / len(rails)
            hr_cons = sum(r.consistency for r in rails) / len(rails)
            hr_vx = sum(r.vx for r in rails) / len(rails)
            hr_vy = sum(r.vy for r in rails) / len(rails)
        else:
            hr_mag = hr_cons = hr_vx = hr_vy = 0.0

        direction_gate = direction_penalty(hr_vx, hr_vy, cfg) * agreement
        steps_gate = direction_penalty(steps.vx, steps.vy, cfg)

        hr_pixels = int(np.count_nonzero(left_mask)) + int(np.count_nonzero(right_mask))
        st_pixels = int(np.count_nonzero(steps_mask))
        hr_score = st_score = 0.0
        if hr_pixels > MIN_SCORED_PIXELS:
            hr_score = direction_gate * smooth_score(
                hr_mag,
                hr_cons,
                cfg.handrail_mag_gate,
                cfg.handrail_mag_norm,
                cfg.consistency_gate,
                cfg.consistency_norm,
            )
        if st_pixels > MIN_SCORED_PIXELS:
            st_score = steps_gate * smooth_score(
                steps.magnitude,
                steps.consistency,
                cfg.steps_mag_gate,
                cfg.steps_mag_norm,
                cfg.consistency_gate,
                cfg.consistency_norm,
            )

        if st_pixels <= MIN_SCORED_PIXELS:
            confidence = hr_score
        else:
            confidence = cfg.handrail_weight * hr_score + cfg.steps_weight * st_score
        if max(hr_score, st_score) > cfg.strong_region_score:
            confidence = max(confidence, cfg.strong_region_score)

        weight = hr_score + st_score
        vy = (hr_score * hr_vy + st_score * steps.vy) / weight if weight > 0 else 0.0

        return MotionReading(
            valid=True,
            handrail_mag=hr_mag,
            handrail_cons=hr_cons,
            steps_mag=steps.magnitude,
            steps_cons=steps.consistency,
            handrail_score=hr_score,
            steps_score=st_score,
            direction_gate=direction_gate,
            handrail_agreement=agreement,
            confidence=float(confidence),
            is_moving=confidence >= cfg.move_confidence_min,
            vy=float(vy),
            camera_shift=shift,
        )
