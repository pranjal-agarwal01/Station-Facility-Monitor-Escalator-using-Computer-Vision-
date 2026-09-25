"""ROI geometry: the escalator quadrilateral and the region masks derived from it.

The ROI is a quadrilateral whose left and right edges run along the handrails.
It is split into three regions that are analysed separately:

    TL ── iTL ─────────── iTR ── TR
    │ L  │     steps      │  R  │
    BL ── iBL ─────────── iBR ── BR

``L`` and ``R`` are handrail strips, each ``handrail_width_frac`` of the width.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 (x2/y2 exclusive)


def order_corners(points) -> np.ndarray:
    """Order four points clockwise on screen, starting from the top-left one.

    Makes the ROI independent of the click order, as long as the quadrilateral
    is roughly upright (handrails running top-to-bottom in the image).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


@dataclass(frozen=True)
class Quad:
    """Escalator ROI: four corners ordered TL, TR, BR, BL in pixel coordinates."""

    points: tuple[tuple[float, float], ...]

    @classmethod
    def from_points(cls, points) -> Quad:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape != (4, 2):
            raise ValueError(f"ROI needs exactly four (x, y) points, got shape {pts.shape}")
        ordered = order_corners(pts)
        if cv2.contourArea(ordered.astype(np.float32)) < 64:
            raise ValueError("ROI is degenerate: its area is too small")
        return cls(tuple((float(x), float(y)) for x, y in ordered))

    @classmethod
    def parse(cls, text: str) -> Quad:
        """Parse ``"x1,y1,x2,y2,x3,y3,x4,y4"`` (any of ``, ; space`` as separators)."""
        values = [float(v) for v in re.split(r"[,;\s]+", text.strip()) if v]
        if len(values) != 8:
            raise ValueError(f"Expected 8 numbers for the ROI, got {len(values)}: {text!r}")
        return cls.from_points(np.array(values).reshape(4, 2))

    @classmethod
    def default_for(cls, width: int, height: int) -> Quad:
        """Central fallback ROI used when none was configured or picked."""
        w, h = width, height
        return cls.from_points([[w * 0.25, h * 0.20], [w * 0.75, h * 0.20], [w * 0.75, h * 0.95], [w * 0.25, h * 0.95]])

    # ------------------------------------------------------------------------
    def as_array(self) -> np.ndarray:
        return np.array(self.points, dtype=np.float64)

    def as_int(self) -> np.ndarray:
        return np.round(self.as_array()).astype(np.int32)

    def to_list(self) -> list[list[int]]:
        return self.as_int().tolist()

    def scaled(self, factor: float) -> Quad:
        return Quad(tuple((x * factor, y * factor) for x, y in self.points))

    def area(self) -> float:
        return float(cv2.contourArea(self.as_array().astype(np.float32)))

    def contains(self, x: float, y: float) -> bool:
        contour = self.as_array().astype(np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0

    def bbox(self, margin: float = 0.0, width: int | None = None, height: int | None = None) -> Box:
        """Bounding box grown by ``margin`` x its size on each side, clipped to the frame."""
        pts = self.as_array()
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
        x1, y1 = int(np.floor(x1 - dx)), int(np.floor(y1 - dy))
        x2, y2 = int(np.ceil(x2 + dx)) + 1, int(np.ceil(y2 + dy)) + 1
        if width is not None:
            x1, x2 = max(0, x1), min(width, x2)
        if height is not None:
            y1, y2 = max(0, y1), min(height, y2)
        return x1, y1, x2, y2

    def regions(self, width_frac: float) -> dict[str, np.ndarray]:
        """Polygons for the left handrail, right handrail and steps area."""
        tl, tr, br, bl = self.as_array()
        inner_tl = tl + (tr - tl) * width_frac
        inner_bl = bl + (br - bl) * width_frac
        inner_tr = tr + (tl - tr) * width_frac
        inner_br = br + (bl - br) * width_frac
        return {
            "left": np.array([tl, inner_tl, inner_bl, bl]),
            "right": np.array([inner_tr, tr, br, inner_br]),
            "steps": np.array([inner_tl, inner_tr, inner_br, inner_bl]),
        }

    # ------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"points": self.to_list(), "saved_at": datetime.now().isoformat(timespec="seconds")}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Quad:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_points(data["points"])


def rasterize(polygon: np.ndarray, shape: tuple[int, int], offset=(0.0, 0.0), scale=(1.0, 1.0)) -> np.ndarray:
    """Fill ``polygon`` (frame coordinates) into a ``shape`` mask of a scaled crop."""
    pts = (np.asarray(polygon, dtype=np.float64) - np.asarray(offset)) * np.asarray(scale)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


@dataclass
class RegionMasks:
    """0/1 masks for one analysis window: a crop of the frame at some scale."""

    left: np.ndarray
    right: np.ndarray
    steps: np.ndarray
    roi: np.ndarray
    background: np.ndarray  # window pixels clearly outside the ROI


def build_region_masks(
    quad: Quad, window: Box, scale: tuple[float, float], shape: tuple[int, int], width_frac: float
) -> RegionMasks:
    offset = window[:2]
    polys = quad.regions(width_frac)
    roi = rasterize(quad.as_array(), shape, offset, scale)
    grow = max(3, int(round(0.02 * max(shape))) | 1)
    background = 1 - cv2.dilate(roi, np.ones((grow, grow), np.uint8))
    return RegionMasks(
        left=rasterize(polys["left"], shape, offset, scale),
        right=rasterize(polys["right"], shape, offset, scale),
        steps=rasterize(polys["steps"], shape, offset, scale),
        roi=roi,
        background=background,
    )
