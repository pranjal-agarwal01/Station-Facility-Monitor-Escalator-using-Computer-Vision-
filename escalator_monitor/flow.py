"""Dense optical flow and the per-region statistics computed from it."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

MIN_REGION_PIXELS = 40  # below this a region is treated as not visible
MIN_MOVING_PIXELS = 10  # below this there is no usable direction estimate


class FlowEngine:
    """DIS optical flow (fast, part of core OpenCV >= 4) with a Farneback fallback."""

    def __init__(self, use_dis: bool = True):
        self._dis = None
        self.name = "Farneback"
        if use_dis and hasattr(cv2, "DISOpticalFlow_create"):
            self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            self.name = "DIS"
        elif use_dis:
            log.warning("DIS optical flow unavailable in this OpenCV build; using Farneback")

    def adapt_to(self, width: int, height: int) -> None:
        """The FAST preset estimates flow at 1/4 of the input resolution, which
        washes out thin handrails on small inputs; refine to 1/2 there."""
        if self._dis is not None:
            self._dis.setFinestScale(1 if min(width, height) < 320 else 2)

    def compute(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        if self._dis is not None:
            return self._dis.calc(prev_gray, curr_gray, None)
        return cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )


@dataclass(frozen=True)
class RegionFlow:
    magnitude: float = 0.0  # mean |v| over the whole region (px/frame)
    consistency: float = 0.0  # |sum v| / sum |v| over moving pixels: 1 = all aligned
    vx: float = 0.0  # mean velocity of the moving pixels
    vy: float = 0.0
    pixels: int = 0  # visible pixels in the region

    @property
    def visible(self) -> bool:
        return self.pixels >= MIN_REGION_PIXELS


def region_flow(flow: np.ndarray, mask: np.ndarray, mag_floor: float = 0.1) -> RegionFlow:
    """Summarise the flow vectors inside ``mask`` (non-zero = use the pixel)."""
    valid = mask > 0
    n = int(np.count_nonzero(valid))
    if n < MIN_REGION_PIXELS:
        return RegionFlow(pixels=n)
    vx = flow[..., 0][valid]
    vy = flow[..., 1][valid]
    mag = np.hypot(vx, vy)
    mean_mag = float(mag.mean())
    strong = mag > mag_floor
    k = int(np.count_nonzero(strong))
    if k < MIN_MOVING_PIXELS:
        return RegionFlow(magnitude=mean_mag, pixels=n)
    sum_x = float(vx[strong].sum())
    sum_y = float(vy[strong].sum())
    consistency = float(np.hypot(sum_x, sum_y) / (float(mag[strong].sum()) + 1e-6))
    return RegionFlow(mean_mag, consistency, sum_x / k, sum_y / k, n)


def background_motion(flow: np.ndarray, mask: np.ndarray, min_pixels: int = 200) -> tuple[float, float]:
    """Median flow of static scenery: the camera's own motion. (0, 0) if unknown."""
    valid = mask > 0
    if int(np.count_nonzero(valid)) < min_pixels:
        return 0.0, 0.0
    return float(np.median(flow[..., 0][valid])), float(np.median(flow[..., 1][valid]))
