"""Overlay drawing for the annotated output video."""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .geometry import Quad, rasterize
from .pipeline import FrameResult
from .state import State

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR fill colour and text colour per state.
STATE_STYLE: dict[State, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    State.WORKING: ((0, 160, 0), (255, 255, 255)),
    State.STOPPED: ((0, 0, 205), (255, 255, 255)),
    State.IDLE: ((0, 190, 255), (20, 20, 20)),
    State.INITIALIZING: ((120, 120, 120), (255, 255, 255)),
}
MOVING_TINT = (0, 220, 0)
STILL_TINT = (0, 100, 220)
PERSON_COLOR = (0, 220, 0)


def blend_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color, alpha: float) -> None:
    """Blend a solid rectangle into ``img`` in place, clipped to the image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1 - alpha, 0, dst=roi)


class Renderer:
    def __init__(self, cfg: Config, quad: Quad, frame_size: tuple[int, int]):
        self.cfg = cfg
        self.w, self.h = frame_size
        self.ui = float(np.clip(min(self.w, self.h) / 720, 0.5, 2.5))
        self.thick = max(1, round(1.6 * self.ui))
        self.set_roi(quad)

    def set_roi(self, quad: Quad) -> None:
        self.quad = quad
        x0, y0, x1, y1 = quad.bbox(0.0, self.w, self.h)
        self.box = (x0, y0, x1, y1)
        shape = (y1 - y0, x1 - x0)
        polys = quad.regions(self.cfg.handrail_width_frac)
        rails = rasterize(polys["left"], shape, (x0, y0)) | rasterize(polys["right"], shape, (x0, y0))
        self.rail_mask = rails.astype(bool)
        self.steps_mask = rasterize(polys["steps"], shape, (x0, y0)).astype(bool)
        self.region_mask = (rails | self.steps_mask).astype(np.uint8)
        self._layers: dict[tuple[bool, bool], np.ndarray] = {}

    # ------------------------------------------------------------------------
    def draw(self, frame: np.ndarray, r: FrameResult, total_frames: int = 0, proc_fps: float = 0.0) -> np.ndarray:
        out = frame.copy()
        fill, _ = STATE_STYLE[r.state]
        if r.motion.valid:
            self._tint_regions(out, r)
        self._draw_people(out, r)
        cv2.polylines(out, [self.quad.as_int()], True, fill, self.thick + 1, cv2.LINE_AA)
        tl = self.quad.as_int()[0]
        cv2.putText(
            out,
            "Escalator ROI",
            (int(tl[0]), max(12, int(tl[1] - 8 * self.ui))),
            FONT,
            0.55 * self.ui,
            fill,
            self.thick,
            cv2.LINE_AA,
        )
        self._draw_badge(out, r)
        if self.cfg.show_debug_overlay:
            self._draw_panel(out, r, total_frames, proc_fps)
        if total_frames > 0:
            bar = int(self.w * min(1.0, r.index / total_frames))
            cv2.rectangle(out, (0, self.h - max(3, int(4 * self.ui))), (bar, self.h), fill, -1)
        return out

    def _tint_layer(self, rails_moving: bool, steps_moving: bool) -> np.ndarray:
        key = (rails_moving, steps_moving)
        if key not in self._layers:
            layer = np.zeros((*self.rail_mask.shape, 3), np.uint8)
            layer[self.rail_mask] = MOVING_TINT if rails_moving else STILL_TINT
            layer[self.steps_mask] = MOVING_TINT if steps_moving else STILL_TINT
            self._layers[key] = layer
        return self._layers[key]

    def _tint_regions(self, out: np.ndarray, r: FrameResult) -> None:
        x0, y0, x1, y1 = self.box
        roi = out[y0:y1, x0:x1]
        layer = self._tint_layer(r.motion.handrail_score > 0.3, r.motion.steps_score > 0.3)
        blended = cv2.addWeighted(layer, 0.14, roi, 0.86, 0)
        roi[...] = cv2.copyTo(blended, self.region_mask, roi.copy())  # blend inside the ROI only

    def _draw_people(self, out: np.ndarray, r: FrameResult) -> None:
        for track in r.people:
            d = track.det
            cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), PERSON_COLOR, self.thick)
            if d.conf < 1.0:
                cv2.putText(
                    out,
                    f"{d.conf:.2f}",
                    (d.x1, max(10, d.y1 - 4)),
                    FONT,
                    0.45 * self.ui,
                    PERSON_COLOR,
                    self.thick,
                    cv2.LINE_AA,
                )

    def _draw_badge(self, out: np.ndarray, r: FrameResult) -> None:
        labels = [(r.state.value, *STATE_STYLE[r.state])]
        if r.wrong_direction:
            labels.append(("WRONG DIRECTION", (0, 0, 205), (255, 255, 255)))
        y = int(15 * self.ui)
        scale = 0.85 * self.ui
        pad = int(14 * self.ui)
        for text, fill, fg in labels:
            (tw, th), _ = cv2.getTextSize(text, FONT, scale, self.thick + 1)
            x1 = (self.w - tw) // 2 - pad
            x2, y2 = x1 + tw + 2 * pad, y + th + 2 * pad
            blend_rect(out, x1, y, x2, y2, fill, 0.88)
            cv2.rectangle(out, (max(0, x1), y), (min(self.w - 1, x2), y2), (255, 255, 255), max(1, self.thick - 1))
            cv2.putText(out, text, (x1 + pad, y2 - pad), FONT, scale, fg, self.thick + 1, cv2.LINE_AA)
            y = y2 + int(8 * self.ui)

    def _draw_panel(self, out: np.ndarray, r: FrameResult, total_frames: int, proc_fps: float) -> None:
        m, cfg = r.motion, self.cfg
        white, dim, good, blue = (255, 255, 255), (200, 200, 200), (180, 255, 180), (255, 210, 200)
        minutes, seconds = divmod(r.time, 60)
        frame_txt = f"Frame {r.index}/{total_frames}" if total_frames > 0 else f"Frame {r.index}"
        lines = [
            (f"{frame_txt}  Time {int(minutes):02d}:{seconds:05.2f}", white),
            (f"People in ROI: {len(r.people)}  (window {r.window.people_ratio:.0%})", white),
            (
                f"Handrail mag={m.handrail_mag:.2f} cons={m.handrail_cons:.2f} score={m.handrail_score:.2f}",
                good if m.handrail_score > 0.3 else dim,
            ),
            (
                f"Steps    mag={m.steps_mag:.2f} cons={m.steps_cons:.2f} score={m.steps_score:.2f}",
                good if m.steps_score > 0.3 else dim,
            ),
            (f"Direction gate {m.direction_gate:.2f}  rail agreement {m.handrail_agreement:.2f}", blue),
            (
                f"Move confidence {m.confidence:.2f} (need {cfg.move_confidence_min:.2f})  moving {r.window.moving_ratio:.0%}",
                white,
            ),
        ]
        extra = []
        if r.direction:
            extra.append(f"Surface moving {r.direction} (image)")
        if cfg.compensate_camera_motion:
            extra.append(f"camera shift {m.camera_shift[0]:+.2f},{m.camera_shift[1]:+.2f}")
        if proc_fps > 0:
            extra.append(f"{proc_fps:.1f} FPS")
        if extra:
            lines.append(("  ".join(extra), (170, 170, 170)))

        scale = 0.5 * self.ui
        line_h = int(24 * self.ui)
        width = max(cv2.getTextSize(t, FONT, scale, self.thick)[0][0] for t, _ in lines) + int(28 * self.ui)
        height = line_h * len(lines) + int(14 * self.ui)
        x1 = int(12 * self.ui)
        y1 = max(0, self.h - height - int(16 * self.ui))
        blend_rect(out, x1, y1, x1 + width, y1 + height, (15, 15, 15), 0.8)
        for i, (text, color) in enumerate(lines):
            cv2.putText(
                out, text, (x1 + int(14 * self.ui), y1 + line_h * (i + 1)), FONT, scale, color, self.thick, cv2.LINE_AA
            )
