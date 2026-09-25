"""Desktop preview window and ROI picker (OpenCV HighGUI).

Only used by the CLI when a display is available; the rest of the package runs
headless.
"""

from __future__ import annotations

import logging
import os
import sys

import cv2
import numpy as np

from .geometry import Quad
from .pipeline import FrameResult
from .runner import HookAction, save_snapshot
from .video import Frame

log = logging.getLogger(__name__)

WINDOW = "Escalator Monitor"
PICKER_WINDOW = "Click 4 corners (TL, TR, BR, BL), then ENTER"
KEY_ENTER, KEY_ESC, KEY_BACKSPACE = 13, 27, 8


def gui_available() -> bool:
    """True if OpenCV can open a window here (display present, non-headless build)."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


def _fit_window(name: str, w: int, h: int, max_w: int, max_h: int) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    scale = min(max_w / w, max_h / h, 1.0)
    cv2.resizeWindow(name, max(320, int(w * scale)), max(240, int(h * scale)))


class QuadPicker:
    """Click the four escalator corners. ENTER confirms, BACKSPACE undoes, ESC cancels.

    With a WINDOW_NORMAL window OpenCV reports clicks in image coordinates, so
    large frames can be shown scaled down without any conversion.
    """

    LABELS = ("TL", "TR", "BR", "BL")

    def __init__(self, frame: np.ndarray, max_w: int = 1280, max_h: int = 720):
        self.frame = frame
        self.max_w, self.max_h = max_w, max_h
        self.points: list[list[int]] = []

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append([x, y])

    def _render(self) -> np.ndarray:
        img = self.frame.copy()
        ui = max(1.0, min(img.shape[:2]) / 720)
        for i, (x, y) in enumerate(self.points):
            cv2.circle(img, (x, y), int(6 * ui), (0, 255, 0), -1)
            cv2.putText(img, self.LABELS[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, (0, 255, 0), 2)
        if len(self.points) >= 2:
            pts = np.array(self.points, np.int32)
            cv2.polylines(img, [pts], len(self.points) == 4, (0, 255, 255), 2)
        hint = (
            "ENTER to confirm, BACKSPACE to undo"
            if len(self.points) == 4
            else f"Click {4 - len(self.points)} more corner(s)"
        )
        cv2.putText(img, hint, (15, int(30 * ui)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * ui, (255, 255, 255), 2)
        return img

    def pick(self) -> Quad | None:
        h, w = self.frame.shape[:2]
        _fit_window(PICKER_WINDOW, w, h, self.max_w, self.max_h)
        cv2.setMouseCallback(PICKER_WINDOW, self._on_mouse)
        confirmed = False
        while True:
            cv2.imshow(PICKER_WINDOW, self._render())
            key = cv2.waitKey(20) & 0xFF
            if key == KEY_ENTER and len(self.points) == 4:
                confirmed = True
                break
            if key == KEY_ESC:
                break
            if key == KEY_BACKSPACE and self.points:
                self.points.pop()
        cv2.destroyWindow(PICKER_WINDOW)
        if not confirmed:
            return None
        try:
            return Quad.from_points(self.points)
        except ValueError as exc:
            log.warning("Invalid ROI: %s", exc)
            return None


def make_picker(max_w: int, max_h: int):
    def picker(frame: np.ndarray) -> Quad | None:
        print("\nClick the 4 corners of the escalator: top-left, top-right, bottom-right, bottom-left.")
        print("BACKSPACE undoes a point, ENTER confirms, ESC uses a default region.\n")
        return QuadPicker(frame, max_w, max_h).pick()

    return picker


class PreviewWindow:
    """Live preview with keyboard controls; used as the runner's frame hook.

    q quit - p pause/resume - r re-select ROI - s snapshot - f fullscreen
    """

    def __init__(self, width: int, height: int, max_w: int, max_h: int, snapshot_dir: str):
        self.max_w, self.max_h = max_w, max_h
        self.snapshot_dir = snapshot_dir
        self.fullscreen = False
        _fit_window(WINDOW, width, height, max_w, max_h)

    def __call__(self, frame: Frame, annotated: np.ndarray | None, result: FrameResult) -> HookAction:
        cv2.imshow(WINDOW, annotated if annotated is not None else frame.image)
        paused = False
        while True:
            key = cv2.waitKey(50 if paused else 1) & 0xFF
            if key == ord("q"):
                return "quit"
            if key == ord("p"):
                paused = not paused
                log.info("Paused" if paused else "Resumed")
            elif key == ord("r"):
                quad = QuadPicker(frame.image, self.max_w, self.max_h).pick()
                if quad is not None:
                    return quad
            elif key == ord("s"):
                path = save_snapshot(
                    annotated if annotated is not None else frame.image, self.snapshot_dir, "manual", result.index
                )
                log.info("Saved snapshot -> %s", path)
            elif key == ord("f"):
                self.fullscreen = not self.fullscreen
                cv2.setWindowProperty(
                    WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
                )
            if not paused:
                return None

    def close(self) -> None:
        cv2.destroyAllWindows()
