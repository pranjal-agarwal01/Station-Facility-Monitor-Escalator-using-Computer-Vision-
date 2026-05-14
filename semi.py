"""
Escalator State Monitor v2.1

Patch over v2:
  * Fixed preview window appearing "zoomed in" on high-resolution input.
    The saved MP4 was always correct; only the on-screen cv2.imshow window
    was opening at 1:1 pixel size, which exceeds typical laptop screens
    for 1080p+ video so you only saw the center of the frame.
  * Fix: cv2.namedWindow with WINDOW_NORMAL + WINDOW_KEEPRATIO, then
    cv2.resizeWindow scaled to fit the screen. Preview is now letterboxed,
    full frame visible, no zoom.
  * Plus optional cv2.WND_PROP_FULLSCREEN toggle via 'f' key.

All other v2 functionality unchanged. Key upgrades over v1:
  1. Quadrilateral (perspective-aware) ROI instead of axis-aligned box.
  2. ROI persistence per video (JSON sidecar, no re-drawing on every run).
  3. Directional optical-flow check — kills the "people walking on a stopped
     escalator" false positive. Handrails must move vertically AND in the same
     direction; steps must agree.
  4. ByteTrack-style detection skipping — YOLO every N frames, IoU-track in
     between. ~3x compute saving.
  5. Snapshot saving on WORKING -> STOPPED/FAULT transition.
  6. Per-camera YAML config (constants are now overrideable, not hardcoded).
  7. Patched state-machine: low motion + ANY people now correctly enters fault.
  8. DIS optical flow (3-5x faster than Farneback, similar quality).
     Falls back to Farneback if opencv-contrib not installed.

Controls: q=quit, p=pause/resume, r=reselect ROI, s=snapshot, f=fullscreen toggle
"""

import argparse
import csv
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =====================================================
# CONFIG (overridable via YAML)
# =====================================================
@dataclass
class Config:
    # I/O
    input_video: str = "input/stop3.mp4"
    output_video: str = "output/result.mp4"
    events_csv: str = "output/events.csv"
    snapshot_dir: str = "output/snapshots"
    roi_file: str = "output/roi.json"

    # Model
    yolo_model: str = "yolo11n.pt"
    person_conf_threshold: float = 0.35
    person_mask_padding: int = 18
    detect_every_n_frames: int = 3

    # Optical flow
    flow_downscale: float = 0.5
    use_dis_flow: bool = True

    # Motion gates
    handrail_mag_gate: float = 0.10
    steps_mag_gate: float = 0.15
    consistency_gate: float = 0.55

    # Score-shaping
    handrail_mag_norm: float = 0.30
    steps_mag_norm: float = 0.50
    consistency_norm: float = 0.85

    # Geometry
    handrail_width_frac: float = 0.10

    # Direction checks
    require_vertical_motion: bool = True
    vertical_ratio_min: float = 1.5
    require_handrail_agreement: bool = True
    direction_dot_min: float = 0.3

    # Fusion
    handrail_weight: float = 0.55
    steps_weight: float = 0.45
    move_confidence_min: float = 0.35

    # State machine
    window_size: int = 45
    enter_working_ratio: float = 0.45
    exit_working_ratio: float = 0.25
    enter_stopped_ratio: float = 0.60
    idle_people_ratio: float = 0.20
    fault_score_max: float = 0.20

    # Alerting
    webhook_url: str = ""

    # Display
    show_debug_overlay: bool = True
    save_fault_snapshots: bool = True

    # NEW: preview window sizing
    preview_max_width: int = 1280   # cap preview width to this (px)
    preview_max_height: int = 720   # cap preview height to this (px)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        if not HAS_YAML:
            raise RuntimeError("Install pyyaml: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =====================================================
# ROI: quadrilateral, persisted to JSON
# =====================================================
@dataclass
class Quad:
    points: list

    def as_array(self) -> np.ndarray:
        return np.array(self.points, dtype=np.int32)

    def bbox(self) -> tuple:
        a = self.as_array()
        return int(a[:, 0].min()), int(a[:, 1].min()), int(a[:, 0].max()), int(a[:, 1].max())

    def make_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.as_array()], 1)
        return mask

    def make_handrail_masks(self, h: int, w: int, width_frac: float):
        pts = self.as_array().astype(np.float32)
        tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
        inner_tl = tl + (tr - tl) * width_frac
        inner_bl = bl + (br - bl) * width_frac
        inner_tr = tr + (tl - tr) * width_frac
        inner_br = br + (bl - br) * width_frac
        left = np.array([tl, inner_tl, inner_bl, bl], dtype=np.int32)
        right = np.array([inner_tr, tr, br, inner_br], dtype=np.int32)
        steps = np.array([inner_tl, inner_tr, inner_br, inner_bl], dtype=np.int32)
        left_mask = np.zeros((h, w), dtype=np.uint8); cv2.fillPoly(left_mask, [left], 1)
        right_mask = np.zeros((h, w), dtype=np.uint8); cv2.fillPoly(right_mask, [right], 1)
        steps_mask = np.zeros((h, w), dtype=np.uint8); cv2.fillPoly(steps_mask, [steps], 1)
        return left_mask, right_mask, steps_mask


class QuadPicker:
    """Click 4 corners: TL, TR, BR, BL. ENTER confirm, ESC cancel, BACKSPACE undo.
    Window is sized to fit the screen so high-res input is fully visible."""
    WIN = "Pick 4 corners (TL, TR, BR, BL) then ENTER"

    def __init__(self, frame, max_w=1280, max_h=720):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.points = []
        self.done = False
        self.cancelled = False
        self.max_w = max_w
        self.max_h = max_h

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            # Note: x, y are in window coords. With WINDOW_KEEPRATIO + resizeWindow,
            # OpenCV maps clicks back to image coords automatically. No conversion needed.
            self.points.append([x, y])

    def _redraw(self):
        self.display = self.frame.copy()
        labels = ["TL", "TR", "BR", "BL"]
        for i, p in enumerate(self.points):
            cv2.circle(self.display, tuple(p), 6, (0, 255, 0), -1)
            cv2.putText(self.display, labels[i], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(self.display, tuple(self.points[i]),
                         tuple(self.points[i + 1]), (0, 255, 255), 2)
        if len(self.points) == 4:
            cv2.line(self.display, tuple(self.points[3]),
                     tuple(self.points[0]), (0, 255, 255), 2)
            cv2.putText(self.display, "ENTER to confirm, BACKSPACE to undo",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            need = 4 - len(self.points)
            cv2.putText(self.display, f"Click {need} more corner(s)",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def pick(self):
        # KEY FIX: create resizable window and size it to fit screen
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        h, w = self.frame.shape[:2]
        scale = min(self.max_w / w, self.max_h / h, 1.0)
        cv2.resizeWindow(self.WIN, int(w * scale), int(h * scale))
        cv2.setMouseCallback(self.WIN, self._on_mouse)
        while True:
            self._redraw()
            cv2.imshow(self.WIN, self.display)
            k = cv2.waitKey(20) & 0xFF
            if k == 13 and len(self.points) == 4:
                self.done = True; break
            elif k == 27:
                self.cancelled = True; break
            elif k == 8 and self.points:
                self.points.pop()
        cv2.destroyWindow(self.WIN)
        if self.cancelled or len(self.points) != 4:
            return None
        return Quad(points=self.points)


def load_or_pick_roi(frame, roi_path: str, max_w=1280, max_h=720) -> Quad:
    if os.path.exists(roi_path):
        try:
            with open(roi_path) as f:
                data = json.load(f)
            q = Quad(points=data["points"])
            print(f"Loaded ROI from {roi_path}")
            return q
        except Exception as e:
            print(f"Warning: could not load {roi_path}: {e}, redrawing.")

    print("\n>>> Click the 4 corners of the escalator:")
    print("    1. Top-Left   2. Top-Right   3. Bottom-Right   4. Bottom-Left")
    print("    BACKSPACE to undo, ENTER when done, ESC to cancel.\n")
    quad = QuadPicker(frame, max_w=max_w, max_h=max_h).pick()
    if quad is None:
        h, w = frame.shape[:2]
        quad = Quad(points=[
            [int(w*0.25), int(h*0.20)],
            [int(w*0.75), int(h*0.20)],
            [int(w*0.75), int(h*0.95)],
            [int(w*0.25), int(h*0.95)],
        ])
        print("Using fallback rectangular ROI.")
    save_roi(quad, roi_path)
    return quad


def save_roi(quad: Quad, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"points": quad.points,
                   "saved_at": datetime.now().isoformat()}, f, indent=2)
    print(f"Saved ROI to {path}")


# =====================================================
# OPTICAL FLOW
# =====================================================
class FlowEngine:
    def __init__(self, use_dis: bool):
        self.dis = None
        if use_dis:
            try:
                self.dis = cv2.optflow.DISOpticalFlow_create(
                    cv2.optflow.DISOPTICAL_FLOW_PRESET_FAST)
                print("Using DIS optical flow.")
            except AttributeError:
                print("opencv-contrib not installed -> falling back to Farneback.")
                print("  (For speed, install: pip install opencv-contrib-python)")

    def compute(self, prev_gray, curr_gray):
        if self.dis is not None:
            return self.dis.calc(prev_gray, curr_gray, None)
        return cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def flow_stats_directional(flow, mask, mag_floor=0.1):
    valid = mask > 0
    if valid.sum() < 40:
        return 0.0, 0.0, 0.0, 0.0
    vx = flow[..., 0][valid]
    vy = flow[..., 1][valid]
    mag = np.sqrt(vx * vx + vy * vy)
    mean_mag = float(np.mean(mag))
    strong = mag > mag_floor
    if strong.sum() < 10:
        return mean_mag, 0.0, 0.0, 0.0
    sx = vx[strong]; sy = vy[strong]; sm = mag[strong]
    mean_vx = float(np.sum(sx) / strong.sum())
    mean_vy = float(np.sum(sy) / strong.sum())
    ang = np.arctan2(sy, sx)
    cx = np.sum(sm * np.cos(ang)); cy = np.sum(sm * np.sin(ang))
    consistency = float(np.sqrt(cx * cx + cy * cy) / (np.sum(sm) + 1e-6))
    return mean_mag, consistency, mean_vx, mean_vy


def smooth_score(mag, cons, mag_gate, mag_norm, cons_gate, cons_norm):
    if mag < mag_gate or cons < cons_gate:
        return 0.0
    mag_term = mag / (mag + mag_norm)
    cons_term = cons / (cons + (1.0 - cons_norm))
    return float(np.sqrt(mag_term * min(cons_term, 1.0)))


def direction_penalty(mean_vx, mean_vy, cfg: Config):
    if not cfg.require_vertical_motion:
        return 1.0
    if abs(mean_vy) < 1e-3:
        return 0.0
    ratio = abs(mean_vy) / (abs(mean_vx) + 1e-3)
    if ratio < cfg.vertical_ratio_min:
        return 0.0
    return float(min(1.0, ratio / (cfg.vertical_ratio_min * 2)))


def handrail_agreement(l_vx, l_vy, r_vx, r_vy, cfg: Config) -> float:
    if not cfg.require_handrail_agreement:
        return 1.0
    ln = np.sqrt(l_vx * l_vx + l_vy * l_vy)
    rn = np.sqrt(r_vx * r_vx + r_vy * r_vy)
    if ln < 1e-3 or rn < 1e-3:
        return 0.0
    dot = (l_vx * r_vx + l_vy * r_vy) / (ln * rn)
    if dot < cfg.direction_dot_min:
        return 0.0
    return float(max(0.0, dot))


# =====================================================
# IOU TRACKER
# =====================================================
def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aarea = (ax2 - ax1) * (ay2 - ay1)
    barea = (bx2 - bx1) * (by2 - by1)
    return inter / (aarea + barea - inter + 1e-6)


class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_misses=5):
        self.tracks = {}
        self.next_id = 0
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses

    def update(self, detections: list) -> list:
        if not detections:
            survivors = []
            for tid, t in list(self.tracks.items()):
                t["misses"] += 1
                if t["misses"] > self.max_misses:
                    del self.tracks[tid]
                else:
                    survivors.append(t["box"])
            return survivors

        used_tracks = set()
        for det in detections:
            best_id, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                if tid in used_tracks:
                    continue
                i = iou(det, t["box"])
                if i > best_iou:
                    best_iou, best_id = i, tid
            if best_iou >= self.iou_thresh:
                self.tracks[best_id]["box"] = det
                self.tracks[best_id]["misses"] = 0
                used_tracks.add(best_id)
            else:
                self.tracks[self.next_id] = {"box": det, "misses": 0}
                used_tracks.add(self.next_id)
                self.next_id += 1

        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid]["misses"] += 1
                if self.tracks[tid]["misses"] > self.max_misses:
                    del self.tracks[tid]

        return [t["box"] for t in self.tracks.values()]


# =====================================================
# UI HELPERS
# =====================================================
def draw_panel(frame, lines, origin, width=460):
    h_per = 26
    panel_h = 18 + h_per * len(lines)
    x1, y1 = origin
    x2, y2 = x1 + width, y1 + panel_h
    fh, fw = frame.shape[:2]
    x2 = min(x2, fw); y2 = min(y2, fh)
    sub = frame[y1:y2, x1:x2].copy()
    bg = np.full(sub.shape, (15, 15, 15), dtype=np.uint8)
    blended = cv2.addWeighted(bg, 0.82, sub, 0.18, 0)
    frame[y1:y2, x1:x2] = blended
    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (x1 + 14, y1 + 26 + i * h_per),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_status_badge(frame, status, color):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    pad = 16
    bx1 = (w - tw) // 2 - pad
    bx2 = bx1 + tw + 2 * pad
    by1 = 15
    by2 = by1 + th + 2 * pad
    sub = frame[by1:by2, bx1:bx2].copy()
    bg = np.full(sub.shape, color, dtype=np.uint8)
    blended = cv2.addWeighted(bg, 0.88, sub, 0.12, 0)
    frame[by1:by2, bx1:bx2] = blended
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
    cv2.putText(frame, status, (bx1 + pad, by2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)


# =====================================================
# ALERTING
# =====================================================
def post_webhook(url: str, payload: dict):
    if not url:
        return
    try:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=2.0).read()
    except Exception as e:
        print(f"Webhook failed: {e}")


def save_snapshot(frame, snapshot_dir: str, label: str, frame_idx: int) -> str:
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_f{frame_idx}_{label.replace(' ', '_').replace('/', '_')}.jpg"
    path = os.path.join(snapshot_dir, fname)
    cv2.imwrite(path, frame)
    return path


# =====================================================
# PREVIEW WINDOW HELPER  (NEW)
# =====================================================
WIN_NAME = "Escalator Monitor v2"

def setup_preview_window(frame_w: int, frame_h: int, max_w: int, max_h: int):
    """Create a properly-sized preview window for an arbitrarily-large video.
    The saved MP4 is unaffected — this only controls the live display.
    Returns the (display_w, display_h) used for the window.
    """
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    scale = min(max_w / frame_w, max_h / frame_h, 1.0)
    disp_w = max(320, int(frame_w * scale))
    disp_h = max(240, int(frame_h * scale))
    cv2.resizeWindow(WIN_NAME, disp_w, disp_h)
    print(f"Preview window sized to {disp_w}x{disp_h} "
          f"(input {frame_w}x{frame_h}, scale={scale:.2f})")
    return disp_w, disp_h


# =====================================================
# MAIN
# =====================================================
def main(cfg: Config):
    if not os.path.exists(cfg.input_video):
        print(f"Error: input not found at {cfg.input_video}"); return

    Path(cfg.output_video).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.events_csv).parent.mkdir(parents=True, exist_ok=True)

    roi_path = cfg.roi_file
    if not roi_path or roi_path == "output/roi.json":
        stem = Path(cfg.input_video).stem
        roi_path = f"output/roi_{stem}.json"

    print(f"Loading {cfg.yolo_model}...")
    model = YOLO(cfg.yolo_model)

    cap = cv2.VideoCapture(cfg.input_video)
    if not cap.isOpened():
        print("Error opening video"); return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps}fps, {total} frames")

    ret, first = cap.read()
    if not ret:
        print("Cannot read first frame"); return

    quad = load_or_pick_roi(first, roi_path,
                            max_w=cfg.preview_max_width,
                            max_h=cfg.preview_max_height)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Output writer at native resolution (UNCHANGED)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(cfg.output_video, fourcc, fps, (width, height))
    events_file = open(cfg.events_csv, "w", newline="")
    events_writer = csv.writer(events_file)
    events_writer.writerow(["frame", "timestamp_wall", "video_time",
                            "event", "details", "snapshot_path"])

    flow_engine = FlowEngine(cfg.use_dis_flow)
    tracker = SimpleTracker(iou_thresh=0.3,
                            max_misses=cfg.detect_every_n_frames * 2)

    # NEW: set up resizable preview window once
    setup_preview_window(width, height,
                         cfg.preview_max_width, cfg.preview_max_height)
    is_fullscreen = False

    prev_gray = None
    motion_hist = deque(maxlen=cfg.window_size)
    people_hist = deque(maxlen=cfg.window_size)
    score_hist = deque(maxlen=cfg.window_size)

    status = "INITIALIZING"
    prev_status = None
    state_since_frame = 0
    state_duration_log = {}

    frame_idx = 0
    paused = False
    t_start = time.time()

    print("\nControls: q=quit, p=pause/resume, r=reselect ROI, s=snapshot, f=fullscreen\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. People detection
        if frame_idx % cfg.detect_every_n_frames == 0 or frame_idx == 1:
            results = model(frame, verbose=False, classes=[0],
                            conf=cfg.person_conf_threshold)
            detections = []
            quad_mask_full = quad.make_mask(height, width)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if 0 <= cy < height and 0 <= cx < width and quad_mask_full[cy, cx]:
                        detections.append((x1, y1, x2, y2))
            person_boxes = tracker.update(detections)
        else:
            person_boxes = tracker.update([])

        for (x1, y1, x2, y2) in person_boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 0), 2)
        people_count = len(person_boxes)

        # 2. Motion analysis
        handrail_mag = handrail_cons = 0.0
        steps_mag = steps_cons = 0.0
        hr_score = st_score = 0.0
        move_confidence = 0.0
        is_moving = False
        dir_mult = hr_agree = 0.0

        if prev_gray is not None:
            if cfg.flow_downscale != 1.0:
                size = (max(8, int(width * cfg.flow_downscale)),
                        max(8, int(height * cfg.flow_downscale)))
                prev_s = cv2.resize(prev_gray, size)
                curr_s = cv2.resize(gray, size)
                scale = cfg.flow_downscale
            else:
                prev_s, curr_s, scale = prev_gray, gray, 1.0

            flow = flow_engine.compute(prev_s, curr_s)

            sh, sw = curr_s.shape
            left_mask, right_mask, steps_mask_arr = quad.make_handrail_masks(
                height, width, cfg.handrail_width_frac)
            left_s = cv2.resize(left_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            right_s = cv2.resize(right_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            steps_s = cv2.resize(steps_mask_arr, (sw, sh), interpolation=cv2.INTER_NEAREST)

            for (x1, y1, x2, y2) in person_boxes:
                pad = cfg.person_mask_padding
                bx1 = max(0, int((x1 - pad) * scale))
                by1 = max(0, int((y1 - pad) * scale))
                bx2 = min(sw, int((x2 + pad) * scale))
                by2 = min(sh, int((y2 + pad) * scale))
                if bx2 > bx1 and by2 > by1:
                    left_s[by1:by2, bx1:bx2] = 0
                    right_s[by1:by2, bx1:bx2] = 0
                    steps_s[by1:by2, bx1:bx2] = 0

            l_mag, l_cons, l_vx, l_vy = flow_stats_directional(flow, left_s)
            r_mag, r_cons, r_vx, r_vy = flow_stats_directional(flow, right_s)
            s_mag, s_cons, s_vx, s_vy = flow_stats_directional(flow, steps_s)

            handrail_mag = (l_mag + r_mag) / 2.0
            handrail_cons = (l_cons + r_cons) / 2.0
            steps_mag, steps_cons = s_mag, s_cons

            hr_dir = direction_penalty((l_vx + r_vx) / 2, (l_vy + r_vy) / 2, cfg)
            st_dir = direction_penalty(s_vx, s_vy, cfg)
            hr_agree = handrail_agreement(l_vx, l_vy, r_vx, r_vy, cfg)
            dir_mult = hr_dir * hr_agree

            if (left_s.sum() + right_s.sum()) > 200:
                hr_raw = smooth_score(
                    handrail_mag, handrail_cons,
                    cfg.handrail_mag_gate, cfg.handrail_mag_norm,
                    cfg.consistency_gate, cfg.consistency_norm)
                hr_score = hr_raw * dir_mult

            if steps_s.sum() > 200:
                st_raw = smooth_score(
                    steps_mag, steps_cons,
                    cfg.steps_mag_gate, cfg.steps_mag_norm,
                    cfg.consistency_gate, cfg.consistency_norm)
                st_score = st_raw * st_dir

            if steps_s.sum() < 200:
                move_confidence = hr_score
            else:
                move_confidence = (cfg.handrail_weight * hr_score +
                                   cfg.steps_weight * st_score)
            if max(hr_score, st_score) > 0.7:
                move_confidence = max(move_confidence, 0.7)

            is_moving = move_confidence >= cfg.move_confidence_min

            overlay = display.copy()
            hr_col = (0, 220, 0) if hr_score > 0.3 else (0, 100, 220)
            st_col = (0, 220, 0) if st_score > 0.3 else (0, 100, 220)
            cv2.fillPoly(overlay, [np.array(quad.points, dtype=np.int32)], (40, 40, 40))
            lf, rf, sf = quad.make_handrail_masks(height, width, cfg.handrail_width_frac)
            overlay[lf > 0] = hr_col
            overlay[rf > 0] = hr_col
            overlay[sf > 0] = st_col
            cv2.addWeighted(overlay, 0.12, display, 0.88, 0, display)

        prev_gray = gray
        motion_hist.append(is_moving)
        people_hist.append(people_count > 0)
        score_hist.append(move_confidence)

        # 3. State machine
        if len(motion_hist) >= cfg.window_size // 3:
            mr = sum(motion_hist) / len(motion_hist)
            pr = sum(people_hist) / len(people_hist)
            avg_score = sum(score_hist) / len(score_hist)
            new_status = status

            if status in ("INITIALIZING", "IDLE"):
                if mr >= cfg.enter_working_ratio or avg_score > 0.5:
                    new_status = "WORKING"
                elif pr >= 0.5 and (1.0 - mr) >= cfg.enter_stopped_ratio \
                        and avg_score < 0.15:
                    new_status = "STOPPED / FAULT"
                elif pr < cfg.idle_people_ratio and status == "INITIALIZING":
                    new_status = "IDLE"
            elif status == "WORKING":
                if mr < cfg.exit_working_ratio and avg_score < 0.2:
                    if pr >= cfg.idle_people_ratio:
                        new_status = "STOPPED / FAULT"
                    else:
                        new_status = "IDLE"
            elif status == "STOPPED / FAULT":
                if mr >= cfg.enter_working_ratio or avg_score > 0.4:
                    new_status = "WORKING"
                elif pr < cfg.idle_people_ratio:
                    new_status = "IDLE"

            if pr < cfg.idle_people_ratio and mr < cfg.exit_working_ratio \
                    and avg_score < 0.2:
                new_status = "IDLE"

            status = new_status

        status_color = {
            "WORKING": (0, 200, 0),
            "STOPPED / FAULT": (0, 0, 220),
            "IDLE": (180, 180, 0),
        }.get(status, (200, 200, 200))

        # 4. Logging
        if status != prev_status and prev_status is not None:
            video_ts = str(timedelta(seconds=frame_idx / fps))
            wall_ts = datetime.now().isoformat(timespec="seconds")
            duration = (frame_idx - state_since_frame) / fps
            state_duration_log[prev_status] = \
                state_duration_log.get(prev_status, 0) + duration

            snapshot_path = ""
            if cfg.save_fault_snapshots and status == "STOPPED / FAULT":
                snapshot_path = save_snapshot(
                    display, cfg.snapshot_dir, status, frame_idx)

            details = (f"people={people_count} hr_mag={handrail_mag:.2f} "
                       f"hr_score={hr_score:.2f} st_score={st_score:.2f} "
                       f"dir={dir_mult:.2f} agree={hr_agree:.2f} "
                       f"conf={move_confidence:.2f} prev_dur={duration:.1f}s")
            events_writer.writerow([frame_idx, wall_ts, video_ts,
                                    f"{prev_status} -> {status}",
                                    details, snapshot_path])
            events_file.flush()
            print(f"[{video_ts}] {prev_status} -> {status}  "
                  f"(conf={move_confidence:.2f}, dir={dir_mult:.2f})")

            post_webhook(cfg.webhook_url, {
                "event": f"{prev_status} -> {status}",
                "frame": frame_idx,
                "video_time": video_ts,
                "wall_time": wall_ts,
                "people": people_count,
                "move_confidence": round(move_confidence, 3),
                "snapshot": snapshot_path,
            })
            state_since_frame = frame_idx

        prev_status = status

        # 5. Visualization
        cv2.polylines(display, [quad.as_array()], True, status_color, 2)
        cv2.putText(display, "Escalator ROI",
                    tuple(quad.points[0] + np.array([0, -8])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        elapsed = time.time() - t_start
        fps_proc = frame_idx / elapsed if elapsed > 0 else 0
        timestamp = str(timedelta(seconds=frame_idx / fps)).split(".")[0]

        draw_status_badge(display, status, status_color)

        if cfg.show_debug_overlay:
            panel_y = height - 250
            draw_panel(display, [
                (f"Frame {frame_idx}/{total}  Time {timestamp}", (255, 255, 255)),
                (f"People in ROI: {people_count}", (255, 255, 255)),
                (f"Handrail mag={handrail_mag:.2f} cons={handrail_cons:.2f} score={hr_score:.2f}",
                 (180, 255, 180) if hr_score > 0.3 else (200, 200, 200)),
                (f"Steps    mag={steps_mag:.2f} cons={steps_cons:.2f} score={st_score:.2f}",
                 (180, 255, 180) if st_score > 0.3 else (200, 200, 200)),
                (f"Direction gate: {dir_mult:.2f}  Handrail agree: {hr_agree:.2f}",
                 (200, 200, 255)),
                (f"Move confidence: {move_confidence:.2f}  (need {cfg.move_confidence_min})",
                 (255, 255, 255)),
                (f"FPS: {fps_proc:.1f}", (170, 170, 170)),
            ], origin=(15, max(15, panel_y)))

        if total > 0:
            bar = int(width * (frame_idx / total))
            cv2.rectangle(display, (0, height - 4), (bar, height), status_color, -1)

        # Write FULL-resolution annotated frame to disk (UNCHANGED)
        out.write(display)

        # Show in resizable window. OpenCV scales the image to fit the window
        # automatically when WINDOW_KEEPRATIO is set. No manual cv2.resize needed.
        cv2.imshow(WIN_NAME, display)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("p"):
            paused = not paused
        elif k == ord("r"):
            new_quad = QuadPicker(frame,
                                  max_w=cfg.preview_max_width,
                                  max_h=cfg.preview_max_height).pick()
            if new_quad is not None:
                quad = new_quad
                save_roi(quad, roi_path)
        elif k == ord("s"):
            p = save_snapshot(display, cfg.snapshot_dir, "manual", frame_idx)
            print(f"Saved snapshot -> {p}")
        elif k == ord("f"):
            # NEW: fullscreen toggle
            is_fullscreen = not is_fullscreen
            cv2.setWindowProperty(
                WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
            print("Fullscreen:", is_fullscreen)

    if prev_status:
        duration = (frame_idx - state_since_frame) / fps
        state_duration_log[prev_status] = \
            state_duration_log.get(prev_status, 0) + duration

    cap.release(); out.release(); events_file.close(); cv2.destroyAllWindows()

    print(f"\nDone.")
    print(f"  Video : {cfg.output_video}")
    print(f"  Events: {cfg.events_csv}")
    print(f"  ROI   : {roi_path}")
    print(f"  State durations:")
    for s, d in state_duration_log.items():
        print(f"    {s:20s}  {d:.1f}s  ({d/60:.1f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Escalator state monitor v2.1")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--input", help="Override input video path")
    parser.add_argument("--output", help="Override output video path")
    parser.add_argument("--webhook", help="Override webhook URL")
    parser.add_argument("--preview-max-width", type=int,
                        help="Max preview window width in pixels (default 1280)")
    parser.add_argument("--preview-max-height", type=int,
                        help="Max preview window height in pixels (default 720)")
    args = parser.parse_args()

    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()
    if args.input: cfg.input_video = args.input
    if args.output: cfg.output_video = args.output
    if args.webhook: cfg.webhook_url = args.webhook
    if args.preview_max_width: cfg.preview_max_width = args.preview_max_width
    if args.preview_max_height: cfg.preview_max_height = args.preview_max_height

    main(cfg)