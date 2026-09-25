"""Runtime configuration.

Every field has a sensible default and can be overridden from a YAML file, so a
per-camera config only needs to list the values that differ.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DIRECTIONS = ("any", "up", "down")


@dataclass
class Config:
    # --- I/O ---------------------------------------------------------------
    input_video: str = "input/stopp.mp4"  # file path, RTSP/HTTP URL or webcam index ("0")
    output_video: str = "output/result.mp4"  # "" disables the annotated video
    events_csv: str = "output/events.csv"
    timeline_csv: str = "output/timeline.csv"  # per-frame log; "" disables it
    report_json: str = "output/report.json"
    snapshot_dir: str = "output/snapshots"
    roi_file: str = ""  # "" -> output/roi_<source name>.json
    roi_points: list | None = None  # [[x, y] x 4]; skips interactive ROI selection
    process_max_width: int = 0  # downscale wider frames before processing (0 = native)
    frame_stride: int = 1  # analyse every Nth frame of a file (offline speed-up)
    max_frames: int = 0  # stop after this many analysed frames (0 = no limit)

    # --- Person detection --------------------------------------------------
    yolo_model: str = "yolo11n.pt"
    yolo_imgsz: int = 640
    device: str = ""  # "" = auto, or "cpu", "0", "mps", ...
    person_conf_threshold: float = 0.35
    person_mask_padding: int = 18
    detect_every_n_frames: int = 3
    detect_roi_margin: float = 0.15  # detect on the ROI crop grown by this fraction; <0 = full frame

    # --- Optical flow ------------------------------------------------------
    flow_downscale: float = 0.5
    use_dis_flow: bool = True
    flow_roi_margin: float = 0.15  # flow is computed on the ROI crop grown by this fraction
    compensate_camera_motion: bool = False  # subtract background motion (camera shake)

    # --- Motion gates ------------------------------------------------------
    handrail_mag_gate: float = 0.10
    steps_mag_gate: float = 0.15
    consistency_gate: float = 0.55

    # --- Score shaping -----------------------------------------------------
    handrail_mag_norm: float = 0.30
    steps_mag_norm: float = 0.50
    consistency_norm: float = 0.85

    # --- Geometry ----------------------------------------------------------
    handrail_width_frac: float = 0.10

    # --- Direction checks --------------------------------------------------
    require_vertical_motion: bool = True
    vertical_ratio_min: float = 1.5
    require_handrail_agreement: bool = True
    direction_dot_min: float = 0.3
    expected_direction: str = "any"  # "up"/"down" (in image space) raises a WRONG DIRECTION alert

    # --- Fusion ------------------------------------------------------------
    handrail_weight: float = 0.55
    steps_weight: float = 0.45
    move_confidence_min: float = 0.35
    strong_region_score: float = 0.70  # one region this sure is enough on its own

    # --- State machine -----------------------------------------------------
    window_size: int = 45
    enter_working_ratio: float = 0.45
    exit_working_ratio: float = 0.25
    enter_stopped_ratio: float = 0.60
    idle_people_ratio: float = 0.20
    stopped_people_ratio: float = 0.50
    enter_working_score: float = 0.50
    resume_working_score: float = 0.40
    stopped_score_max: float = 0.15
    fault_score_max: float = 0.20

    # --- Alerting ----------------------------------------------------------
    webhook_url: str = ""
    webhook_timeout: float = 3.0

    # --- Display -----------------------------------------------------------
    show_debug_overlay: bool = True
    save_fault_snapshots: bool = True
    preview_max_width: int = 1280
    preview_max_height: int = 720

    # ------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            log.warning("Ignoring unknown config keys: %s", ", ".join(unknown))
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - pyyaml is a hard dependency
            raise RuntimeError("Reading YAML configs needs pyyaml: pip install pyyaml") from exc
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping of config keys at the top level")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        errors: list[str] = []

        def check(ok: bool, msg: str) -> None:
            if not ok:
                errors.append(msg)

        check(self.detect_every_n_frames >= 1, "detect_every_n_frames must be >= 1")
        check(self.frame_stride >= 1, "frame_stride must be >= 1")
        check(self.max_frames >= 0, "max_frames must be >= 0")
        check(self.process_max_width >= 0, "process_max_width must be >= 0")
        check(0.05 <= self.flow_downscale <= 1.0, "flow_downscale must be in [0.05, 1]")
        check(0.0 <= self.flow_roi_margin <= 1.0, "flow_roi_margin must be in [0, 1]")
        check(self.detect_roi_margin <= 1.0, "detect_roi_margin must be <= 1")
        check(0.0 < self.handrail_width_frac < 0.5, "handrail_width_frac must be in (0, 0.5)")
        check(self.window_size >= 3, "window_size must be >= 3")
        check(self.yolo_imgsz >= 32, "yolo_imgsz must be >= 32")
        check(self.person_mask_padding >= 0, "person_mask_padding must be >= 0")
        check(self.handrail_weight >= 0 and self.steps_weight >= 0, "fusion weights must be >= 0")
        check(self.vertical_ratio_min > 0, "vertical_ratio_min must be > 0")
        check(self.expected_direction in DIRECTIONS, f"expected_direction must be one of {DIRECTIONS}")
        check(self.webhook_timeout > 0, "webhook_timeout must be > 0")
        for name in (
            "person_conf_threshold",
            "consistency_gate",
            "consistency_norm",
            "move_confidence_min",
            "strong_region_score",
            "enter_working_ratio",
            "exit_working_ratio",
            "enter_stopped_ratio",
            "idle_people_ratio",
            "stopped_people_ratio",
            "enter_working_score",
            "resume_working_score",
            "stopped_score_max",
            "fault_score_max",
        ):
            value = getattr(self, name)
            check(0.0 <= value <= 1.0, f"{name} must be in [0, 1] (got {value})")
        check(
            self.exit_working_ratio <= self.enter_working_ratio,
            "exit_working_ratio must not exceed enter_working_ratio (hysteresis)",
        )
        if self.roi_points is not None:
            ok = isinstance(self.roi_points, (list, tuple)) and len(self.roi_points) == 4
            ok = ok and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in self.roi_points)
            check(ok, "roi_points must be four [x, y] pairs")
        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))

    def resolved_roi_file(self) -> str:
        if self.roi_file:
            return self.roi_file
        return str(Path("output") / f"roi_{source_name(self.input_video)}.json")


def source_name(source: str) -> str:
    """Short, filesystem-safe name for a video source (file, URL or camera index)."""
    source = str(source).strip()
    if source.isdigit():
        return f"camera{source}"
    if "://" in source:
        rest = source.split("://", 1)[1].split("@")[-1]  # drop credentials
        return re.sub(r"[^A-Za-z0-9]+", "_", rest).strip("_")[:60] or "stream"
    return Path(source).stem or "video"
