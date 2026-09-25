"""Command-line interface.

escalator-monitor run --input input/clip.mp4             # desktop preview + outputs
escalator-monitor run --input rtsp://cam/stream --headless --webhook https://...
escalator-monitor synth demo.mp4                         # synthetic clip + ground truth
escalator-monitor evaluate output/timeline.csv labels.csv
escalator-monitor benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import Config

log = logging.getLogger("escalator_monitor")
COMMANDS = ("run", "synth", "evaluate", "benchmark")


def _add_run(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="analyse a video file, stream or camera")
    p.add_argument("--config", help="YAML config file (only the keys you want to change)")
    p.add_argument("--input", help="video file, RTSP/HTTP URL or webcam index")
    p.add_argument("--output", help="annotated output video ('' to disable)")
    p.add_argument("--output-dir", help="put every output file in this directory")
    p.add_argument("--roi", help="escalator corners 'x1,y1,x2,y2,x3,y3,x4,y4' in source pixels")
    p.add_argument("--webhook", help="POST each state change as JSON to this URL")
    p.add_argument("--detector", default="yolo", help="'yolo' (default), 'none', or 'replay:<boxes.json>'")
    p.add_argument("--headless", action="store_true", help="no preview window (servers, Docker, CI)")
    p.add_argument("--max-frames", type=int, help="stop after N analysed frames")
    p.add_argument("--process-max-width", type=int, help="downscale wider frames before analysis")
    p.add_argument("--stride", type=int, help="analyse every Nth frame of a file")
    p.add_argument(
        "--expected-direction", choices=["any", "up", "down"], help="alert if the surface moves the other way"
    )
    p.add_argument("--compensate-camera-motion", action="store_true", help="cancel camera shake using the background")
    p.add_argument("--preview-max-width", type=int, help="max preview window width (default 1280)")
    p.add_argument("--preview-max-height", type=int, help="max preview window height (default 720)")


def _config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.input:
        cfg.input_video = args.input
    if args.output_dir:
        out = Path(args.output_dir)
        cfg.output_video = str(out / "result.mp4")
        cfg.events_csv = str(out / "events.csv")
        cfg.timeline_csv = str(out / "timeline.csv")
        cfg.report_json = str(out / "report.json")
        cfg.snapshot_dir = str(out / "snapshots")
    if args.output is not None:
        cfg.output_video = args.output
    if args.roi:
        from .geometry import Quad

        cfg.roi_points = Quad.parse(args.roi).to_list()
    overrides = {
        "webhook_url": args.webhook,
        "max_frames": args.max_frames,
        "process_max_width": args.process_max_width,
        "frame_stride": args.stride,
        "expected_direction": args.expected_direction,
        "preview_max_width": args.preview_max_width,
        "preview_max_height": args.preview_max_height,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if args.compensate_camera_motion:
        cfg.compensate_camera_motion = True
    cfg.validate()
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    from .detector import build_detector
    from .runner import run

    cfg = _config_from_args(args)
    detector = build_detector(args.detector, cfg)

    picker = hook = None
    if not args.headless:
        from .gui import PreviewWindow, gui_available, make_picker

        if gui_available():
            picker = make_picker(cfg.preview_max_width, cfg.preview_max_height)
            # The window is created lazily on the first frame, once the size is known.
            state: dict = {}

            def hook(frame, annotated, result):
                if "win" not in state:
                    h, w = frame.image.shape[:2]
                    state["win"] = PreviewWindow(w, h, cfg.preview_max_width, cfg.preview_max_height, cfg.snapshot_dir)
                return state["win"](frame, annotated, result)

            print("Controls: q quit | p pause | r re-select ROI | s snapshot | f fullscreen")
        else:
            log.warning("No display available; running headless")

    try:
        result = run(cfg, detector, picker=picker, hook=hook)
    finally:
        if hook is not None:
            import cv2

            cv2.destroyAllWindows()

    s = result.summary
    print(f"\nAnalysed {s['frames_analyzed']} frames ({s['duration_s']} s) at {s['processing_fps']} FPS")
    for state_name, seconds in s["time_in_state_s"].items():
        print(f"  {state_name:<16} {seconds:8.1f} s  ({s['time_in_state_pct'].get(state_name, 0):.1f} %)")
    if s["availability_pct"] is not None:
        print(f"  Availability     {s['availability_pct']:.1f} %  (WORKING vs STOPPED time)")
    print(f"  Faults           {s['fault_count']}")
    for label, path in (
        ("Video", result.output_video),
        ("Events", result.events_csv),
        ("Timeline", result.timeline_csv),
        ("Report", result.report_json),
    ):
        if path:
            print(f"  {label:<9}{path}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from .synthetic import DEMO_SEGMENTS, SceneSpec, Segment, write_clip

    segments = DEMO_SEGMENTS
    if args.segments:
        segments = []
        for part in args.segments.split(","):
            kind, seconds, people = (part.split(":") + ["0"])[:3]
            segments.append(
                Segment(float(seconds), kind.strip().lower() in ("move", "moving", "working", "run"), int(people))
            )
    spec = SceneSpec(
        segments,
        width=args.width,
        height=args.height,
        fps=args.fps,
        noise=args.noise,
        jitter=args.jitter,
        flicker=args.flicker,
        direction=args.direction,
        seed=args.seed,
    )
    paths = write_clip(spec, args.output)
    print(json.dumps(paths, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate, format_report, load_intervals, load_timeline

    report = evaluate(load_timeline(args.timeline), load_intervals(args.ground_truth))
    print(format_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import accuracy_suite, format_accuracy, format_speed, speed_suite

    cfg = Config()
    if not args.skip_accuracy:
        print("Accuracy on synthetic scenes (simulated person boxes, 3 seeds each):\n")
        print(format_accuracy(accuracy_suite(cfg)))
    print(
        "\nMean time per stage in ms (detect runs every "
        f"{cfg.detect_every_n_frames} frames; pipeline_fps excludes video decode/encode):\n"
    )
    print(format_speed(speed_suite(cfg, yolo_model=args.yolo)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="escalator-monitor", description="Escalator state monitoring from CCTV video")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)

    p = sub.add_parser("synth", help="render a synthetic escalator clip with ground truth")
    p.add_argument("output", help="output .mp4 path")
    p.add_argument("--segments", help="e.g. 'moving:6:3,stopped:8:3,moving:6:2,stopped:7:0' (state:seconds:people)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--noise", type=float, default=3.0)
    p.add_argument("--jitter", type=float, default=0.0)
    p.add_argument("--flicker", type=float, default=0.0)
    p.add_argument("--direction", choices=["up", "down"], default="up")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("evaluate", help="score a timeline.csv against labelled intervals")
    p.add_argument("timeline", help="timeline.csv written by 'run'")
    p.add_argument("ground_truth", help="CSV with start_s,end_s,state")
    p.add_argument("--json", help="also write the metrics as JSON")

    p = sub.add_parser("benchmark", help="accuracy and speed on synthetic scenes")
    p.add_argument("--yolo", help="include YOLO inference in the timing, e.g. yolo11n.pt")
    p.add_argument("--skip-accuracy", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backwards compatible: `python -m escalator_monitor --config x.yaml` means `run`.
    if (
        argv
        and argv[0] not in COMMANDS
        and argv[0] not in ("-h", "--help", "--version")
        and not argv[0].startswith("--log-level")
    ):
        argv.insert(0, "run")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    handlers = {"run": cmd_run, "synth": cmd_synth, "evaluate": cmd_evaluate, "benchmark": cmd_benchmark}
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
