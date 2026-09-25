import csv
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import pytest

from escalator_monitor.config import Config
from escalator_monitor.detector import ReplayDetector
from escalator_monitor.geometry import Quad
from escalator_monitor.runner import run
from escalator_monitor.synthetic import write_clip


@pytest.fixture(scope="module")
def clip(tmp_path_factory, request):
    from conftest import SHORT_SEGMENTS

    from escalator_monitor.synthetic import SceneSpec

    path = tmp_path_factory.mktemp("clip") / "scene.mp4"
    return write_clip(SceneSpec(SHORT_SEGMENTS, width=480, height=360, seed=3), path)


def make_cfg(clip, out, **kw) -> Config:
    paths = dict(
        input_video=clip["video"],
        output_video=str(out / "result.mp4"),
        events_csv=str(out / "events.csv"),
        timeline_csv=str(out / "timeline.csv"),
        report_json=str(out / "report.json"),
        snapshot_dir=str(out / "snapshots"),
        roi_file=str(out / "roi.json"),
    )
    return Config(**{**paths, **kw})


def test_run_writes_all_outputs(clip, tmp_path):
    cfg = make_cfg(clip, tmp_path)
    result = run(cfg, ReplayDetector.from_json(clip["boxes"]), roi=Quad.load(clip["roi"]))

    with open(cfg.events_csv) as f:
        events = list(csv.DictReader(f))
    assert [e["to_state"] for e in events] == ["WORKING", "STOPPED / FAULT", "WORKING", "IDLE"]
    assert events[1]["snapshot_path"] and (tmp_path / "snapshots").exists()

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["fault_count"] == 1
    assert report["frames_analyzed"] == 350
    assert abs(report["duration_s"] - 14.0) < 0.1
    assert 0 < report["availability_pct"] < 100

    cap = cv2.VideoCapture(cfg.output_video)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 350
    cap.release()
    assert result.summary["final_state"] == "IDLE"


def test_roi_from_config_and_stride(clip, tmp_path):
    roi = Quad.load(clip["roi"]).to_list()
    cfg = make_cfg(clip, tmp_path, roi_points=roi, frame_stride=2, output_video="")
    result = run(cfg, ReplayDetector.from_json(clip["boxes"]))
    assert result.summary["frames_analyzed"] == 175
    assert abs(result.summary["duration_s"] - 14.0) < 0.2
    assert result.summary["fault_count"] == 1


def test_downscaled_processing_scales_the_roi(clip, tmp_path):
    roi = Quad.load(clip["roi"]).to_list()
    cfg = make_cfg(clip, tmp_path, roi_points=roi, process_max_width=240, output_video="", max_frames=40)
    boxes = {int(k): [[v // 2 for v in b] for b in bs] for k, bs in json.loads(Path(clip["boxes"]).read_text()).items()}
    result = run(cfg, ReplayDetector(boxes))
    assert result.summary["frame_size"] == [240, 180]
    assert result.quad == Quad.from_points(roi).scaled(0.5)
    assert result.summary["frames_analyzed"] == 40


def test_missing_input_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        run(Config(input_video=str(tmp_path / "nope.mp4")), ReplayDetector({}))


def test_webhook_receives_transitions(clip, tmp_path):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = make_cfg(clip, tmp_path, output_video="", webhook_url=f"http://127.0.0.1:{server.server_port}/hook")
        cfg = replace(cfg, save_fault_snapshots=False)
        run(cfg, ReplayDetector.from_json(clip["boxes"]), roi=Quad.load(clip["roi"]))
    finally:
        server.shutdown()
    assert [p["to_state"] for p in received] == ["WORKING", "STOPPED / FAULT", "WORKING", "IDLE"]
    assert received[1]["source"] == "scene"
