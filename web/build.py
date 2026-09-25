"""Assemble the static site: app files, vendored JS/WASM, the ONNX model and demo clips.

    cd web && npm ci && cd ..
    python web/build.py --out _site --model yolo11n.onnx
    python -m http.server -d _site 8000

Needs the escalator_monitor package (for the synthetic demo clip) and imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2

from escalator_monitor.config import Config
from escalator_monitor.detector import ReplayDetector
from escalator_monitor.geometry import Quad
from escalator_monitor.runner import run
from escalator_monitor.synthetic import DEMO_SEGMENTS, SceneSpec, Segment, write_clip

WEB = Path(__file__).resolve().parent
REPO = WEB.parent
VENDOR_FILES = {
    "@techstark/opencv-js/dist/opencv.js": "opencv.js",
    "onnxruntime-web/dist/ort.wasm.min.mjs": "ort.wasm.min.mjs",
    "onnxruntime-web/dist/ort-wasm-simd-threaded.mjs": "ort-wasm-simd-threaded.mjs",
    "onnxruntime-web/dist/ort-wasm-simd-threaded.wasm": "ort-wasm-simd-threaded.wasm",
    "onnxruntime-web/dist/ort.webgpu.min.mjs": "ort.webgpu.min.mjs",
    "onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs": "ort-wasm-simd-threaded.asyncify.mjs",
    "onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm": "ort-wasm-simd-threaded.asyncify.wasm",
}
DEMOS = [
    {
        "id": "fault-cycle",
        "tag": "Synthetic · 27 s",
        "title": "Running, fault, running, idle",
        "description": "Rendered scene with known ground truth, so the result shows its own accuracy.",
        "spec": SceneSpec(DEMO_SEGMENTS, width=640, height=480, seed=5),
    },
    {
        "id": "crowded-shaky",
        "tag": "Synthetic · 20 s",
        "title": "Crowded escalator, shaky camera",
        "description": "Busier scene with camera shake. Try it with shake compensation on and off.",
        "spec": SceneSpec(
            [Segment(6, True, 6), Segment(8, False, 5), Segment(6, True, 4)], jitter=1.0, noise=5.0, seed=9
        ),
    },
]


def ffmpeg(*args: str) -> bool:
    """Run the ffmpeg binary bundled with imageio-ffmpeg; False if unavailable or failing."""
    try:
        import imageio_ffmpeg

        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", *args], check=True)
        return True
    except (ImportError, OSError, subprocess.CalledProcessError) as exc:
        print(f"  (ffmpeg step skipped: {exc})")
        return False


def to_webm(mp4: Path, webm: Path, crf: int = 36) -> bool:
    """VP9 copy for browsers without H.264 (e.g. open-source Chromium builds)."""
    return ffmpeg("-i", str(mp4), "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf), "-row-mt", "1", "-an", str(webm))


def poster(video: Path, jpg: Path, at_s: float = 1.5, width: int = 320) -> bool:
    """Save one frame of `video` as a small JPEG thumbnail."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    jpg.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(jpg), frame, [cv2.IMWRITE_JPEG_QUALITY, 84]))


def build_hero(out: Path, clip: Path, boxes: Path, roi: list) -> None:
    """Looping annotated clip for the landing section, made by the Python pipeline itself."""
    full = out / "_hero_full.mp4"
    cfg = Config(input_video=str(clip), output_video=str(full), events_csv="", timeline_csv="", report_json="",
                 save_fault_snapshots=False, roi_points=roi)  # fmt: skip
    run(cfg, ReplayDetector.from_json(boxes))
    assets = out / "assets"
    mp4 = assets / "hero.mp4"
    trim = ["-ss", "4.5", "-t", "12", "-i", str(full), "-vf", "scale=640:-2", "-an"]
    ffmpeg(
        *trim,
        "-c:v",
        "libx264",
        "-crf",
        "27",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(mp4),
    )
    to_webm(mp4, assets / "hero.webm", crf=38)
    poster(mp4, assets / "hero.jpg", at_s=4.0, width=640)
    full.unlink(missing_ok=True)


def build_demo(demo: dict, out: Path) -> dict:
    spec: SceneSpec = demo["spec"]
    mp4 = out / "demo" / f"{demo['id']}.mp4"
    mp4.parent.mkdir(parents=True, exist_ok=True)
    paths = write_clip(spec, mp4)
    sources = [{"src": f"demo/{mp4.name}", "type": 'video/mp4; codecs="avc1.42E01E"'}]
    if to_webm(mp4, mp4.with_suffix(".webm")):
        sources.append({"src": f"demo/{mp4.stem}.webm", "type": 'video/webm; codecs="vp9"'})
    with open(paths["ground_truth"], newline="") as f:
        truth = [[float(r["start_s"]), float(r["end_s"]), r["state"]] for r in csv.DictReader(f)]
    roi = Quad.load(paths["roi"]).to_list()
    for key in ("ground_truth", "roi"):
        Path(paths[key]).unlink()  # inlined into config.json
    thumb = out / "demo" / f"{demo['id']}.jpg"
    return {
        "id": demo["id"],
        "tag": demo["tag"],
        "poster": f"demo/{thumb.name}" if poster(mp4, thumb) else None,
        "title": demo["title"],
        "description": demo["description"],
        "sources": sources,
        "fps": spec.fps,
        "roi": roi,
        "boxes": f"demo/{Path(paths['boxes']).name}",
        "groundTruth": truth,
    }


def build_examples(out: Path) -> list[dict]:
    """Real clips from examples/ (see examples/README.md); analysed with YOLO."""
    entries = []
    for clip in sorted((REPO / "examples").glob("*.mp4")):
        target = out / "examples" / clip.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clip, target)
        thumb = target.with_suffix(".jpg")
        entry = {
            "id": clip.stem,
            "tag": "Real footage",
            "poster": f"examples/{thumb.name}" if poster(target, thumb) else None,
            "title": clip.stem.replace("_", " ").replace("-", " ").capitalize(),
            "description": "Real CCTV footage, analysed with YOLO in your browser.",
            "sources": [{"src": f"examples/{clip.name}", "type": "video/mp4"}],
            "fps": None,
        }
        roi = clip.with_suffix(".roi.json")
        if roi.exists():
            entry["roi"] = Quad.load(roi).to_list()
        entries.append(entry)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="_site")
    parser.add_argument("--model", help="YOLO11n ONNX file (exported with imgsz=640)")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(WEB / "index.html", out / "index.html")
    shutil.copytree(WEB / "css", out / "css")
    shutil.copytree(WEB / "js", out / "js")
    shutil.copytree(WEB / "assets", out / "assets")
    (out / ".nojekyll").touch()

    vendor = out / "vendor"
    vendor.mkdir()
    for src, name in VENDOR_FILES.items():
        path = WEB / "node_modules" / src
        if not path.exists():
            raise SystemExit(f"Missing {path}: run `npm ci` in web/ first")
        shutil.copy2(path, vendor / name)

    model = None
    if args.model:
        (out / "models").mkdir()
        shutil.copy2(args.model, out / "models" / "yolo11n.onnx")
        model = "models/yolo11n.onnx"

    demos = []
    for demo in DEMOS:
        print(f"Rendering {demo['id']}")
        demos.append(build_demo(demo, out))
    print("Rendering the landing-page clip")
    first = demos[0]
    build_hero(out, out / first["sources"][0]["src"], out / first["boxes"], first["roi"])
    if model:  # real clips need the detector
        demos += build_examples(out)
    (out / "config.json").write_text(json.dumps({"model": model, "demos": demos}, indent=2))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Site written to {out}/ ({size:.0f} MB)")


if __name__ == "__main__":
    main()
