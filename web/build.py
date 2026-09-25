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

from escalator_monitor.geometry import Quad
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
        "title": "Synthetic clip: running, fault, running, idle",
        "description": "27 s rendered scene with known ground truth. Quick to analyse.",
        "spec": SceneSpec(DEMO_SEGMENTS, width=640, height=480, seed=5),
    },
    {
        "id": "crowded-shaky",
        "title": "Synthetic clip: crowded, shaky camera",
        "description": "Busier scene with camera shake. Try it with shake compensation on and off.",
        "spec": SceneSpec(
            [Segment(6, True, 6), Segment(8, False, 5), Segment(6, True, 4)], jitter=1.0, noise=5.0, seed=9
        ),
    },
]


def to_webm(mp4: Path, webm: Path) -> bool:
    """VP9 copy for browsers without H.264 (e.g. open-source Chromium builds)."""
    try:
        import imageio_ffmpeg

        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(mp4),
               "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36", "-row-mt", "1", "-an", str(webm)]  # fmt: skip
        subprocess.run(cmd, check=True)
        return True
    except (ImportError, OSError, subprocess.CalledProcessError) as exc:
        print(f"  (no WebM copy: {exc})")
        return False


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
    return {
        "id": demo["id"],
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
        entry = {
            "id": clip.stem,
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
    if model:  # real clips need the detector
        demos += build_examples(out)
    (out / "config.json").write_text(json.dumps({"model": model, "demos": demos}, indent=2))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Site written to {out}/ ({size:.0f} MB)")


if __name__ == "__main__":
    main()
