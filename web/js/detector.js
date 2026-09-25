// Person detection in the browser: YOLO11n (ONNX) through ONNX Runtime Web.
// WebGPU when the browser has it, WebAssembly otherwise.

const INPUT = 640;
const PAD = 114; // Ultralytics letterbox grey

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${url} (${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded / total);
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

async function webgpuAvailable() {
  try {
    return Boolean(navigator.gpu && (await navigator.gpu.requestAdapter()));
  } catch {
    return false;
  }
}

async function loadOrt(vendorUrl, gpu) {
  const mod = await import(new URL(gpu ? "ort.webgpu.min.mjs" : "ort.wasm.min.mjs", vendorUrl).href);
  const ort = mod.default ?? mod;
  ort.env.wasm.wasmPaths = vendorUrl;
  // Multi-threading needs cross-origin isolation, which static hosts rarely provide.
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  return ort;
}

function nms(boxes, iouThreshold, maxDet) {
  boxes.sort((a, b) => b.conf - a.conf);
  const kept = [];
  for (const b of boxes) {
    let keep = true;
    for (const k of kept) {
      const iw = Math.max(0, Math.min(b.box[2], k.box[2]) - Math.max(b.box[0], k.box[0]));
      const ih = Math.max(0, Math.min(b.box[3], k.box[3]) - Math.max(b.box[1], k.box[1]));
      const inter = iw * ih;
      const union =
        (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) + (k.box[2] - k.box[0]) * (k.box[3] - k.box[1]) - inter;
      if (union > 0 && inter / union > iouThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) kept.push(b);
    if (kept.length >= maxDet) break;
  }
  return kept;
}

/** Decode a YOLO11 [1, 84, N] output (cx, cy, w, h, 80 class scores) for class 0 (person). */
export function decodePeople(data, anchors, conf, letterbox, iouThreshold = 0.7, maxDet = 100) {
  const { scale, left, top, x0, y0 } = letterbox;
  const found = [];
  for (let i = 0; i < anchors; i++) {
    const score = data[4 * anchors + i];
    if (score < conf) continue;
    const cx = data[i];
    const cy = data[anchors + i];
    const w = data[2 * anchors + i];
    const h = data[3 * anchors + i];
    found.push({
      box: [
        (cx - w / 2 - left) / scale + x0,
        (cy - h / 2 - top) / scale + y0,
        (cx + w / 2 - left) / scale + x0,
        (cy + h / 2 - top) / scale + y0,
      ],
      conf: score,
    });
  }
  return nms(found, iouThreshold, maxDet).map((d) => ({ box: d.box.map(Math.round), conf: d.conf }));
}

export class YoloDetector {
  static async create({ vendorUrl, modelUrl, conf = 0.35, onProgress }) {
    const bytes = await fetchWithProgress(modelUrl, (p) => onProgress?.(`Downloading the person detector`, p));
    onProgress?.("Starting the person detector", 1);
    const gpu = await webgpuAvailable();
    if (gpu) {
      try {
        const ort = await loadOrt(vendorUrl, true);
        const session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
        return new YoloDetector(ort, session, "WebGPU", conf, bytes, vendorUrl);
      } catch (err) {
        console.warn("WebGPU unavailable for this model, using WebAssembly", err);
      }
    }
    const ort = await loadOrt(vendorUrl, false);
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    return new YoloDetector(ort, session, "WebAssembly", conf, bytes, vendorUrl);
  }

  constructor(ort, session, backend, conf, bytes, vendorUrl) {
    this.ort = ort;
    this.session = session;
    this.backend = backend;
    this.conf = conf;
    this.bytes = bytes;
    this.vendorUrl = vendorUrl;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = INPUT;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.input = new Float32Array(3 * INPUT * INPUT);
  }

  /** People in `source` (a canvas), looking only at `region` {x0, y0, x1, y1}. */
  async detect(source, region) {
    const { x0, y0, x1, y1 } = region;
    const rw = x1 - x0;
    const rh = y1 - y0;
    const scale = Math.min(INPUT / rw, INPUT / rh);
    const nw = Math.round(rw * scale);
    const nh = Math.round(rh * scale);
    const left = Math.round((INPUT - nw) / 2 - 0.1);
    const top = Math.round((INPUT - nh) / 2 - 0.1);
    const ctx = this.ctx;
    ctx.fillStyle = `rgb(${PAD},${PAD},${PAD})`;
    ctx.fillRect(0, 0, INPUT, INPUT);
    ctx.drawImage(source, x0, y0, rw, rh, left, top, nw, nh);
    const pixels = ctx.getImageData(0, 0, INPUT, INPUT).data;
    const plane = INPUT * INPUT;
    const input = this.input;
    for (let i = 0; i < plane; i++) {
      input[i] = pixels[4 * i] / 255;
      input[plane + i] = pixels[4 * i + 1] / 255;
      input[2 * plane + i] = pixels[4 * i + 2] / 255;
    }
    const tensor = new this.ort.Tensor("float32", input, [1, 3, INPUT, INPUT]);
    let output;
    try {
      output = (await this.session.run({ [this.session.inputNames[0]]: tensor }))[this.session.outputNames[0]];
    } catch (err) {
      if (this.backend !== "WebGPU") throw err;
      console.warn("WebGPU inference failed, switching to WebAssembly", err);
      this.ort = await loadOrt(this.vendorUrl, false);
      this.session = await this.ort.InferenceSession.create(this.bytes, { executionProviders: ["wasm"] });
      this.backend = "WebAssembly";
      return this.detect(source, region);
    }
    const data = output.location && output.location !== "cpu" ? await output.getData() : output.data;
    const people = decodePeople(data, output.dims[2], this.conf, { scale, left, top, x0, y0 });
    output.dispose?.();
    return people;
  }
}

/** Replays person boxes recorded with a synthetic clip (1-based source frame numbers). */
export class ReplayDetector {
  constructor(boxesByFrame, scale = 1) {
    this.boxes = boxesByFrame;
    this.scale = scale;
    this.backend = "recorded boxes";
  }

  async detect(_source, _region, frameIndex) {
    return (this.boxes[frameIndex] || []).map((b) => ({ box: b.slice(0, 4).map((v) => v * this.scale), conf: 1 }));
  }
}
