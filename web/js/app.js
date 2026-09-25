// Browser app: choose a clip, mark the escalator, analyse it frame by frame.
import { TimelineChart, STATE_LABELS } from "./chart.js";
import {
  DEFAULTS,
  EscalatorMonitor,
  SessionStats,
  State,
  evaluate,
  formatTime,
  makeQuad,
  orderCorners,
  quadRegions,
  scaleQuad,
} from "./core.js";
import { ReplayDetector, YoloDetector } from "./detector.js";
import { makeFarnebackFlow, readyOpenCV } from "./flow.js";
import { Renderer, STATE_COLORS } from "./render.js";
import { estimateFps, loadVideo, probeFps, seek } from "./video.js";

const SITE = new URL("../", import.meta.url);
const VENDOR = new URL("vendor/", SITE).href;
const PROCESS_WIDTH = 960; // frames are analysed at most this wide
const ANALYSIS_FPS = 15; // sample at most this many frames per second of video
const $ = (id) => document.getElementById(id);

const ui = {
  fileInput: $("file-input"),
  dropZone: $("drop-zone"),
  demoList: $("demo-list"),
  stepRoi: $("step-roi"),
  roiCanvas: $("roi-canvas"),
  roiHint: $("roi-hint"),
  roiText: $("roi-text"),
  undo: $("undo-btn"),
  reset: $("reset-btn"),
  sensitivity: $("sensitivity"),
  sensitivityValue: $("sensitivity-value"),
  maxSeconds: $("max-seconds"),
  maxSecondsValue: $("max-seconds-value"),
  detectEvery: $("detect-every"),
  direction: $("direction"),
  shake: $("shake"),
  analyse: $("analyse-btn"),
  cancel: $("cancel-btn"),
  progress: $("progress"),
  progressFill: $("progress-fill"),
  progressText: $("progress-text"),
  error: $("error"),
  results: $("results"),
  replayCanvas: $("replay-canvas"),
  play: $("play-btn"),
  slider: $("replay-slider"),
  replayTime: $("replay-time"),
  finalState: $("final-state"),
  tiles: $("tiles"),
  gtNote: $("gt-note"),
  runInfo: $("run-info"),
  eventsBody: $("events-body"),
  snapshots: $("snapshots"),
  downloads: $("downloads"),
  video: $("video"),
};

let site = { model: null, demos: [] };
let clip = null; // the loaded video and its metadata
let points = []; // ROI corners in processing pixels
let busy = false;
let cancelRequested = false;
let run = null; // last analysis: frames, timeline, events, summary
const chart = new TimelineChart($("chart"), { onSeek: (t) => showFrameAt(t) });

// --- helpers ---------------------------------------------------------------------------

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = !message;
}

function setProgress(fraction, text) {
  ui.progress.hidden = false;
  ui.progressFill.style.width = `${Math.round(100 * Math.max(0, Math.min(1, fraction)))}%`;
  ui.progressText.textContent = text;
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return res.blob();
  const reader = res.body.getReader();
  const parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    onProgress(loaded / total);
  }
  return new Blob(parts);
}

let cvPromise = null;
function getOpenCV(onProgress) {
  cvPromise ??= (async () => {
    const blob = await fetchWithProgress(`${VENDOR}opencv.js`, (p) => onProgress?.("Downloading OpenCV.js", p));
    await new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = URL.createObjectURL(blob);
      s.onload = resolve;
      s.onerror = () => reject(new Error("Could not load OpenCV.js"));
      document.head.append(s);
    });
    onProgress?.("Starting OpenCV.js", 1);
    return readyOpenCV(window.cv);
  })();
  cvPromise.catch(() => (cvPromise = null));
  return cvPromise;
}

let yoloPromise = null;
function getYolo(onProgress) {
  if (!site.model) return Promise.reject(new Error("This build has no person detector model."));
  yoloPromise ??= YoloDetector.create({
    vendorUrl: VENDOR,
    modelUrl: new URL(site.model, SITE).href,
    conf: DEFAULTS.personConf,
    onProgress,
  });
  yoloPromise.catch(() => (yoloPromise = null));
  return yoloPromise;
}

function download(name, content, type) {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.textContent = name;
  return a;
}

const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
const toCsv = (header, rows) => [header, ...rows].map((r) => r.map(csvCell).join(",")).join("\n") + "\n";

function saveRoi() {
  if (!clip?.key) return;
  try {
    if (points.length === 4) localStorage.setItem(clip.key, JSON.stringify(scaleQuad(points, 1 / clip.scale)));
  } catch {
    /* storage unavailable: nothing to remember */
  }
}

function loadSavedRoi() {
  try {
    const saved = clip?.key && JSON.parse(localStorage.getItem(clip.key) || "null");
    return Array.isArray(saved) && saved.length === 4 ? saved : null;
  } catch {
    return null;
  }
}

// --- choosing a clip -------------------------------------------------------------------

async function openClip({ url, name, file = null, demo = null }) {
  if (busy) return;
  showError("");
  stopReplay();
  ui.results.hidden = true;
  ui.progress.hidden = true;
  try {
    await loadVideo(ui.video, url);
  } catch (err) {
    showError(err.message);
    return;
  }
  const v = ui.video;
  let fps = demo?.fps ?? null;
  if (!fps && file) fps = await probeFps(file);
  if (!fps) fps = await estimateFps(v);
  const scale = Math.min(1, PROCESS_WIDTH / v.videoWidth);
  clip = {
    url,
    name,
    demo,
    fps: fps || 25,
    fpsKnown: Boolean(fps),
    duration: v.duration,
    width: v.videoWidth,
    height: v.videoHeight,
    scale,
    procW: Math.round(v.videoWidth * scale),
    procH: Math.round(v.videoHeight * scale),
    key: file ? `escalator-roi:${file.name}:${file.size}` : null,
  };
  await seek(v, 0.5 / clip.fps);
  const first = document.createElement("canvas");
  first.width = clip.procW;
  first.height = clip.procH;
  first.getContext("2d").drawImage(v, 0, 0, clip.procW, clip.procH);
  clip.firstFrame = first;

  const roi = demo?.roi ?? loadSavedRoi();
  points = roi ? orderCorners(scaleQuad(roi, scale)) : [];
  ui.stepRoi.hidden = false;
  const seconds = Number.isFinite(clip.duration) ? `${clip.duration.toFixed(1)} s` : "unknown length";
  const fpsText = clip.fpsKnown ? `${clip.fps.toFixed(clip.fps % 1 ? 2 : 0)} fps` : "frame rate unknown, assuming 25 fps";
  ui.roiHint.textContent =
    `${name}: ${clip.width}x${clip.height}, ${fpsText}, ${seconds}. ` +
    (points.length === 4
      ? "Escalator region filled in (orange = handrails, blue = steps). Click to redraw it."
      : "Click the four corners of the escalator, including both handrails.");
  drawRoi();
  ui.stepRoi.scrollIntoView({ behavior: "smooth", block: "start" });
  // Warm up the heavy libraries while the user marks the escalator.
  getOpenCV().catch(() => {});
  if (!demo?.boxes) getYolo().catch(() => {});
}

function onFile(file) {
  if (!file) return;
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|m4v|webm|mkv)$/i.test(file.name)) {
    showError("That doesn't look like a video file.");
    return;
  }
  openClip({ url: URL.createObjectURL(file), name: file.name, file });
}

async function openDemo(demo) {
  const probe = document.createElement("video");
  const source = demo.sources.find((s) => probe.canPlayType(s.type)) || demo.sources[0];
  showError("");
  try {
    // Load into memory: blob URLs are always seekable, even from hosts without HTTP range support.
    const res = await fetch(new URL(source.src, SITE));
    if (!res.ok) throw new Error(`Could not download the example (${res.status})`);
    const url = URL.createObjectURL(await res.blob());
    await openClip({ url, name: demo.title, demo });
  } catch (err) {
    showError(err.message);
  }
}

// --- ROI picking ---------------------------------------------------------------------

function drawRoi() {
  const c = ui.roiCanvas;
  if (!clip) return;
  c.width = clip.procW;
  c.height = clip.procH;
  const ctx = c.getContext("2d");
  ctx.drawImage(clip.firstFrame, 0, 0);
  const s = Math.max(1, Math.min(c.width, c.height) / 480);
  let pts = points;
  if (points.length === 4) {
    pts = orderCorners(points);
    const regions = quadRegions(pts, DEFAULTS.handrailWidthFrac);
    for (const [key, poly] of Object.entries(regions)) {
      ctx.fillStyle = key === "steps" ? "rgba(57, 135, 229, 0.28)" : "rgba(236, 131, 90, 0.34)";
      ctx.beginPath();
      poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fill();
    }
  }
  if (pts.length >= 2) {
    ctx.strokeStyle = pts.length === 4 ? "#2ee66b" : "#ffd23f";
    ctx.lineWidth = 2 * s;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (pts.length === 4) ctx.closePath();
    ctx.stroke();
  }
  const labels = ["TL", "TR", "BR", "BL"];
  ctx.font = `700 ${Math.round(13 * s)}px system-ui, sans-serif`;
  pts.forEach(([x, y], i) => {
    ctx.fillStyle = "#2ee66b";
    ctx.beginPath();
    ctx.arc(x, y, 5.5 * s, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(0,0,0,0.6)";
    const label = pts.length === 4 ? labels[i] : String(i + 1);
    ctx.strokeText(label, x + 8 * s, y - 8 * s);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 8 * s, y - 8 * s);
  });
  ui.roiText.value =
    points.length === 4
      ? scaleQuad(orderCorners(points), 1 / clip.scale)
          .map(([x, y]) => `${Math.round(x)},${Math.round(y)}`)
          .join(", ")
      : "";
}

ui.roiCanvas.addEventListener("click", (evt) => {
  if (!clip || busy) return;
  const rect = ui.roiCanvas.getBoundingClientRect();
  const x = ((evt.clientX - rect.left) * ui.roiCanvas.width) / rect.width;
  const y = ((evt.clientY - rect.top) * ui.roiCanvas.height) / rect.height;
  points = points.length >= 4 ? [[x, y]] : [...points, [x, y]];
  if (points.length === 4) {
    try {
      points = makeQuad(points);
      saveRoi();
    } catch (err) {
      showError(`${err.message}. Click the corners again.`);
      points = [];
    }
  }
  drawRoi();
});
ui.undo.addEventListener("click", () => {
  points = points.length === 4 ? [] : points.slice(0, -1);
  drawRoi();
});
ui.reset.addEventListener("click", () => {
  points = [];
  drawRoi();
});
ui.roiText.addEventListener("change", () => {
  if (!clip) return;
  const nums = ui.roiText.value.split(/[\s,;]+/).filter(Boolean).map(Number);
  try {
    if (nums.length !== 8 || nums.some((n) => !Number.isFinite(n))) throw new Error("Enter 8 numbers: x1,y1, ..., x4,y4");
    points = makeQuad(
      scaleQuad(
        [0, 2, 4, 6].map((i) => [nums[i], nums[i + 1]]),
        clip.scale,
      ),
    );
    saveRoi();
    showError("");
  } catch (err) {
    showError(err.message);
  }
  drawRoi();
});

// --- analysis --------------------------------------------------------------------------

function settings() {
  const stride = Math.max(1, Math.round(clip.fps / ANALYSIS_FPS));
  const analysisFps = clip.fps / stride;
  return {
    cfg: {
      ...DEFAULTS,
      moveConfidenceMin: Number(ui.sensitivity.value),
      detectEvery: Number(ui.detectEvery.value),
      compensateCameraMotion: ui.shake.checked,
      expectedDirection: ui.direction.value,
      // Keep the Python defaults' time scale (45 frames at 25 fps = 1.8 s).
      windowSize: Math.max(9, Math.round((DEFAULTS.windowSize * analysisFps) / 25)),
      showDebug: true,
    },
    stride,
    analysisFps,
    maxSeconds: Number(ui.maxSeconds.value),
  };
}

function canvasToBlob(canvas) {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.82));
}

async function analyse() {
  if (!clip || busy) return;
  showError("");
  busy = true;
  cancelRequested = false;
  stopReplay();
  ui.analyse.disabled = true;
  ui.cancel.hidden = false;
  const notes = [];
  try {
    let quadPoints = points;
    if (quadPoints.length !== 4) {
      const { procW: w, procH: h } = clip;
      quadPoints = [
        [w * 0.25, h * 0.2],
        [w * 0.75, h * 0.2],
        [w * 0.75, h * 0.95],
        [w * 0.25, h * 0.95],
      ];
      notes.push("No escalator region was marked, so a central default region was used.");
    }
    const quad = makeQuad(quadPoints);
    const { cfg, stride, analysisFps, maxSeconds } = settings();

    const cv = await getOpenCV((stage, p) => setProgress(p * 0.5, `${stage} (${Math.round(p * 100)}%)`));
    let detector;
    if (clip.demo?.boxes) {
      const boxes = await (await fetch(new URL(clip.demo.boxes, SITE))).json();
      detector = new ReplayDetector(boxes, clip.scale);
      notes.push("Synthetic clip: the person boxes come from the simulator, because YOLO does not recognise the drawn figures.");
    } else {
      detector = await getYolo((stage, p) => setProgress(0.5 + p * 0.5, `${stage} (${Math.round(p * 100)}%)`));
    }

    const { procW: w, procH: h } = clip;
    const frame = document.createElement("canvas");
    frame.width = w;
    frame.height = h;
    const frameCtx = frame.getContext("2d", { willReadFrequently: true });
    const out = ui.replayCanvas;
    out.width = w;
    out.height = h;
    const outCtx = out.getContext("2d");

    const monitor = new EscalatorMonitor(cfg, quad, w, h, makeFarnebackFlow(cv));
    const renderer = new Renderer(cfg, quad, w, h);
    const [fw, fh] = monitor.motion.size;
    const flowCanvas = document.createElement("canvas");
    flowCanvas.width = fw;
    flowCanvas.height = fh;
    const flowCtx = flowCanvas.getContext("2d", { willReadFrequently: true });
    flowCtx.imageSmoothingQuality = "high";

    const end = Math.min(clip.duration, maxSeconds);
    if (clip.duration > maxSeconds) notes.push(`Only the first ${maxSeconds} s were analysed (see Settings).`);
    const lastFrame = Math.max(1, Math.floor(end * clip.fps + 1e-6));
    const stats = new SessionStats();
    const frames = [];
    const timeline = [];
    const events = [];
    const snapshots = [];
    let wrongActive = false;
    const t0 = performance.now();
    ui.results.hidden = false;
    ui.play.disabled = true;

    for (let src = 0; src < lastFrame && !cancelRequested; src += stride) {
      const time = src / clip.fps;
      await seek(ui.video, (src + 0.5) / clip.fps);
      frameCtx.drawImage(ui.video, 0, 0, w, h);

      const { x0, y0, x1, y1 } = monitor.motion.window;
      flowCtx.drawImage(frame, x0, y0, x1 - x0, y1 - y0, 0, 0, fw, fh);
      const px = flowCtx.getImageData(0, 0, fw, fh).data;
      const gray = new Uint8Array(fw * fh);
      for (let i = 0; i < gray.length; i++) gray[i] = (px[4 * i] * 77 + px[4 * i + 1] * 150 + px[4 * i + 2] * 29) >> 8;

      const detections = monitor.wantsDetection ? await detector.detect(frame, monitor.detectRegion, src + 1) : null;
      const r = monitor.step({ index: src + 1, time, gray, detections, stride });
      renderer.draw(outCtx, frame, r, end);
      const blob = await canvasToBlob(out);
      frames.push({ t: time, blob });

      let snapshot = "";
      if (r.previous) {
        if (r.state === State.STOPPED) {
          snapshot = `fault_${formatTime(time).replace(/[:.]/g, "-")}.jpg`;
          snapshots.push({ t: time, blob, name: snapshot });
        }
        events.push({ r, event: `${r.previous} -> ${r.state}`, snapshot });
      }
      if (r.wrongDirection && !wrongActive) {
        stats.wrongDirectionAlerts += 1;
        events.push({ r, event: "WRONG DIRECTION", snapshot: "" });
      }
      wrongActive = r.wrongDirection;
      stats.update(r, snapshot);
      timeline.push({
        t: time,
        state: r.state,
        conf: r.motion.confidence,
        people: r.people.length,
        moving: r.motion.isMoving,
        hr: r.motion.handrailScore,
        st: r.motion.stepsScore,
        mr: r.window.movingRatio,
        pr: r.window.peopleRatio,
        dir: r.direction,
      });

      const elapsed = (performance.now() - t0) / 1000;
      setProgress(time / end, `Analysing ${time.toFixed(1)} / ${end.toFixed(1)} s  (${(frames.length / elapsed).toFixed(1)} frames/s, ${detector.backend})`);
    }
    if (!frames.length) throw new Error("No frames could be read from this video.");
    stats.finish(stride / clip.fps);
    const elapsed = (performance.now() - t0) / 1000;
    const summary = {
      source: clip.name,
      generated_at: new Date().toISOString().slice(0, 19),
      ...stats.summary(),
      processing_fps: +(frames.length / elapsed).toFixed(2),
      analysis_fps: +analysisFps.toFixed(2),
      frame_size: [w, h],
      roi: scaleQuad(quad, 1 / clip.scale).map((p) => p.map(Math.round)),
      detector: detector.backend,
      flow_algorithm: "Farneback (OpenCV.js)",
      config: cfg,
    };
    if (cancelRequested) notes.push("Stopped early: results cover the part analysed so far.");
    run = { frames, timeline, events, snapshots, summary, cfg, analysisFps, notes, end };
    setProgress(1, `Done: ${frames.length} frames in ${elapsed.toFixed(1)} s`);
    showResults();
  } catch (err) {
    console.error(err);
    showError(err.message || String(err));
  } finally {
    busy = false;
    ui.analyse.disabled = false;
    ui.cancel.hidden = true;
  }
}

// --- results -----------------------------------------------------------------------------

function tile(label, value, sub = "", color = "") {
  const dot = color ? `<i class="state-dot" style="background:${color}"></i>` : "";
  return `<div class="tile"><div class="label">${dot}${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
}

function showResults() {
  const { summary: s, timeline, events, snapshots } = run;
  const final = s.final_state;
  ui.finalState.innerHTML = `<i class="state-dot" style="background:${STATE_COLORS[final]}"></i>Final state: ${STATE_LABELS[final] || final}`;
  const secs = s.time_in_state_s;
  const pct = s.time_in_state_pct;
  const stateTile = (st) => tile(STATE_LABELS[st], `${Math.round(pct[st] || 0)}%`, `${(secs[st] || 0).toFixed(1)} s`, STATE_COLORS[st]);
  ui.tiles.innerHTML =
    stateTile(State.WORKING) +
    stateTile(State.STOPPED) +
    stateTile(State.IDLE) +
    tile("Fault episodes", String(s.fault_count), s.fault_count ? `first at ${s.faults[0].start_s.toFixed(1)} s` : "none") +
    tile("Availability", s.availability_pct === null ? "-" : `${s.availability_pct}%`, "working vs. stopped time") +
    tile("Speed", `${s.processing_fps}`, `frames/s on this device (${s.detector})`);

  ui.gtNote.hidden = true;
  if (clip.demo?.groundTruth) {
    const truth = clip.demo.groundTruth.map(([a, b, st]) => [a, b, st]);
    const e = evaluate(
      timeline.map((r) => [r.t, r.state]),
      truth,
    );
    const lat = e.meanLatency[State.STOPPED];
    ui.gtNote.hidden = false;
    ui.gtNote.textContent =
      `Against this clip's ground truth: ${(100 * e.accuracy).toFixed(1)}% of frames correct` +
      (lat !== undefined ? `, fault confirmed ${lat.toFixed(2)} s after the stop` : "") +
      `, ${e.falseAlarms} false alarms. The delay is the deliberate confirmation window.`;
  }
  ui.runInfo.textContent = [
    `${s.frames_analyzed} frames analysed at ${run.analysisFps.toFixed(1)} fps, ${s.frame_size[0]}x${s.frame_size[1]}.`,
    ...run.notes,
  ].join(" ");

  chart.setData(timeline, run.cfg.moveConfidenceMin);

  ui.eventsBody.replaceChildren();
  if (!events.length) ui.eventsBody.innerHTML = `<tr><td colspan="4">No state changes.</td></tr>`;
  for (const { r, event } of events) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${formatTime(r.time)}</td><td class="change">${event}</td><td>${r.people.length}</td><td>${r.motion.confidence.toFixed(2)}</td>`;
    tr.addEventListener("click", () => showFrameAt(r.time));
    ui.eventsBody.append(tr);
  }

  ui.snapshots.replaceChildren();
  if (!snapshots.length) ui.snapshots.innerHTML = `<p class="hint">No faults detected.</p>`;
  for (const snap of snapshots) {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = URL.createObjectURL(snap.blob);
    img.alt = `Fault at ${snap.t.toFixed(1)} s`;
    img.addEventListener("click", () => showFrameAt(snap.t));
    const cap = document.createElement("figcaption");
    cap.textContent = `Stopped at ${formatTime(snap.t)}`;
    fig.append(img, cap);
    ui.snapshots.append(fig);
  }

  const eventRows = events.map(({ r, event, snapshot }) => [
    r.index,
    formatTime(r.time),
    event,
    snapshot,
    r.previous || r.state,
    r.state,
    r.motion.confidence.toFixed(3),
    r.people.length,
  ]);
  const timelineRows = timeline.map((r) => [
    r.t.toFixed(3),
    r.state,
    r.conf.toFixed(3),
    r.moving ? 1 : 0,
    r.people,
    r.hr.toFixed(3),
    r.st.toFixed(3),
    r.mr.toFixed(3),
    r.pr.toFixed(3),
    r.dir,
  ]);
  const record = document.createElement("button");
  record.type = "button";
  record.textContent = "Save annotated video";
  record.title = "Plays the result once in real time while recording it";
  record.addEventListener("click", () => recordVideo(record));
  ui.downloads.replaceChildren(
    download(
      "events.csv",
      toCsv(["frame", "video_time", "event", "snapshot", "from_state", "to_state", "move_confidence", "people"], eventRows),
      "text/csv",
    ),
    download(
      "timeline.csv",
      toCsv(["time_s", "state", "move_confidence", "is_moving", "people", "handrail_score", "steps_score", "moving_ratio", "people_ratio", "direction"], timelineRows),
      "text/csv",
    ),
    download("report.json", JSON.stringify(s, null, 2), "application/json"),
    ...snapshots.map((snap) => download(snap.name, snap.blob)),
    record,
  );

  ui.slider.max = String(run.frames.length - 1);
  ui.play.disabled = false;
  showFrame(run.frames.length - 1);
  ui.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

// --- replay ----------------------------------------------------------------------------

let replayTimer = null;
let shownIndex = 0;

async function showFrame(i) {
  if (!run) return;
  shownIndex = Math.max(0, Math.min(run.frames.length - 1, i));
  const f = run.frames[shownIndex];
  const bmp = await createImageBitmap(f.blob);
  ui.replayCanvas.getContext("2d").drawImage(bmp, 0, 0);
  bmp.close();
  ui.slider.value = String(shownIndex);
  ui.replayTime.textContent = `${f.t.toFixed(2)} s`;
  chart.setCursor(f.t);
}

function showFrameAt(t) {
  if (!run) return;
  const i = run.frames.findIndex((f) => f.t >= t - 1e-6);
  stopReplay();
  showFrame(i < 0 ? run.frames.length - 1 : i);
}

function stopReplay() {
  clearTimeout(replayTimer);
  replayTimer = null;
  ui.play.textContent = "Play";
  ui.play.setAttribute("aria-label", "Play");
}

function playReplay(onEnd) {
  if (!run) return;
  if (shownIndex >= run.frames.length - 1) shownIndex = 0;
  ui.play.textContent = "Pause";
  ui.play.setAttribute("aria-label", "Pause");
  const period = 1000 / run.analysisFps;
  let next = performance.now();
  const tick = async () => {
    await showFrame(shownIndex);
    if (shownIndex >= run.frames.length - 1) {
      stopReplay();
      onEnd?.();
      return;
    }
    shownIndex += 1;
    next += period;
    replayTimer = setTimeout(tick, Math.max(0, next - performance.now()));
  };
  replayTimer = setTimeout(tick, 0);
}

ui.play.addEventListener("click", () => (replayTimer ? stopReplay() : playReplay()));
ui.slider.addEventListener("input", () => {
  stopReplay();
  showFrame(Number(ui.slider.value));
});

function recordVideo(button) {
  if (!run || !window.MediaRecorder) {
    showError("This browser cannot record video. Use the CSV/JSON downloads instead.");
    return;
  }
  const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
  const mimeType = types.find((t) => MediaRecorder.isTypeSupported(t)) || "";
  const stream = ui.replayCanvas.captureStream(Math.round(run.analysisFps));
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
  const chunks = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    const ext = recorder.mimeType.includes("mp4") ? "mp4" : "webm";
    const a = download(`escalator-annotated.${ext}`, new Blob(chunks, { type: recorder.mimeType }));
    a.click();
    button.disabled = false;
    button.textContent = "Save annotated video";
  };
  button.disabled = true;
  button.textContent = "Recording...";
  stopReplay();
  shownIndex = 0;
  recorder.start();
  playReplay(() => setTimeout(() => recorder.stop(), 200));
}

// --- wiring --------------------------------------------------------------------------------

ui.fileInput.addEventListener("change", () => onFile(ui.fileInput.files[0]));
for (const evt of ["dragenter", "dragover"]) {
  ui.dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    ui.dropZone.classList.add("over");
  });
}
for (const evt of ["dragleave", "drop"]) {
  ui.dropZone.addEventListener(evt, () => ui.dropZone.classList.remove("over"));
}
ui.dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  onFile(e.dataTransfer.files[0]);
});
ui.sensitivity.addEventListener("input", () => (ui.sensitivityValue.value = Number(ui.sensitivity.value).toFixed(2)));
ui.maxSeconds.addEventListener("input", () => (ui.maxSecondsValue.value = `${ui.maxSeconds.value} s`));
ui.analyse.addEventListener("click", analyse);
ui.cancel.addEventListener("click", () => (cancelRequested = true));

async function init() {
  try {
    site = await (await fetch(new URL("config.json", SITE))).json();
  } catch {
    showError("Could not load the site configuration.");
    return;
  }
  for (const demo of site.demos) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "demo";
    b.dataset.demoId = demo.id;
    const poster = demo.poster ? `<img src="${new URL(demo.poster, SITE).href}" alt="" loading="lazy" />` : "<span></span>";
    b.innerHTML = `${poster}<div><span class="tag">${demo.tag || "Example"}</span><strong>${demo.title}</strong><span>${demo.description}</span></div>`;
    b.addEventListener("click", () => openDemo(demo));
    ui.demoList.append(b);
  }
  if (!site.model) {
    ui.dropZone.querySelector("span").textContent = "Uploads need the person detector, which this build does not include.";
  }
}

$("hero-demo").addEventListener("click", () => {
  if (site.demos.length) openDemo(site.demos[0]);
  else $("try").scrollIntoView({ behavior: "smooth" });
});
$("hero-upload").addEventListener("click", () => {
  $("try").scrollIntoView({ behavior: "smooth" });
  ui.fileInput.click();
});

// Small hook for automated browser tests.
window.escalatorMonitor = {
  get run() {
    return run;
  },
  get busy() {
    return busy;
  },
};

init();
