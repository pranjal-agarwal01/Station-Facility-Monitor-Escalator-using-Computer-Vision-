// Core escalator-monitoring logic for the browser app.
//
// A line-by-line port of the Python package (escalator_monitor/): geometry,
// flow statistics, scoring, tracking and the state machine. Nothing here
// touches the DOM, so it runs (and is tested) in Node as well.

export const State = Object.freeze({
  INITIALIZING: "INITIALIZING",
  WORKING: "WORKING",
  STOPPED: "STOPPED / FAULT",
  IDLE: "IDLE",
});

// Same defaults as escalator_monitor/config.py.
export const DEFAULTS = Object.freeze({
  personConf: 0.35,
  personMaskPadding: 18,
  detectEvery: 3,
  detectRoiMargin: 0.15,
  flowDownscale: 0.5,
  flowRoiMargin: 0.15,
  compensateCameraMotion: false,
  handrailMagGate: 0.1,
  stepsMagGate: 0.15,
  consistencyGate: 0.55,
  handrailMagNorm: 0.3,
  stepsMagNorm: 0.5,
  consistencyNorm: 0.85,
  handrailWidthFrac: 0.1,
  requireVerticalMotion: true,
  verticalRatioMin: 1.5,
  requireHandrailAgreement: true,
  directionDotMin: 0.3,
  expectedDirection: "any",
  handrailWeight: 0.55,
  stepsWeight: 0.45,
  moveConfidenceMin: 0.35,
  strongRegionScore: 0.7,
  windowSize: 45,
  enterWorkingRatio: 0.45,
  exitWorkingRatio: 0.25,
  enterStoppedRatio: 0.6,
  idlePeopleRatio: 0.2,
  stoppedPeopleRatio: 0.5,
  enterWorkingScore: 0.5,
  resumeWorkingScore: 0.4,
  stoppedScoreMax: 0.15,
  faultScoreMax: 0.2,
});

const MIN_REGION_PIXELS = 40;
const MIN_MOVING_PIXELS = 10;
const MIN_SCORED_PIXELS = 200;
const DIRECTION_EPS = 0.05;

// --- geometry ------------------------------------------------------------------

/** Order four points clockwise on screen starting from the top-left one. */
export function orderCorners(points) {
  const pts = points.map((p) => [Number(p[0]), Number(p[1])]);
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  pts.sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
  let start = 0;
  for (let i = 1; i < pts.length; i++) {
    if (pts[i][0] + pts[i][1] < pts[start][0] + pts[start][1]) start = i;
  }
  return pts.slice(start).concat(pts.slice(0, start));
}

export function polygonArea(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

/** Validated, ordered ROI quad (TL, TR, BR, BL). */
export function makeQuad(points) {
  if (!Array.isArray(points) || points.length !== 4) throw new Error("The escalator region needs exactly 4 corners");
  const quad = orderCorners(points);
  if (polygonArea(quad) < 64) throw new Error("The escalator region is too small");
  return quad;
}

export function scaleQuad(quad, s) {
  return quad.map(([x, y]) => [x * s, y * s]);
}

const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

/** Left handrail, right handrail and steps polygons. */
export function quadRegions(quad, frac) {
  const [tl, tr, br, bl] = quad;
  const itl = lerp(tl, tr, frac);
  const ibl = lerp(bl, br, frac);
  const itr = lerp(tr, tl, frac);
  const ibr = lerp(br, bl, frac);
  return { left: [tl, itl, ibl, bl], right: [itr, tr, br, ibr], steps: [itl, itr, ibr, ibl] };
}

/** Bounding box grown by `margin` x its size per side, clipped to the frame. */
export function quadBBox(quad, margin = 0, width = Infinity, height = Infinity) {
  const xs = quad.map((p) => p[0]);
  const ys = quad.map((p) => p[1]);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const dx = (maxX - minX) * margin;
  const dy = (maxY - minY) * margin;
  return {
    x0: Math.max(0, Math.floor(minX - dx)),
    y0: Math.max(0, Math.floor(minY - dy)),
    x1: Math.min(width, Math.ceil(maxX + dx) + 1),
    y1: Math.min(height, Math.ceil(maxY + dy) + 1),
  };
}

export function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Fill a convex polygon (frame coordinates) into a width x height mask of a scaled crop. */
export function rasterize(poly, width, height, ox = 0, oy = 0, sx = 1, sy = 1) {
  const mask = new Uint8Array(width * height);
  const pts = poly.map(([x, y]) => [Math.round((x - ox) * sx), Math.round((y - oy) * sy)]);
  const ys = pts.map((p) => p[1]);
  const r0 = Math.max(0, Math.min(...ys));
  const r1 = Math.min(height - 1, Math.max(...ys));
  for (let r = r0; r <= r1; r++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < pts.length; i++) {
      const [x1, y1] = pts[i];
      const [x2, y2] = pts[(i + 1) % pts.length];
      if (r < Math.min(y1, y2) || r > Math.max(y1, y2)) continue;
      if (y1 === y2) {
        lo = Math.min(lo, x1, x2);
        hi = Math.max(hi, x1, x2);
      } else {
        const x = x1 + ((r - y1) * (x2 - x1)) / (y2 - y1);
        lo = Math.min(lo, x);
        hi = Math.max(hi, x);
      }
    }
    const c0 = Math.max(0, Math.ceil(lo - 1e-9));
    const c1 = Math.min(width - 1, Math.floor(hi + 1e-9));
    if (c1 >= c0) mask.fill(1, r * width + c0, r * width + c1 + 1);
  }
  return mask;
}

/** Square-kernel dilation of a 0/1 mask (separable max filter). */
export function dilate(mask, width, height, k) {
  const r = Math.floor(k / 2);
  const tmp = new Uint8Array(mask.length);
  const out = new Uint8Array(mask.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let v = 0;
      for (let d = Math.max(0, x - r); d <= Math.min(width - 1, x + r) && !v; d++) v = mask[y * width + d];
      tmp[y * width + x] = v;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let v = 0;
      for (let d = Math.max(0, y - r); d <= Math.min(height - 1, y + r) && !v; d++) v = tmp[d * width + x];
      out[y * width + x] = v;
    }
  }
  return out;
}

// --- flow statistics & scoring ----------------------------------------------------

/** Summarise interleaved (vx, vy) flow vectors where mask is non-zero. */
export function regionFlow(flow, mask, magFloor = 0.1) {
  let n = 0;
  let sumMag = 0;
  let k = 0;
  let sx = 0;
  let sy = 0;
  let sumStrong = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    n++;
    const vx = flow[2 * i];
    const vy = flow[2 * i + 1];
    const m = Math.hypot(vx, vy);
    sumMag += m;
    if (m > magFloor) {
      k++;
      sx += vx;
      sy += vy;
      sumStrong += m;
    }
  }
  const empty = { magnitude: 0, consistency: 0, vx: 0, vy: 0, pixels: n };
  if (n < MIN_REGION_PIXELS) return empty;
  if (k < MIN_MOVING_PIXELS) return { ...empty, magnitude: sumMag / n };
  return { magnitude: sumMag / n, consistency: Math.hypot(sx, sy) / (sumStrong + 1e-6), vx: sx / k, vy: sy / k, pixels: n };
}

function median(values) {
  if (!values.length) return 0;
  const s = Float32Array.from(values).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Median flow of static scenery (the camera's own motion); [0, 0] if unknown. */
export function backgroundMotion(flow, mask, minPixels = 200) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) {
      xs.push(flow[2 * i]);
      ys.push(flow[2 * i + 1]);
    }
  }
  if (xs.length < minPixels) return [0, 0];
  return [median(xs), median(ys)];
}

export function smoothScore(mag, cons, magGate, magNorm, consGate, consNorm) {
  if (mag < magGate || cons < consGate) return 0;
  const magTerm = mag / (mag + magNorm);
  const consTerm = cons / (cons + (1 - consNorm));
  return Math.sqrt(magTerm * Math.min(consTerm, 1));
}

export function directionPenalty(vx, vy, cfg) {
  if (!cfg.requireVerticalMotion) return 1;
  if (Math.abs(vy) < 1e-3) return 0;
  const ratio = Math.abs(vy) / (Math.abs(vx) + 1e-3);
  if (ratio < cfg.verticalRatioMin) return 0;
  return Math.min(1, ratio / (cfg.verticalRatioMin * 2));
}

export function handrailAgreement(left, right, cfg) {
  if (!cfg.requireHandrailAgreement) return 1;
  const ln = Math.hypot(left.vx, left.vy);
  const rn = Math.hypot(right.vx, right.vy);
  if (ln < 1e-3 || rn < 1e-3) return 0;
  const dot = (left.vx * right.vx + left.vy * right.vy) / (ln * rn);
  if (dot < cfg.directionDotMin) return 0;
  return Math.max(0, dot);
}

const countNonZero = (m) => {
  let n = 0;
  for (let i = 0; i < m.length; i++) n += m[i] ? 1 : 0;
  return n;
};

const andMask = (a, b) => {
  const out = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] & b[i];
  return out;
};

export const EMPTY_READING = Object.freeze({
  valid: false,
  handrailMag: 0,
  handrailCons: 0,
  stepsMag: 0,
  stepsCons: 0,
  handrailScore: 0,
  stepsScore: 0,
  directionGate: 0,
  handrailAgreement: 0,
  confidence: 0,
  isMoving: false,
  vy: 0,
  cameraShift: [0, 0],
});

/**
 * Surface motion between consecutive frames of the ROI crop.
 *
 * The caller supplies grey crops of `this.window` resized to `this.size`
 * and a `flowFn(prev, curr, w, h)` returning interleaved (vx, vy) flow.
 */
export class MotionAnalyzer {
  constructor(cfg, quad, frameWidth, frameHeight, flowFn) {
    this.cfg = cfg;
    this.frameWidth = frameWidth;
    this.frameHeight = frameHeight;
    this.flowFn = flowFn;
    this.setRoi(quad);
  }

  setRoi(quad) {
    const cfg = this.cfg;
    this.quad = quad;
    this.window = quadBBox(quad, cfg.flowRoiMargin, this.frameWidth, this.frameHeight);
    const { x0, y0, x1, y1 } = this.window;
    const w = Math.max(8, Math.round((x1 - x0) * cfg.flowDownscale));
    const h = Math.max(8, Math.round((y1 - y0) * cfg.flowDownscale));
    this.size = [w, h];
    this.scale = [w / (x1 - x0), h / (y1 - y0)];
    const regions = quadRegions(quad, cfg.handrailWidthFrac);
    const r = (poly) => rasterize(poly, w, h, x0, y0, this.scale[0], this.scale[1]);
    const roi = r(quad);
    const grow = Math.max(3, Math.round(0.02 * Math.max(w, h)) | 1);
    const near = dilate(roi, w, h, grow);
    this.masks = {
      left: r(regions.left),
      right: r(regions.right),
      steps: r(regions.steps),
      roi,
      background: near.map((v) => 1 - v),
    };
    this.prev = null;
  }

  reset() {
    this.prev = null;
  }

  usablePixels(boxes) {
    const [w, h] = this.size;
    const keep = new Uint8Array(w * h).fill(1);
    const pad = this.cfg.personMaskPadding;
    const { x0, y0 } = this.window;
    const [sx, sy] = this.scale;
    for (const [bx1, by1, bx2, by2] of boxes) {
      const c1 = Math.max(0, Math.floor((bx1 - pad - x0) * sx));
      const r1 = Math.max(0, Math.floor((by1 - pad - y0) * sy));
      const c2 = Math.min(w, Math.ceil((bx2 + pad - x0) * sx));
      const r2 = Math.min(h, Math.ceil((by2 + pad - y0) * sy));
      for (let y = r1; y < r2; y++) keep.fill(0, y * w + c1, y * w + Math.max(c1, c2));
    }
    return keep;
  }

  /** `gray`: Uint8Array crop (size[0] x size[1]); `stride`: source frames since the last crop. */
  update(gray, personBoxes = [], stride = 1) {
    const prev = this.prev;
    this.prev = gray;
    if (!prev) return EMPTY_READING;
    const cfg = this.cfg;
    const [w, h] = this.size;
    const flow = this.flowFn(prev, gray, w, h);
    if (stride > 1) for (let i = 0; i < flow.length; i++) flow[i] /= stride;

    const keep = this.usablePixels(personBoxes);
    let shift = [0, 0];
    if (cfg.compensateCameraMotion) {
      shift = backgroundMotion(flow, andMask(this.masks.background, keep));
      if (shift[0] || shift[1]) {
        for (let i = 0; i < flow.length; i += 2) {
          flow[i] -= shift[0];
          flow[i + 1] -= shift[1];
        }
      }
    }

    const leftMask = andMask(this.masks.left, keep);
    const rightMask = andMask(this.masks.right, keep);
    const stepsMask = andMask(this.masks.steps, keep);
    const left = regionFlow(flow, leftMask);
    const right = regionFlow(flow, rightMask);
    const steps = regionFlow(flow, stepsMask);

    // Handrails: average both sides; if one is hidden (crowd), trust the other.
    const rails = [left, right].filter((r) => r.pixels >= MIN_REGION_PIXELS);
    let agreement = 0;
    if (rails.length === 2) agreement = handrailAgreement(left, right, cfg);
    else if (rails.length === 1) agreement = 1;
    const avg = (key) => (rails.length ? rails.reduce((s, r) => s + r[key], 0) / rails.length : 0);
    const hrMag = avg("magnitude");
    const hrCons = avg("consistency");
    const hrVx = avg("vx");
    const hrVy = avg("vy");

    const directionGate = directionPenalty(hrVx, hrVy, cfg) * agreement;
    const stepsGate = directionPenalty(steps.vx, steps.vy, cfg);
    const hrPixels = countNonZero(leftMask) + countNonZero(rightMask);
    const stPixels = countNonZero(stepsMask);

    let hrScore = 0;
    let stScore = 0;
    if (hrPixels > MIN_SCORED_PIXELS) {
      hrScore =
        directionGate *
        smoothScore(hrMag, hrCons, cfg.handrailMagGate, cfg.handrailMagNorm, cfg.consistencyGate, cfg.consistencyNorm);
    }
    if (stPixels > MIN_SCORED_PIXELS) {
      stScore =
        stepsGate *
        smoothScore(steps.magnitude, steps.consistency, cfg.stepsMagGate, cfg.stepsMagNorm, cfg.consistencyGate, cfg.consistencyNorm);
    }
    let confidence = stPixels <= MIN_SCORED_PIXELS ? hrScore : cfg.handrailWeight * hrScore + cfg.stepsWeight * stScore;
    if (Math.max(hrScore, stScore) > cfg.strongRegionScore) confidence = Math.max(confidence, cfg.strongRegionScore);
    const weight = hrScore + stScore;
    return {
      valid: true,
      handrailMag: hrMag,
      handrailCons: hrCons,
      stepsMag: steps.magnitude,
      stepsCons: steps.consistency,
      handrailScore: hrScore,
      stepsScore: stScore,
      directionGate,
      handrailAgreement: agreement,
      confidence,
      isMoving: confidence >= cfg.moveConfidenceMin,
      vy: weight > 0 ? (hrScore * hrVy + stScore * steps.vy) / weight : 0,
      cameraShift: shift,
    };
  }
}

// --- tracking ---------------------------------------------------------------------

export function iou(a, b) {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/** Greedy IoU association; coasts between detector runs (`update(null)`). */
export class IoUTracker {
  constructor(iouThreshold = 0.3, maxMisses = 2) {
    this.iouThreshold = iouThreshold;
    this.maxMisses = maxMisses;
    this.tracks = new Map();
    this.nextId = 0;
  }

  reset() {
    this.tracks.clear();
  }

  /** detections: [{box: [x1, y1, x2, y2], conf}] or null when the detector did not run. */
  update(detections) {
    if (detections === null) return [...this.tracks.values()];
    const pairs = [];
    detections.forEach((d, di) => {
      for (const [tid, t] of this.tracks) pairs.push([iou(d.box, t.box), di, tid]);
    });
    pairs.sort((a, b) => b[0] - a[0]);
    const usedDets = new Set();
    const usedTracks = new Set();
    for (const [score, di, tid] of pairs) {
      if (score < this.iouThreshold) break;
      if (usedDets.has(di) || usedTracks.has(tid)) continue;
      const t = this.tracks.get(tid);
      t.box = detections[di].box;
      t.conf = detections[di].conf;
      t.misses = 0;
      usedDets.add(di);
      usedTracks.add(tid);
    }
    for (const [tid, t] of [...this.tracks]) {
      if (usedTracks.has(tid)) continue;
      t.misses += 1;
      if (t.misses > this.maxMisses) this.tracks.delete(tid);
    }
    detections.forEach((d, di) => {
      if (!usedDets.has(di)) this.tracks.set(this.nextId, { id: this.nextId++, box: d.box, conf: d.conf, misses: 0 });
    });
    return [...this.tracks.values()];
  }
}

// --- state machine ------------------------------------------------------------------

/** WORKING / STOPPED / IDLE with separate enter and exit thresholds (see state.py). */
export class StateMachine {
  constructor(cfg) {
    this.cfg = cfg;
    this.reset();
  }

  reset() {
    this.state = State.INITIALIZING;
    this.moving = [];
    this.people = [];
    this.scores = [];
  }

  stats() {
    const n = this.moving.length;
    if (!n) return { movingRatio: 0, peopleRatio: 0, meanScore: 0, frames: 0, recentPeopleRatio: 0 };
    const sum = (a) => a.reduce((s, v) => s + Number(v), 0);
    const k = Math.min(n, Math.max(1, Math.floor(this.cfg.windowSize / 3)));
    return {
      movingRatio: sum(this.moving) / n,
      peopleRatio: sum(this.people) / n,
      meanScore: sum(this.scores) / n,
      frames: n,
      recentPeopleRatio: sum(this.people.slice(-k)) / k,
    };
  }

  update(isMoving, peoplePresent, score) {
    const cfg = this.cfg;
    const push = (arr, v) => {
      arr.push(v);
      if (arr.length > cfg.windowSize) arr.shift();
    };
    push(this.moving, Boolean(isMoving));
    push(this.people, Boolean(peoplePresent));
    push(this.scores, Number(score));
    if (this.moving.length < Math.max(1, Math.floor(cfg.windowSize / 3))) return this.state;

    const w = this.stats();
    const { movingRatio: mr, peopleRatio: pr, meanScore: avg } = w;
    const state = this.state;
    let next = state;
    if (state === State.INITIALIZING || state === State.IDLE) {
      if (mr >= cfg.enterWorkingRatio || avg > cfg.enterWorkingScore) next = State.WORKING;
      else if (
        pr >= cfg.stoppedPeopleRatio &&
        w.recentPeopleRatio >= cfg.stoppedPeopleRatio && // people there now, not just earlier
        1 - mr >= cfg.enterStoppedRatio &&
        avg < cfg.stoppedScoreMax
      )
        next = State.STOPPED;
      else if (pr < cfg.idlePeopleRatio && state === State.INITIALIZING) next = State.IDLE;
    } else if (state === State.WORKING) {
      if (mr < cfg.exitWorkingRatio && avg < cfg.faultScoreMax) {
        // Occupancy from recent frames only: riders who stepped off just before
        // an energy-saving stop should not turn it into a fault.
        next = w.recentPeopleRatio >= cfg.idlePeopleRatio ? State.STOPPED : State.IDLE;
      }
    } else if (state === State.STOPPED) {
      if (mr >= cfg.enterWorkingRatio || avg > cfg.resumeWorkingScore) next = State.WORKING;
      else if (pr < cfg.idlePeopleRatio) next = State.IDLE;
    }
    if (pr < cfg.idlePeopleRatio && mr < cfg.exitWorkingRatio && avg < cfg.faultScoreMax) next = State.IDLE;
    this.state = next;
    return next;
  }
}

// --- per-frame pipeline -------------------------------------------------------------

/** Detection -> motion -> state for one analysed frame (see pipeline.py). */
export class EscalatorMonitor {
  constructor(cfg, quad, frameWidth, frameHeight, flowFn) {
    this.cfg = cfg;
    this.frameWidth = frameWidth;
    this.frameHeight = frameHeight;
    this.tracker = new IoUTracker(0.3, 2);
    this.motion = new MotionAnalyzer(cfg, quad, frameWidth, frameHeight, flowFn);
    this.states = new StateMachine(cfg);
    this.vy = [];
    this.processed = 0;
    this.setRoi(quad);
  }

  setRoi(quad) {
    this.quad = quad;
    this.motion.setRoi(quad);
    this.tracker.reset();
    this.detectRegion = quadBBox(quad, this.cfg.detectRoiMargin, this.frameWidth, this.frameHeight);
  }

  /** True when the detector should run on the next frame. */
  get wantsDetection() {
    return this.processed % this.cfg.detectEvery === 0;
  }

  inRoi(det) {
    const [x1, y1, x2, y2] = det.box;
    const cx = (x1 + x2) / 2;
    return pointInPolygon(cx, (y1 + y2) / 2, this.quad) || pointInPolygon(cx, y2, this.quad);
  }

  direction(reading) {
    if (reading.isMoving) {
      this.vy.push(reading.vy);
      if (this.vy.length > this.cfg.windowSize) this.vy.shift();
    }
    if (this.vy.length < 5) return "";
    const m = median(this.vy);
    if (Math.abs(m) < DIRECTION_EPS) return "";
    return m < 0 ? "up" : "down";
  }

  /** detections: array (detector ran on this frame) or null; gray: flow crop. */
  step({ index, time, gray, detections, stride = 1 }) {
    this.processed += 1;
    const tracks = this.tracker.update(detections === null ? null : detections.filter((d) => this.inRoi(d)));
    const reading = this.motion.update(
      gray,
      tracks.map((t) => t.box),
      stride,
    );
    const before = this.states.state;
    const state = this.states.update(reading.isMoving, tracks.length > 0, reading.confidence);
    let direction = this.direction(reading);
    if (state !== State.WORKING) direction = "";
    const wrong =
      this.cfg.expectedDirection !== "any" && state === State.WORKING && direction !== "" && direction !== this.cfg.expectedDirection;
    return {
      index,
      time,
      state,
      previous: state !== before ? before : null,
      people: tracks.map((t) => ({ ...t })),
      motion: reading,
      window: this.states.stats(),
      direction,
      wrongDirection: wrong,
    };
  }
}

// --- run summary & evaluation ---------------------------------------------------------

/** Time in each state, fault episodes and transitions (see events.py SessionStats). */
export class SessionStats {
  constructor() {
    this.durations = {};
    this.transitions = 0;
    this.faults = [];
    this.wrongDirectionAlerts = 0;
    this.frames = 0;
    this.last = null;
  }

  update(r, snapshot = "") {
    this.frames += 1;
    if (this.last) this.durations[this.last.state] = (this.durations[this.last.state] || 0) + Math.max(0, r.time - this.last.time);
    if (r.previous) {
      this.transitions += 1;
      const open = this.faults.at(-1);
      if (r.previous === State.STOPPED && open && open.end_s === null) this.closeFault(r.time);
      if (r.state === State.STOPPED) this.faults.push({ start_s: round(r.time, 3), end_s: null, frame: r.index, snapshot });
    }
    this.last = r;
  }

  closeFault(t) {
    const f = this.faults.at(-1);
    f.end_s = round(t, 3);
    f.duration_s = round(t - f.start_s, 3);
  }

  finish(framePeriod) {
    if (!this.last) return;
    this.durations[this.last.state] = (this.durations[this.last.state] || 0) + framePeriod;
    const open = this.faults.at(-1);
    if (open && open.end_s === null) {
      this.closeFault(this.last.time + framePeriod);
      open.ongoing = true;
    }
  }

  summary() {
    const total = Object.values(this.durations).reduce((s, v) => s + v, 0);
    const working = this.durations[State.WORKING] || 0;
    const stopped = this.durations[State.STOPPED] || 0;
    const pct = {};
    for (const [k, v] of Object.entries(this.durations)) pct[k] = total ? round((100 * v) / total, 1) : 0;
    const secs = {};
    for (const [k, v] of Object.entries(this.durations)) secs[k] = round(v, 2);
    return {
      frames_analyzed: this.frames,
      duration_s: round(total, 2),
      time_in_state_s: secs,
      time_in_state_pct: pct,
      // Availability ignores IDLE time: an empty, stopped escalator is not a failure.
      availability_pct: working + stopped > 0 ? round((100 * working) / (working + stopped), 1) : null,
      fault_count: this.faults.length,
      faults: this.faults,
      transitions: this.transitions,
      wrong_direction_alerts: this.wrongDirectionAlerts,
      final_state: this.last ? this.last.state : null,
    };
  }
}

/** Frame accuracy, time to detect each true change and false fault alarms (see evaluate.py). */
export function evaluate(timeline, intervals) {
  const truthAt = (t) => {
    for (const [s, e, st] of intervals) if (t >= s && t < e) return st;
    return null;
  };
  let frames = 0;
  let correct = 0;
  for (const [t, pred] of timeline) {
    const truth = truthAt(t);
    if (truth === null || pred === State.INITIALIZING) continue;
    frames++;
    if (truth === pred) correct++;
  }
  const latencies = {};
  let missed = 0;
  intervals.forEach(([start, end, state], i) => {
    if (i === 0 || intervals[i - 1][2] === state) return;
    const hit = timeline.find(([t, pred]) => t >= start && t < end && pred === state);
    if (!hit) missed++;
    else (latencies[state] ||= []).push(hit[0] - start);
  });
  const meanLatency = {};
  for (const [k, v] of Object.entries(latencies)) meanLatency[k] = round(v.reduce((s, x) => s + x, 0) / v.length, 3);
  let falseAlarms = 0;
  let episodeStart = null;
  for (const [t, pred] of [...timeline, [Infinity, State.INITIALIZING]]) {
    if (pred === State.STOPPED && episodeStart === null) episodeStart = t;
    else if (pred !== State.STOPPED && episodeStart !== null) {
      const overlaps = intervals.some(([s, e, st]) => s < t && e > episodeStart && st === State.STOPPED);
      if (!overlaps) falseAlarms++;
      episodeStart = null;
    }
  }
  return { frames, accuracy: frames ? correct / frames : 0, meanLatency, missed, falseAlarms };
}

export function round(v, digits) {
  const f = 10 ** digits;
  return Math.round(v * f) / f;
}

export function formatTime(seconds) {
  const ms = Math.round(Math.max(0, seconds) * 1000);
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
}
