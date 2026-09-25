// Unit tests for the browser port: npm test  (run from web/)
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { test } from "node:test";

import {
  DEFAULTS,
  EscalatorMonitor,
  IoUTracker,
  MotionAnalyzer,
  SessionStats,
  State,
  StateMachine,
  directionPenalty,
  evaluate,
  formatTime,
  handrailAgreement,
  iou,
  makeQuad,
  orderCorners,
  quadBBox,
  quadRegions,
  rasterize,
  regionFlow,
  smoothScore,
} from "../js/core.js";
import { makeFarnebackFlow, readyOpenCV } from "../js/flow.js";

const TRAPEZOID = [
  [40, 10],
  [60, 10],
  [90, 100],
  [10, 100],
];

test("corners are ordered regardless of click order", () => {
  for (const order of [
    [0, 1, 2, 3],
    [2, 0, 3, 1],
    [3, 2, 1, 0],
    [1, 3, 0, 2],
  ]) {
    assert.deepEqual(
      orderCorners(order.map((i) => TRAPEZOID[i])),
      TRAPEZOID,
    );
  }
  assert.throws(() => makeQuad([[0, 0], [1, 0], [1, 1], [0, 1]]));
});

test("bbox matches the Python implementation", () => {
  assert.deepEqual(quadBBox(TRAPEZOID), { x0: 10, y0: 10, x1: 91, y1: 101 });
  assert.deepEqual(quadBBox(TRAPEZOID, 0.5, 100, 105), { x0: 0, y0: 0, x1: 100, y1: 105 });
});

test("region masks partition the ROI", () => {
  const quad = makeQuad([[20, 10], [80, 10], [95, 190], [5, 190]]);
  const w = 100;
  const h = 200;
  const regions = quadRegions(quad, 0.1);
  const roi = rasterize(quad, w, h);
  const left = rasterize(regions.left, w, h);
  const right = rasterize(regions.right, w, h);
  const steps = rasterize(regions.steps, w, h);
  let union = 0;
  let roiCount = 0;
  let meanCol = { left: 0, right: 0, steps: 0 };
  const counts = { left: 0, right: 0, steps: 0 };
  for (let i = 0; i < roi.length; i++) {
    roiCount += roi[i];
    union += left[i] | right[i] | steps[i] ? 1 : 0;
    for (const [name, m] of Object.entries({ left, right, steps })) {
      if (m[i]) {
        meanCol[name] += i % w;
        counts[name]++;
      }
    }
  }
  meanCol = Object.fromEntries(Object.entries(meanCol).map(([k, v]) => [k, v / counts[k]]));
  assert.ok(union >= 0.97 * roiCount && union <= 1.03 * roiCount);
  assert.ok(meanCol.left < meanCol.steps && meanCol.steps < meanCol.right);
});

test("flow statistics", () => {
  const n = 400;
  const flow = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) flow[2 * i + 1] = -0.8;
  const mask = new Uint8Array(n).fill(1);
  const r = regionFlow(flow, mask);
  assert.ok(Math.abs(r.magnitude - 0.8) < 1e-6);
  assert.ok(Math.abs(r.consistency - 1) < 1e-3);
  assert.ok(Math.abs(r.vy + 0.8) < 1e-6);
  const noisy = flow.map(() => Math.random() * 2 - 1);
  assert.ok(regionFlow(noisy, mask).consistency < 0.3);
  assert.equal(regionFlow(flow, new Uint8Array(n)).pixels, 0);
});

test("scoring gates", () => {
  const cfg = { ...DEFAULTS };
  assert.equal(smoothScore(0.05, 0.9, 0.1, 0.3, 0.55, 0.85), 0);
  assert.equal(smoothScore(0.5, 0.4, 0.1, 0.3, 0.55, 0.85), 0);
  const low = smoothScore(0.2, 0.9, 0.1, 0.3, 0.55, 0.85);
  const high = smoothScore(2, 0.9, 0.1, 0.3, 0.55, 0.85);
  assert.ok(low > 0 && low < high && high < 1);
  assert.equal(directionPenalty(0, 1, cfg), 1);
  assert.equal(directionPenalty(1, 0.2, cfg), 0);
  const up = { vx: 0, vy: -1 };
  const down = { vx: 0, vy: 1 };
  assert.ok(Math.abs(handrailAgreement(up, up, cfg) - 1) < 1e-9);
  assert.equal(handrailAgreement(up, down, cfg), 0);
});

test("tracker keeps ids, coasts and expires", () => {
  assert.ok(Math.abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 50 / 150) < 1e-9);
  const t = new IoUTracker(0.3, 2);
  const first = t.update([{ box: [0, 0, 10, 20], conf: 0.9 }, { box: [50, 0, 60, 20], conf: 0.8 }]);
  const ids = first.map((x) => x.id).sort();
  const moved = t.update([{ box: [2, 0, 12, 20], conf: 0.9 }, { box: [51, 0, 61, 20], conf: 0.8 }]);
  assert.deepEqual(moved.map((x) => x.id).sort(), ids);
  assert.equal(t.update(null).length, 2);
  t.update([]);
  t.update([]);
  assert.equal(t.update([]).length, 0);
});

function feed(sm, n, moving, people, score) {
  let s;
  for (let i = 0; i < n; i++) s = sm.update(moving, people, score ?? (moving ? 0.7 : 0));
  return s;
}

test("state machine cycle and hysteresis", () => {
  const sm = new StateMachine({ ...DEFAULTS });
  assert.equal(feed(sm, 5, true, true), State.INITIALIZING);
  assert.equal(feed(sm, 40, true, true), State.WORKING);
  assert.equal(feed(sm, 45, false, true), State.STOPPED);
  assert.equal(feed(sm, 45, true, true), State.WORKING);
  assert.equal(feed(sm, 60, false, false), State.IDLE);

  const h = new StateMachine({ ...DEFAULTS });
  feed(h, 45, true, true);
  for (let i = 0; i < 300; i++) {
    h.update(i % 3 === 0, true, i % 3 === 0 ? 0.25 : 0.1);
    assert.equal(h.state, State.WORKING);
  }
});

test("riders leaving right before a stop is not a fault", () => {
  const sm = new StateMachine({ ...DEFAULTS });
  feed(sm, 45, true, true);
  feed(sm, 10, true, false);
  const history = Array.from({ length: 45 }, () => sm.update(false, false, 0));
  assert.ok(!history.includes(State.STOPPED));
  assert.equal(history.at(-1), State.IDLE);
});

test("people who already left do not turn idle into a fault", () => {
  const sm = new StateMachine({ ...DEFAULTS });
  feed(sm, 45, true, true);
  feed(sm, 19, false, true);
  const history = Array.from({ length: 60 }, () => sm.update(false, false, 0));
  assert.ok(!history.includes(State.STOPPED));
  assert.equal(history.at(-1), State.IDLE);
});

test("evaluation metrics", () => {
  const W = State.WORKING;
  const S = State.STOPPED;
  const I = State.IDLE;
  const truth = [
    [0, 10, W],
    [10, 20, S],
    [20, 30, I],
  ];
  const tl = Array.from({ length: 300 }, (_, i) => {
    const t = i / 10;
    return [t, t < 11.5 ? W : t < 21 ? S : I];
  });
  const r = evaluate(tl, truth);
  assert.ok(Math.abs(r.meanLatency[S] - 1.5) < 1e-9);
  assert.ok(Math.abs(r.accuracy - (1 - 25 / 300)) < 1e-9);
  assert.equal(r.falseAlarms, 0);
  assert.equal(formatTime(7.32), "0:00:07.320");
});

test("session stats: durations, faults, availability", () => {
  const s = new SessionStats();
  const states = [State.WORKING, State.WORKING, State.STOPPED, State.STOPPED, State.WORKING];
  let prev = null;
  states.forEach((st, i) => {
    s.update({ index: i + 1, time: i, state: st, previous: prev && prev !== st ? prev : null });
    prev = st;
  });
  s.finish(1);
  const sum = s.summary();
  assert.equal(sum.fault_count, 1);
  assert.equal(sum.faults[0].duration_s, 2);
  assert.equal(sum.availability_pct, 60);
});

// --- motion analysis with real optical flow (OpenCV.js) ------------------------------

const require = createRequire(import.meta.url);
const cv = await readyOpenCV(require("@techstark/opencv-js"));
const flowFn = makeFarnebackFlow(cv);

/** Frame with a static textured background and a textured belt inside the quad. */
function makeScene(width, height, quad, seed) {
  let s = seed;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const bg = new Uint8Array(width * height).map(() => 60 + rand() * 120);
  const beltH = height * 4;
  const belt = new Uint8Array(width * beltH);
  for (let y = 0; y < beltH; y++) for (let x = 0; x < width; x++) belt[y * width + x] = (y % 16 < 3 ? 40 : 150) + rand() * 60;
  const inside = rasterize(quad, width, height);
  return (offset) => {
    const f = bg.slice();
    for (let i = 0; i < f.length; i++) {
      if (!inside[i]) continue;
      const y = Math.floor(i / width);
      const x = i % width;
      f[i] = belt[(((y + offset) % beltH) + beltH) % beltH * width + x];
    }
    return f;
  };
}

function cropFor(analyzer, frame, width) {
  // Nearest-neighbour crop + resize, standing in for the canvas drawImage in the app.
  const { x0, y0, x1, y1 } = analyzer.window;
  const [w, h] = analyzer.size;
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const sx = Math.min(x1 - 1, Math.floor(x0 + ((x + 0.5) * (x1 - x0)) / w));
      const sy = Math.min(y1 - 1, Math.floor(y0 + ((y + 0.5) * (y1 - y0)) / h));
      out[y * w + x] = frame[sy * width + sx];
    }
  }
  return out;
}

test("moving belt is detected, still belt is not", () => {
  const W = 320;
  const H = 240;
  const quad = makeQuad([[130, 20], [190, 20], [230, 230], [90, 230]]);
  const scene = makeScene(W, H, quad, 7);
  for (const [speed, expectMoving] of [
    [2, true],
    [-2, true],
    [0, false],
  ]) {
    const analyzer = new MotionAnalyzer({ ...DEFAULTS }, quad, W, H, flowFn);
    const flags = [];
    for (let t = 0; t < 12; t++) {
      const r = analyzer.update(cropFor(analyzer, scene(t * speed), W));
      if (r.valid) flags.push(r.isMoving);
    }
    const frac = flags.filter(Boolean).length / flags.length;
    assert.ok(expectMoving ? frac > 0.9 : frac < 0.1, `speed ${speed}: moving fraction ${frac}`);
  }
});

test("full monitor: running, then stopped with people = fault", () => {
  const W = 320;
  const H = 240;
  const quad = makeQuad([[130, 20], [190, 20], [230, 230], [90, 230]]);
  const scene = makeScene(W, H, quad, 3);
  const mon = new EscalatorMonitor({ ...DEFAULTS }, quad, W, H, flowFn);
  const person = { box: [150, 100, 170, 160], conf: 0.9 };
  const states = [];
  let offset = 0;
  for (let i = 0; i < 120; i++) {
    const moving = i < 60;
    if (moving) offset += 2;
    const gray = cropFor(mon.motion, scene(offset), W);
    const detections = mon.wantsDetection ? [person] : null;
    const r = mon.step({ index: i + 1, time: i / 25, gray, detections });
    if (r.previous) states.push(r.state);
  }
  assert.deepEqual(states, [State.WORKING, State.STOPPED]);
});

// --- YOLO output decoding ------------------------------------------------------------------

test("YOLO person boxes are decoded, filtered and mapped back through the letterbox", async () => {
  const { decodePeople } = await import("../js/detector.js");
  const anchors = 6;
  const data = new Float32Array(84 * anchors);
  const set = (i, cx, cy, w, h, person, other = 0) => {
    data[i] = cx;
    data[anchors + i] = cy;
    data[2 * anchors + i] = w;
    data[3 * anchors + i] = h;
    data[4 * anchors + i] = person;
    data[5 * anchors + i] = other; // class 1 must be ignored
  };
  set(0, 320, 320, 100, 200, 0.9); // a person
  set(1, 322, 318, 100, 200, 0.8); // duplicate of it -> removed by NMS
  set(2, 100, 100, 40, 80, 0.2); // below the threshold
  set(3, 500, 400, 60, 120, 0.1, 0.99); // confident, but not a person
  set(4, 500, 400, 60, 120, 0.5); // second person
  // Region (x0=50, y0=20) resized by 0.5 and padded 10 px on the left.
  const people = decodePeople(data, anchors, 0.35, { scale: 0.5, left: 10, top: 0, x0: 50, y0: 20 });
  assert.equal(people.length, 2);
  assert.deepEqual(people[0].box, [(270 - 10) / 0.5 + 50, 220 / 0.5 + 20, (370 - 10) / 0.5 + 50, 420 / 0.5 + 20]);
  assert.equal(people[1].conf, 0.5);
});
