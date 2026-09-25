// Annotated-frame overlay, drawn with the 2D canvas API (mirrors escalator_monitor/render.py).
import { State, quadRegions } from "./core.js";

// Status colours (fixed across themes): good / critical / warning / muted.
export const STATE_COLORS = {
  [State.WORKING]: "#0ca30c",
  [State.STOPPED]: "#d03b3b",
  [State.IDLE]: "#fab219",
  [State.INITIALIZING]: "#898781",
};
const STATE_TEXT = {
  [State.WORKING]: "#ffffff",
  [State.STOPPED]: "#ffffff",
  [State.IDLE]: "#1a1a19",
  [State.INITIALIZING]: "#ffffff",
};
const MOVING_TINT = "rgba(12, 163, 12, 0.18)";
const STILL_TINT = "rgba(236, 131, 90, 0.18)";
const PERSON = "#2ee66b";
const FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';

function path(ctx, poly) {
  ctx.beginPath();
  poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
}

function pill(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

export class Renderer {
  constructor(cfg, quad, width, height) {
    this.cfg = cfg;
    this.width = width;
    this.height = height;
    this.ui = Math.min(2.5, Math.max(0.7, Math.min(width, height) / 540));
    this.setRoi(quad);
  }

  setRoi(quad) {
    this.quad = quad;
    this.regions = quadRegions(quad, this.cfg.handrailWidthFrac);
  }

  /** Draw `frame` (a canvas) plus the overlay for result `r` into ctx. */
  draw(ctx, frame, r, duration = 0) {
    const { width: w, height: h, ui } = this;
    ctx.drawImage(frame, 0, 0, w, h);
    const color = STATE_COLORS[r.state];

    if (r.motion.valid) {
      ctx.fillStyle = r.motion.handrailScore > 0.3 ? MOVING_TINT : STILL_TINT;
      path(ctx, this.regions.left);
      ctx.fill();
      path(ctx, this.regions.right);
      ctx.fill();
      ctx.fillStyle = r.motion.stepsScore > 0.3 ? MOVING_TINT : STILL_TINT;
      path(ctx, this.regions.steps);
      ctx.fill();
    }

    ctx.lineWidth = Math.max(1.5, 2 * ui);
    ctx.strokeStyle = PERSON;
    ctx.font = `600 ${Math.round(11 * ui)}px ${FONT}`;
    ctx.fillStyle = PERSON;
    for (const p of r.people) {
      const [x1, y1, x2, y2] = p.box;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      if (p.conf < 1) ctx.fillText(p.conf.toFixed(2), x1, Math.max(10, y1 - 4));
    }

    ctx.lineWidth = Math.max(2, 2.5 * ui);
    ctx.strokeStyle = color;
    path(ctx, this.quad);
    ctx.stroke();

    this.badge(ctx, r);
    if (this.cfg.showDebug) this.panel(ctx, r);
    if (duration > 0) {
      ctx.fillStyle = color;
      ctx.fillRect(0, h - Math.max(3, 4 * ui), (w * Math.min(1, r.time / duration)) | 0, Math.max(3, 4 * ui));
    }
  }

  badge(ctx, r) {
    const { width: w, ui } = this;
    const labels = [[r.state, STATE_COLORS[r.state], STATE_TEXT[r.state]]];
    if (r.wrongDirection) labels.push(["WRONG DIRECTION", STATE_COLORS[State.STOPPED], "#ffffff"]);
    ctx.font = `700 ${Math.round(17 * ui)}px ${FONT}`;
    ctx.textBaseline = "middle";
    let y = 12 * ui;
    for (const [text, bg, fg] of labels) {
      const tw = ctx.measureText(text).width;
      const bw = tw + 28 * ui;
      const bh = 34 * ui;
      const x = (w - bw) / 2;
      ctx.globalAlpha = 0.94;
      ctx.fillStyle = bg;
      pill(ctx, x, y, bw, bh, 8 * ui);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.fillStyle = fg;
      ctx.fillText(text, x + 14 * ui, y + bh / 2 + 1);
      y += bh + 6 * ui;
    }
    ctx.textBaseline = "alphabetic";
  }

  panel(ctx, r) {
    const { height: h, ui } = this;
    const m = r.motion;
    const minutes = Math.floor(r.time / 60);
    const seconds = (r.time % 60).toFixed(2).padStart(5, "0");
    const lines = [
      [`Time ${String(minutes).padStart(2, "0")}:${seconds}   People in ROI ${r.people.length}`, "#ffffff"],
      [`Handrails  mag ${m.handrailMag.toFixed(2)}  cons ${m.handrailCons.toFixed(2)}  score ${m.handrailScore.toFixed(2)}`, m.handrailScore > 0.3 ? "#a8f0a8" : "#c9c9c9"],
      [`Steps      mag ${m.stepsMag.toFixed(2)}  cons ${m.stepsCons.toFixed(2)}  score ${m.stepsScore.toFixed(2)}`, m.stepsScore > 0.3 ? "#a8f0a8" : "#c9c9c9"],
      [`Move confidence ${m.confidence.toFixed(2)} (need ${this.cfg.moveConfidenceMin.toFixed(2)})  moving ${Math.round(100 * r.window.movingRatio)}%`, "#ffffff"],
    ];
    if (r.direction) lines.push([`Surface moving ${r.direction} (image)`, "#c9c9c9"]);
    ctx.font = `500 ${Math.round(11.5 * ui)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    const lineH = 17 * ui;
    const width = Math.max(...lines.map(([t]) => ctx.measureText(t).width)) + 20 * ui;
    const height = lines.length * lineH + 12 * ui;
    const x = 10 * ui;
    const y = h - height - 14 * ui;
    ctx.fillStyle = "rgba(15, 15, 15, 0.78)";
    pill(ctx, x, y, width, height, 6 * ui);
    ctx.fill();
    lines.forEach(([text, c], i) => {
      ctx.fillStyle = c;
      ctx.fillText(text, x + 10 * ui, y + 6 * ui + lineH * (i + 0.8));
    });
  }
}
