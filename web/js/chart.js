// Timeline: three small panels on one shared time axis (state, move confidence,
// people in the ROI) with a crosshair tooltip. SVG, themed through CSS variables.
import { State } from "./core.js";
import { STATE_COLORS } from "./render.js";

const LABELS = {
  [State.WORKING]: "Working",
  [State.STOPPED]: "Stopped / fault",
  [State.IDLE]: "Idle",
  [State.INITIALIZING]: "Starting",
};
const NS = "http://www.w3.org/2000/svg";

function niceStep(span, target) {
  const raw = span / target;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const f = raw / pow;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * pow;
}

function el(name, attrs = {}, text) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

export class TimelineChart {
  constructor(container, { onSeek } = {}) {
    this.container = container;
    this.onSeek = onSeek;
    this.rows = [];
    this.threshold = 0.35;
    this.cursor = null;
    new ResizeObserver(() => this.render()).observe(container);
  }

  setData(rows, threshold) {
    this.rows = rows; // [{t, state, conf, people}]
    this.threshold = threshold;
    this.render();
  }

  setCursor(t) {
    this.cursor = t;
    if (this.cursorLine && this.x) {
      const x = this.x(t);
      this.cursorLine.setAttribute("x1", x);
      this.cursorLine.setAttribute("x2", x);
      this.cursorLine.style.display = t === null ? "none" : "";
    }
  }

  render() {
    const rows = this.rows;
    this.container.replaceChildren();
    if (!rows.length) return;
    const width = Math.max(320, this.container.clientWidth);
    const m = { left: 44, right: 14, top: 8 };
    const bandH = 26;
    const confH = 120;
    const peopleH = 56;
    const gap = 26;
    const axisH = 30;
    const yBand = m.top;
    const yConf = yBand + bandH + gap;
    const yPeople = yConf + confH + gap;
    const height = yPeople + peopleH + axisH;
    const t0 = rows[0].t;
    const t1 = rows.at(-1).t + (rows.length > 1 ? rows[1].t - rows[0].t : 0.04);
    const plotW = width - m.left - m.right;
    const x = (t) => m.left + ((t - t0) / (t1 - t0 || 1)) * plotW;
    this.x = x;

    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", class: "timeline-svg" });
    svg.append(el("title", {}, "Escalator state, move confidence and people in the ROI over time"));

    // Panel titles (they name the single series in each panel).
    const title = (y, text) => svg.append(el("text", { x: m.left, y: y - 8, class: "panel-title" }, text));
    title(yConf, "Move confidence");
    title(yPeople, "People in ROI");

    // State band.
    let start = 0;
    for (let i = 1; i <= rows.length; i++) {
      if (i < rows.length && rows[i].state === rows[start].state) continue;
      const xa = x(rows[start].t);
      const xb = i < rows.length ? x(rows[i].t) : x(t1);
      const state = rows[start].state;
      svg.append(el("rect", { x: xa, y: yBand, width: Math.max(0, xb - xa - 2), height: bandH, rx: 4, fill: STATE_COLORS[state] }));
      const label = LABELS[state];
      if (xb - xa > label.length * 7 + 16) {
        svg.append(
          el(
            "text",
            { x: xa + 8, y: yBand + bandH / 2 + 4, class: state === State.IDLE ? "band-label dark" : "band-label" },
            label,
          ),
        );
      }
      start = i;
    }
    svg.append(el("text", { x: m.left - 8, y: yBand + bandH / 2 + 4, class: "axis-label", "text-anchor": "end" }, "State"));

    // Shared helpers for the two value panels.
    const grid = (yTop, h, ticks, fmt) => {
      for (const v of ticks.values) {
        const y = yTop + h - ((v - ticks.min) / (ticks.max - ticks.min)) * h;
        svg.append(el("line", { x1: m.left, x2: width - m.right, y1: y, y2: y, class: v === ticks.min ? "baseline" : "gridline" }));
        svg.append(el("text", { x: m.left - 8, y: y + 4, class: "axis-label", "text-anchor": "end" }, fmt(v)));
      }
    };
    const line = (yTop, h, max, key, step) => {
      let d = "";
      rows.forEach((r, i) => {
        const px = x(r.t);
        const py = yTop + h - (r[key] / max) * h;
        if (step && i) d += `H${px.toFixed(1)}`;
        d += `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`;
      });
      svg.append(el("path", { d, class: "series-line" }));
    };

    grid(yConf, confH, { min: 0, max: 1, values: [0, 0.5, 1] }, (v) => v.toFixed(1));
    const yThr = yConf + confH - this.threshold * confH;
    svg.append(el("line", { x1: m.left, x2: width - m.right, y1: yThr, y2: yThr, class: "threshold" }));
    svg.append(el("text", { x: width - m.right, y: yThr - 5, class: "axis-label", "text-anchor": "end" }, `moving threshold ${this.threshold.toFixed(2)}`));
    line(yConf, confH, 1, "conf", false);

    const maxPeople = Math.max(2, ...rows.map((r) => r.people));
    grid(yPeople, peopleH, { min: 0, max: maxPeople, values: [0, maxPeople] }, (v) => String(v));
    line(yPeople, peopleH, maxPeople, "people", true);

    // Time axis.
    const step = niceStep(t1 - t0, Math.max(3, Math.floor(plotW / 90)));
    for (let t = Math.ceil(t0 / step) * step; t <= t1 + 1e-9; t += step) {
      svg.append(el("text", { x: x(t), y: height - 10, class: "axis-label", "text-anchor": "middle" }, `${+t.toFixed(2)} s`));
    }

    // Crosshair, cursor (replay position) and tooltip.
    this.cursorLine = el("line", { y1: yBand, y2: yPeople + peopleH, class: "cursor" });
    const hoverLine = el("line", { y1: yBand, y2: yPeople + peopleH, class: "crosshair", style: "display:none" });
    const hit = el("rect", { x: m.left, y: yBand, width: plotW, height: yPeople + peopleH - yBand, fill: "transparent" });
    svg.append(this.cursorLine, hoverLine, hit);
    this.setCursor(this.cursor);

    const tip = document.createElement("div");
    tip.className = "chart-tooltip";
    tip.hidden = true;
    this.container.append(svg, tip);

    const nearest = (evt) => {
      const rect = svg.getBoundingClientRect();
      const t = t0 + ((evt.clientX - rect.left - m.left) / plotW) * (t1 - t0);
      let lo = 0;
      let hi = rows.length - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (rows[mid].t <= t) lo = mid;
        else hi = mid - 1;
      }
      return rows[lo];
    };
    hit.addEventListener("pointermove", (evt) => {
      const r = nearest(evt);
      const px = x(r.t);
      hoverLine.setAttribute("x1", px);
      hoverLine.setAttribute("x2", px);
      hoverLine.style.display = "";
      tip.hidden = false;
      tip.innerHTML = `<strong>${r.t.toFixed(2)} s</strong>
        <span><i style="background:${STATE_COLORS[r.state]}"></i>${LABELS[r.state]}</span>
        <span>Move confidence ${r.conf.toFixed(2)}</span>
        <span>People in ROI ${r.people}</span>`;
      const left = Math.min(px + 12, width - tip.offsetWidth - 4);
      tip.style.left = `${Math.max(4, left)}px`;
      tip.style.top = `${yConf}px`;
    });
    hit.addEventListener("pointerleave", () => {
      hoverLine.style.display = "none";
      tip.hidden = true;
    });
    hit.addEventListener("click", (evt) => this.onSeek?.(nearest(evt).t));
  }
}

export { LABELS as STATE_LABELS };
