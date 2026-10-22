// Strip charts (uPlot) fed from the telemetry ring buffer.
/* global uPlot */

export const C = {
  kapton: "#d4a84b", cell: "#5b8def", cyan: "#4fc3d9", nominal: "#3dbb8c",
  caution: "#f0923c", alarm: "#ec4f55", violet: "#a78bfa", ink2: "#9aa9c2", ink3: "#6b7b97",
};

export class Telemetry {
  constructor(names, epochUnix, max = 1800) {
    this.names = names;
    this.idx = Object.fromEntries(names.map((n, i) => [n, i]));
    this.cols = names.map(() => []);
    this.x = [];
    this.epoch = epochUnix;
    this.max = max;
  }
  push(rows) {
    for (const r of rows) {
      this.x.push(this.epoch + r[0]);
      r.forEach((v, i) => this.cols[i].push(v));
    }
    const extra = this.x.length - this.max;
    if (extra > 0) { this.x.splice(0, extra); this.cols.forEach((c) => c.splice(0, extra)); }
  }
  clear() { this.x = []; this.cols = this.names.map(() => []); }
  col(name) { return this.cols[this.idx[name]]; }
}

const axisStyle = {
  stroke: C.ink3,
  grid: { stroke: "rgba(42,59,92,0.55)", width: 1 },
  ticks: { stroke: "rgba(42,59,92,0.9)", width: 1, size: 4 },
  font: '12px "Barlow Semi Condensed", Barlow, sans-serif',
};

export class Chart {
  // spec: { series: [{ key | fn, label, color, dash?, width? }], y: { label, log, min, max } }
  constructor(el, spec, tm) {
    this.el = el; this.spec = spec; this.tm = tm; this.plot = null;
  }
  ensure() {
    if (this.plot || !this.el.clientWidth) return !!this.plot;
    const s = this.spec;
    const opts = {
      width: this.el.clientWidth,
      height: this.el.clientHeight - 34,
      cursor: { drag: { x: false, y: false }, points: { size: 6 } },
      legend: { live: true },
      scales: { x: { time: true }, y: s.y.log ? { distr: 3, log: 10 } : { range: s.y.range || ((u, mn, mx) => pad(mn, mx, s.y)) } },
      axes: [
        { ...axisStyle, values: (u, splits) => splits.map((ts) => utcHM(ts)) },
        { ...axisStyle, label: s.y.label, labelSize: 18, labelFont: '12px Barlow, sans-serif', size: 52 },
      ],
      series: [{ value: (u, ts) => (ts == null ? "—" : utcHM(ts, true)) }, ...s.series.map((d) => ({
        label: d.label, stroke: d.color, width: d.width || 1.6, dash: d.dash,
        points: { show: false }, value: (u, v) => (v == null ? "—" : fmt(v, s.y.digits)),
      }))],
    };
    this.plot = new uPlot(opts, this.data(), this.el);
    new ResizeObserver(() => {
      if (this.plot && this.el.clientWidth) this.plot.setSize({ width: this.el.clientWidth, height: this.el.clientHeight - 34 });
    }).observe(this.el);
    return true;
  }
  data() {
    const tm = this.tm;
    return [tm.x, ...this.spec.series.map((d) => (d.fn ? tm.x.map((_, i) => d.fn(tm, i)) : tm.col(d.key)).map(
      (v) => (this.spec.y.log ? (v > 0 ? v : null) : v)))];
  }
  update() { if (this.ensure()) this.plot.setData(this.data()); }
}

function pad(mn, mx, y) {
  if (y.min != null) mn = Math.min(mn, y.min);
  if (y.max != null) mx = Math.max(mx, y.max);
  if (mn === mx || mn == null) return [(mn || 0) - 1, (mx || 0) + 1];
  const p = (mx - mn) * 0.08;
  return [y.min != null ? Math.min(y.min, mn) : mn - p, mx + p];
}

export function fmt(v, digits) {
  if (v == null || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  const d = digits != null ? digits : a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 3;
  return v.toFixed(d);
}

const p2 = (n) => String(n).padStart(2, "0");
export function utcHM(ts, sec = false) {
  const d = new Date(ts * 1000);
  return p2(d.getUTCHours()) + ":" + p2(d.getUTCMinutes()) + (sec ? ":" + p2(d.getUTCSeconds()) : "");
}
