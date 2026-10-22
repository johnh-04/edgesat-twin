// Canvas/SVG widgets: ground-track map, sky plot, matrices, mode diagram.
import { fmt } from "./charts.js";

const D2R = Math.PI / 180;

export class GroundMap {
  constructor(canvas, info, land) {
    this.c = canvas; this.info = info; this.land = land;
    this.base = null;
    new ResizeObserver(() => { this.base = null; }).observe(canvas);
  }
  size() {
    const r = window.devicePixelRatio || 1;
    const w = Math.round(this.c.clientWidth * r), h = Math.round(this.c.clientHeight * r);
    if (this.c.width !== w || this.c.height !== h) { this.c.width = w; this.c.height = h; this.base = null; }
    return [w, h];
  }
  X(lon) { return (lon + 180) / 360 * this.c.width; }
  Y(lat) { return (90 - lat) / 180 * this.c.height; }
  drawBase() {
    const [w, h] = [this.c.width, this.c.height];
    const off = document.createElement("canvas"); off.width = w; off.height = h;
    const g = off.getContext("2d");
    g.fillStyle = "#16294a"; g.fillRect(0, 0, w, h);
    g.strokeStyle = "rgba(160,190,230,0.08)"; g.lineWidth = 1;
    for (let lon = -150; lon < 180; lon += 30) { g.beginPath(); g.moveTo(this.X(lon), 0); g.lineTo(this.X(lon), h); g.stroke(); }
    for (let lat = -60; lat <= 60; lat += 30) { g.beginPath(); g.moveTo(0, this.Y(lat)); g.lineTo(w, this.Y(lat)); g.stroke(); }
    g.fillStyle = "#344f43"; g.strokeStyle = "#6d8b7a"; g.lineWidth = 1;
    for (const ring of this.land) {
      g.beginPath();
      ring.forEach(([lo, la], i) => (i ? g.lineTo(this.X(lo), this.Y(la)) : g.moveTo(this.X(lo), this.Y(la))));
      g.closePath(); g.fill(); g.stroke();
    }
    g.setLineDash([5, 3]); g.strokeStyle = "#d4a84b"; g.lineWidth = 1.4;
    for (const a of this.info.aois) g.strokeRect(this.X(a.lon[0]), this.Y(a.lat[1]), this.X(a.lon[1]) - this.X(a.lon[0]), this.Y(a.lat[0]) - this.Y(a.lat[1]));
    g.setLineDash([]);
    this.base = off;
  }
  draw(s) {
    this.size();
    if (!this.base) this.drawBase();
    const g = this.c.getContext("2d");
    const [w, h] = [this.c.width, this.c.height];
    const k = w / 1000;
    g.drawImage(this.base, 0, 0);
    // night side: sun elevation < 0
    const sun = s.orbit.sun_ecef;
    const dec = Math.asin(sun[2]) / D2R, slon = Math.atan2(sun[1], sun[0]) / D2R;
    g.fillStyle = "rgba(4,8,18,0.45)";
    g.beginPath();
    const south = dec > 0;
    g.moveTo(0, south ? h : 0);
    for (let px = 0; px <= w; px += 4) {
      const lon = px / w * 360 - 180;
      const lat = Math.atan(-Math.cos((lon - slon) * D2R) / Math.tan(dec * D2R)) / D2R;
      g.lineTo(px, this.Y(lat));
    }
    g.lineTo(w, south ? h : 0); g.closePath(); g.fill();
    // ground station visibility
    for (const gs of this.info.stations) {
      const lam = 20.5; // ~5 deg elevation mask at 525 km, deg of arc
      g.strokeStyle = "rgba(79,195,217,0.8)"; g.lineWidth = 1.2 * k; g.beginPath();
      for (let a = 0; a <= 360; a += 6) {
        const [la, lo] = destination(gs.lat, gs.lon, a, lam);
        const x = this.X(lo), y = this.Y(la);
        a ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.stroke();
      g.fillStyle = "#4fc3d9"; g.fillRect(this.X(gs.lon) - 3 * k, this.Y(gs.lat) - 3 * k, 6 * k, 6 * k);
    }
    // fires
    for (const [lat, lon, st] of s.fires) {
      g.fillStyle = st === 2 ? "#ff4d5a" : "#ff8a3d";
      g.globalAlpha = st ? 1 : 0.45;
      g.beginPath(); g.arc(this.X(lon), this.Y(lat), (st === 2 ? 3.4 : 1.3) * k, 0, 7); g.fill();
    }
    g.globalAlpha = 1;
    // tracks
    const path = (pts, style, dash) => {
      g.strokeStyle = style; g.lineWidth = 1.6 * k; g.setLineDash(dash || []);
      g.beginPath();
      let prev = null;
      for (const [la, lo] of pts) {
        const x = this.X(lo), y = this.Y(la);
        if (prev === null || Math.abs(lo - prev) > 180) g.moveTo(x, y); else g.lineTo(x, y);
        prev = lo;
      }
      g.stroke(); g.setLineDash([]);
    };
    path(s.orbit.track_hist, "rgba(212,168,75,0.85)");
    if (s.orbit.prediction) {
      const p = s.orbit.prediction;
      path(p.lat.slice(0, 120).map((la, i) => [la, p.lon[i]]), "rgba(154,169,194,0.7)", [4 * k, 4 * k]);
    }
    const x = this.X(s.orbit.lon), y = this.Y(s.orbit.lat);
    g.fillStyle = s.payload.imaging ? "#ffd27a" : "#d4a84b";
    g.beginPath(); g.arc(x, y, 5 * k, 0, 7); g.fill();
    g.strokeStyle = "rgba(212,168,75,0.5)"; g.lineWidth = 1.2 * k;
    g.beginPath(); g.arc(x, y, 10 * k, 0, 7); g.stroke();
  }
}

function destination(lat, lon, bearing, dist) {
  const la1 = lat * D2R, lo1 = lon * D2R, b = bearing * D2R, d = dist * D2R;
  const la2 = Math.asin(Math.sin(la1) * Math.cos(d) + Math.cos(la1) * Math.sin(d) * Math.cos(b));
  const lo2 = lo1 + Math.atan2(Math.sin(b) * Math.sin(d) * Math.cos(la1), Math.cos(d) - Math.sin(la1) * Math.sin(la2));
  return [la2 / D2R, ((lo2 / D2R + 540) % 360) - 180];
}

export class SkyPlot {
  constructor(canvas) { this.c = canvas; this.trace = []; this.wasVisible = false; }
  draw(cm) {
    const r = window.devicePixelRatio || 1;
    const css = Math.min(this.c.parentElement.clientWidth - 32, 340);
    this.c.style.width = css + "px"; this.c.style.height = css + "px";
    this.c.width = css * r; this.c.height = css * r;
    const g = this.c.getContext("2d");
    const W = this.c.width, cx = W / 2, R = W / 2 - 22 * r;
    g.clearRect(0, 0, W, W);
    g.strokeStyle = "#2a3b5c"; g.lineWidth = r; g.fillStyle = "#6b7b97"; g.font = `${12 * r}px Barlow, sans-serif`;
    for (const el of [0, 30, 60]) { g.beginPath(); g.arc(cx, cx, R * (90 - el) / 90, 0, 7); g.stroke(); g.fillText(el + "°", cx + 3 * r, cx - R * (90 - el) / 90 + 12 * r); }
    for (const [a, l] of [[0, "N"], [90, "E"], [180, "S"], [270, "W"]]) {
      const x = cx + Math.sin(a * D2R) * (R + 12 * r), y = cx - Math.cos(a * D2R) * (R + 12 * r);
      g.fillText(l, x - 4 * r, y + 4 * r);
    }
    if (cm.visible) {
      if (!this.wasVisible) this.trace = [];
      this.trace.push([cm.azimuth_deg, cm.elevation_deg]);
    }
    this.wasVisible = cm.visible;
    const pt = (az, el) => [cx + Math.sin(az * D2R) * R * (90 - el) / 90, cx - Math.cos(az * D2R) * R * (90 - el) / 90];
    g.strokeStyle = "#4fc3d9"; g.lineWidth = 2 * r; g.beginPath();
    this.trace.forEach(([az, el], i) => { const [x, y] = pt(az, el); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.stroke();
    if (cm.visible) {
      const [x, y] = pt(cm.azimuth_deg, cm.elevation_deg);
      g.fillStyle = cm.transmitting ? "#d4a84b" : "#9aa9c2";
      g.beginPath(); g.arc(x, y, 6 * r, 0, 7); g.fill();
    } else {
      g.fillStyle = "#6b7b97"; g.textAlign = "center"; g.fillText("below the horizon", cx, cx + 4 * r); g.textAlign = "start";
    }
  }
}

// heat-mapped matrix as an HTML table
export function matrixHTML(M, opts = {}) {
  const rows = opts.rows || M.map((_, i) => ""), cols = opts.cols || M[0].map(() => "");
  const scale = opts.scale || Math.max(1e-12, ...M.flat().map(Math.abs));
  const cell = (v, i, j) => {
    const t = Math.min(1, Math.abs(v) / scale);
    const bg = opts.diverging && v < 0 ? `rgba(236,79,85,${0.12 + 0.6 * t})` : `rgba(91,141,239,${0.08 + 0.62 * t})`;
    const txt = opts.cellText ? opts.cellText(v, i, j) : fmt(v, opts.digits);
    return `<td style="background:${bg}">${txt}</td>`;
  };
  let h = "<table>";
  if (opts.cols) h += "<tr><th></th>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>";
  M.forEach((r, i) => { h += `<tr>${opts.rows ? `<th>${rows[i]}</th>` : ""}${r.map((v, j) => cell(v, i, j)).join("")}</tr>`; });
  return h + "</table>";
}

export function modesSVG(mode, activity) {
  const node = (id, x, y, w, label, on) =>
    `<g class="node ${on ? "on" : ""}"><rect x="${x}" y="${y}" width="${w}" height="34" rx="6"/><text x="${x + w / 2}" y="${y + 22}" text-anchor="middle">${label}</text></g>`;
  const act = (a) => mode === "NOMINAL" && activity === a;
  return `<svg class="svg-modes" viewBox="0 0 760 210" width="100%" role="img" aria-label="Mode diagram, current ${mode} ${activity}">
  <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#6b7b97" stroke="none"/></marker></defs>
  <rect x="250" y="8" width="500" height="194" rx="10" fill="none" stroke="#2a3b5c" stroke-dasharray="4 4"/>
  <text x="262" y="28" style="fill:#6b7b97">Nominal operations, planned on board</text>
  ${node("leop", 10, 30, 130, "Launch & early orbit", mode === "LEOP")}
  ${node("safe", 10, 100, 130, "Safe", mode === "SAFE")}
  ${node("reboot", 10, 166, 130, "OBC reboot", mode === "REBOOT")}
  ${node("img", 280, 48, 200, "Imaging (nadir)", act("IMAGING"))}
  ${node("dl", 510, 48, 220, "Downlink (track Bari)", act("DOWNLINK"))}
  ${node("chg", 280, 128, 200, "Charge (Sun pointing)", act("CHARGE"))}
  ${node("om", 510, 128, 220, "Orbit maintenance burn", act("ORBIT_MAINT"))}
  <path d="M140,47 C200,47 200,64 276,64"/>
  <path d="M140,117 C200,117 200,145 276,145"/>
  <path d="M276,152 C200,160 190,124 144,121"/>
  <path d="M75,166 L75,138"/>
  <path d="M75,64 L75,96"/>
</svg>`;
}
