// EDGESAT-1 mission control: telemetry stream, views and commands.
import { Globe, AttitudeView } from "./globe.js";
import { Telemetry, Chart, C, fmt, utcHM } from "./charts.js";
import { GroundMap, SkyPlot, matrixHTML, modesSVG } from "./widgets.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const CLASSES = ["CLOUD", "WATER", "LAND", "FIRE"];
const CLASS_LABEL = { CLOUD: "Cloud", WATER: "Water", LAND: "Clear land", FIRE: "Wildfire" };

let info, land, tm, globe, attSmall, attBig, gmap, sky, epochUnix;
let S = null, fresh = false, view = "overview", lastT = 0, lastScenario = null;
const charts = {};

// ---------------------------------------------------------------------------------- helpers
const utc = (t) => {
  const d = new Date((epochUnix + t) * 1000);
  return d.toISOString().slice(0, 19).replace("T", " ");
};
const met = (t) => {
  const d = Math.floor(t / 86400), h = Math.floor((t % 86400) / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  return `${d}d ${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
};
const dur = (s) => (s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`);
const kv = (pairs) => `<dl class="kv">${pairs.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
const chip = (txt, cls) => `<span class="status-chip ${cls}">${txt}</span>`;
const gb = (x) => (x >= 1 ? `${x.toFixed(2)} GB` : `${(x * 1000).toFixed(0)} MB`);
const pct = (x, d = 0) => `${x.toFixed(d)}%`;
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 2600); }
async function cmd(c, arg) {
  const r = await fetch("/api/command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cmd: c, arg }) });
  const j = await r.json();
  if (!j.ok) toast(j.msg); else if (c === "inject") toast(`Fault injected: ${j.msg}`);
  return j;
}

// ---------------------------------------------------------------------------------- status
function statuses(s) {
  const a = s.adcs, e = s.eps, p = s.payload;
  const st = {};
  const wheelsBad = a.wheels_isolated?.some(Boolean) || a.wheels_seized?.some(Boolean);
  st.adcs = wheelsBad || a.mode === "DETUMBLE" ? "RED" : (["FAILED", "BLINDED"].includes(a.sensors?.star_tracker) || a.pointing_error_deg > 2) ? "YELLOW" : "GREEN";
  st.eps = e.soc < 0.2 ? "RED" : (e.soc < 0.3 || e.panel_health.some((h) => h < 1) || e.capacity_wh < info.battery_wh - 1) ? "YELLOW" : "GREEN";
  st.tcs = s.tcs.status.includes("RED") ? "RED" : s.tcs.status.includes("YELLOW") || s.tcs.heater_stuck ? "YELLOW" : "GREEN";
  st.comms = s.comms.pa_degradation_db > 0 ? "YELLOW" : "GREEN";
  st.obdh = s.obdh.mode === "REBOOT" ? "RED" : ["SAFE", "LEOP"].includes(s.obdh.mode) ? "YELLOW" : "GREEN";
  st.prop = s.prop.leaking && !s.prop.latch_closed ? "RED" : s.prop.latch_closed ? "YELLOW" : "GREEN";
  st.payload = p.memory.fill_pct > 97 ? "RED" : (p.ai_hung || !p.ai_enabled || p.memory.fill_pct > 80) ? "YELLOW" : "GREEN";
  const all = Object.values(st);
  st.overview = all.includes("RED") ? "RED" : all.includes("YELLOW") ? "YELLOW" : "GREEN";
  return st;
}

// ---------------------------------------------------------------------------------- charts
function defineCharts() {
  const def = {
    point: { y: { label: "deg", log: true }, series: [{ key: "point_err", label: "Pointing error", color: C.kapton }] },
    know: { y: { label: "arcsec", log: true }, series: [
      { key: "know_err", label: "Knowledge error", color: C.cell },
      { fn: (t, i) => Math.max(t.col("sig_x")[i], t.col("sig_y")[i], t.col("sig_z")[i]), label: "MEKF 3σ (max axis)", color: C.ink2, dash: [5, 4] }] },
    wheels: { y: { label: "mNms" }, series: [
      { key: "h_x", label: "X", color: C.alarm }, { key: "h_y", label: "Y", color: C.nominal }, { key: "h_z", label: "Z", color: C.cell }] },
    torques: { y: { label: "µNm", log: true }, series: [
      { key: "tq_gg", label: "Gravity gradient", color: C.cell }, { key: "tq_aero", label: "Aerodynamic", color: C.cyan },
      { key: "tq_srp", label: "Solar pressure", color: C.kapton }, { key: "tq_mag", label: "Residual dipole", color: C.violet }] },
    soc: { y: { label: "%", min: 0, max: 100, digits: 1 }, series: [{ key: "soc", label: "SoC", color: C.nominal }] },
    power: { y: { label: "W", min: 0 }, series: [
      { key: "solar_w", label: "Solar used", color: C.kapton }, { key: "load_w", label: "Load", color: C.cell }] },
    temps: { y: { label: "°C" }, series: [
      { key: "T_STRUCTURE", label: "Structure", color: C.ink2 }, { key: "T_SOLAR_ARRAY", label: "Solar array", color: C.kapton },
      { key: "T_BATTERY", label: "Battery", color: C.nominal }, { key: "T_AVIONICS", label: "Avionics", color: C.alarm },
      { key: "T_PAYLOAD", label: "Payload", color: C.cell }, { key: "T_COMMS", label: "Comms", color: C.violet }] },
    link: { y: { label: "deg / dB", min: 0 }, series: [
      { key: "el", label: "Elevation", color: C.cyan }, { key: "margin", label: "Link margin", color: C.kapton }] },
    rate: { y: { label: "Mbit/s", min: 0 }, series: [{ key: "rate_mbps", label: "Downlink", color: C.cyan }] },
    memory: { y: { label: "% full", min: 0, max: 100, digits: 1 }, series: [
      { key: "mem_ai", label: "Edge AI", color: C.kapton }, { key: "mem_base", label: "Store-everything", color: C.ink3 }] },
    alt: { y: { label: "km", digits: 2 }, series: [{ key: "alt_km", label: "Altitude (osculating)", color: C.cell }] },
    sma: { y: { label: "m", digits: 0, max: 20 }, series: [
      { key: "sma_dev", label: "Mean SMA − reference", color: C.kapton, width: 2 },
      { fn: () => -150, label: "Burn threshold", color: C.alarm, dash: [6, 5], width: 1.2 }] },
    useful: { y: { label: "GB", min: 0, digits: 2 }, series: [
      { key: "useful_ai", label: "Edge AI", color: C.kapton, width: 2.2 }, { key: "useful_base", label: "Store-everything", color: C.ink3 }] },
  };
  for (const el of $$("[data-chart]")) charts[el.dataset.chart] = new Chart(el, def[el.dataset.chart], tm);
}

// ---------------------------------------------------------------------------------- views
function renderHeader(s, st) {
  $("#utc").textContent = utc(s.t);
  $("#met").textContent = met(s.t);
  const mb = $("#mode-badge");
  mb.textContent = { NOMINAL: "Nominal", SAFE: "Safe mode", REBOOT: "OBC reboot", LEOP: "Launch & early orbit" }[s.obdh.mode] || s.obdh.mode;
  mb.dataset.mode = s.obdh.mode;
  $("#activity-badge").textContent = { IMAGING: "Imaging", DOWNLINK: "Downlink", CHARGE: "Charging", ORBIT_MAINT: "Orbit maintenance", SAFE: "Sun pointing", REBOOT: "Free drift", DETUMBLE: "Detumbling", SUN_ACQ: "Sun acquisition" }[s.obdh.activity] || s.obdh.activity;
  $("#achieved").textContent = s.paused ? "paused" : `${s.achieved_warp.toFixed(0)}× real time`;
  $$(".warp button").forEach((b) => b.classList.toggle("on", b.dataset.warp === "pause" ? s.paused : +b.dataset.warp === s.warp && !s.paused));
  $("#ai-toggle").checked = s.ai_enabled;
  for (const d of $$("[data-status]")) d.className = "dot " + (st[d.dataset.status] || "");
}

function renderEvents(s, el, n) {
  el.innerHTML = s.events.slice(0, n).map((e) =>
    `<div class="ev ${e.sev}"><span class="ts">${utcHM(epochUnix + e.t, true)}</span><span class="sub">${e.sub}</span><span class="msg">${e.msg}</span></div>`).join("");
}

function renderOverview(s, st) {
  const o = s.orbit, a = s.adcs, e = s.eps, c = s.comms, p = s.payload, ob = s.obdh;
  $("#ov-latlon").textContent = `${o.lat.toFixed(2)}°, ${o.lon.toFixed(2)}°`;
  $("#ov-alt").textContent = `${o.alt_km.toFixed(1)} km, ${o.speed_km_s.toFixed(2)} km/s`;
  $("#ov-sun").textContent = o.illum > 0.99 ? "in sunlight" : o.illum > 0.01 ? "penumbra" : "eclipse";
  const nc = ob.next_contact;
  $("#ov-next").textContent = c.visible ? `in contact with ${c.station}` : nc ? `${nc.station} in ${dur(nc.aos - s.t)}` : "none in the next 4 h";
  const tile = (key, name, abbr, main, sub) =>
    `<button class="tile ${st[key]}" data-goto="${key}"><div class="t-name">${name}<span>${abbr}</span></div><div class="t-main">${main}</div><div class="t-sub">${sub}</div></button>`;
  const stState = { TRACKING: "tracking", SUN_EXCL: "Sun in field of view", EARTH_EXCL: "Earth in field of view", RATE: "rate too high", BLINDED: "blinded", FAILED: "failed" }[a.sensors.star_tracker];
  $("#tiles").innerHTML = [
    tile("adcs", "Attitude", "ADCS", `${fmt(a.pointing_error_deg, 2)}° error`, `${a.mode.replace("_", " ").toLowerCase()}, star tracker ${stState}`),
    tile("eps", "Power", "EPS", `${e.soc_pct.toFixed(1)}% charge`, `solar ${e.solar_used_w.toFixed(1)} W, load ${e.load_w.toFixed(1)} W`),
    tile("tcs", "Thermal", "TCS", `battery ${s.tcs.temps_c[2].toFixed(1)} °C`, `array ${s.tcs.temps_c[1].toFixed(0)} °C, avionics ${s.tcs.temps_c[3].toFixed(0)} °C`),
    tile("comms", "Communications", "S-band", c.transmitting ? `${(c.rate_bps / 1e6).toFixed(1)} Mbit/s` : c.visible ? "in view" : "no contact", `Bari elevation ${c.elevation_deg.toFixed(0)}°`),
    tile("obdh", "Data handling", "OBDH", ob.activity.replace("_", " ").toLowerCase(), `memory ${pct(p.memory.fill_pct, 1)} full, CPU ${ob.cpu_load_pct.toFixed(0)}%`),
    tile("prop", "Propulsion", "warm gas", `${s.prop.propellant_g.toFixed(1)} g left`, `Δv remaining ${s.prop.dv_remaining_m_s.toFixed(1)} m/s`),
    tile("payload", "Payload & Edge AI", "camera", `${p.frames_acquired} frames`, `${p.decisions.kept} kept, ${p.decisions.alert} fire alerts`),
    tile("overview", "Orbit", "SSO 10:30", `${o.alt_km.toFixed(1)} km`, `${o.in_aoi ? "over an imaging area" : "outside imaging areas"}`),
  ].join("");
  // comparison
  const A = p.delivery_ai, B = p.delivery_baseline;
  const ratio = (d) => (d.frames_delivered ? (100 * d.useful_frames_delivered / d.frames_delivered).toFixed(0) + "%" : "—");
  const card = (cls, title, d, mem) => `<div class="cmp-card ${cls}"><h3>${title}</h3>
    <div class="big">${gb(d.useful_bytes / 1e9)}</div><div class="k">useful imagery on the ground</div>
    <dl><dt>Useful frames sent</dt><dd>${ratio(d)}</dd>
    <dt>Frames sent</dt><dd>${d.frames_delivered}</dd>
    <dt>Median age when sent</dt><dd>${d.median_frame_age_h == null ? "—" : d.median_frame_age_h.toFixed(1) + " h"}</dd>
    <dt>Fire alerts down</dt><dd>${d.fires_alerted}</dd>
    <dt>Fires imaged, down</dt><dd>${d.fires_image_on_ground}</dd>
    <dt>Memory used</dt><dd>${pct(mem.fill_pct, 1)}</dd>
    <dt>Lost, memory full</dt><dd>${d.dropped_memory_full}</dd></dl></div>`;
  const g = B.useful_bytes > 0 ? A.useful_bytes / B.useful_bytes : null;
  $("#cmp-body").innerHTML = `<div class="cmp-row">${card("ai", "Edge AI on board", A, p.memory)}${card("base", "Store everything", B, p.shadow_memory)}</div>
    <div class="gain">${g ? `With the same downlink, Edge AI has put <b>${g.toFixed(1)}×</b> more useful data on the ground.` : "Waiting for the first downlink pass over Bari."}</div>`;
  gmap.draw(s);
  renderTimeline(s);
}

function renderTimeline(s) {
  const H = 4 * 3600, t0 = s.t;
  const x = (t) => `${Math.max(0, Math.min(100, (t - t0) / H * 100))}%`;
  const w = (a, b) => `${Math.max(0.4, (Math.min(b, t0 + H) - Math.max(a, t0)) / H * 100)}%`;
  const bars = (items, k0, k1, cls) => items.filter((it) => it[k0] < t0 + H).map((it) =>
    `<div class="tl-bar ${cls}" style="left:${x(it[k0])};width:${w(it[k0], it[k1])}" title="${utcHM(epochUnix + it[k0])}–${utcHM(epochUnix + it[k1])}"></div>`).join("");
  let axis = "";
  for (let k = 0; k <= 4; k++) axis += `<span style="left:${k * 25}%">${utcHM(epochUnix + t0 + k * 3600)}</span>`;
  $("#timeline").innerHTML = `<div class="tl">
    <div class="tl-label" style="top:2px">Imaging areas</div><div class="tl-row" style="top:0">${bars(s.obdh.aoi_windows || [], "start", "end", "aoi")}</div>
    <div class="tl-label" style="top:26px">Bari contacts</div><div class="tl-row" style="top:24px">${bars(s.obdh.contacts || [], "aos", "los", "gs")}</div>
    <div class="tl-axis">${axis}</div></div>`;
}

function renderADCS(s) {
  const a = s.adcs;
  $("#adcs-kv").innerHTML = kv([
    ["Control mode", a.mode.replace("_", " ").toLowerCase()],
    ["Pointing error", `${fmt(a.pointing_error_deg, 3)}°`],
    ["Knowledge error", `${fmt(a.knowledge_error_arcsec, 1)}″`],
    ["Body rate", `${fmt(a.rate_norm_deg_s, 3)} °/s`],
    ["Total angular momentum", `${fmt(a.momentum_total_mNms, 2)} mNms`],
  ]);
  const stc = a.sensors.star_tracker;
  const wheel = (i) => a.wheels_seized[i] ? chip("seized", "bad") : a.wheels_isolated[i] ? chip("isolated", "warn") : chip("ok", "ok");
  $("#adcs-sensors").innerHTML = `<table class="plain">
    <tr><th>Sensor</th><th>State</th><th></th></tr>
    <tr><td>Star tracker</td><td>${chip(stc.toLowerCase().replace("_", " "), stc === "TRACKING" ? "ok" : ["FAILED", "BLINDED"].includes(stc) ? "bad" : "warn")}</td><td class="n">12″ 1σ</td></tr>
    <tr><td>Coarse Sun sensors</td><td>${chip(a.sensors.sun_sensor ? "Sun in view" : "eclipse", a.sensors.sun_sensor ? "ok" : "warn")}</td><td class="n">0.5°</td></tr>
    <tr><td>Magnetometer</td><td>${chip("ok", "ok")}</td><td class="n">${Math.hypot(...a.B_body_nT).toFixed(0)} nT</td></tr>
    <tr><td>Gyro bias, estimated / true</td><td colspan="2" class="n">${a.bias_est_deg_h.map((x) => x.toFixed(1)).join(" ")} / ${a.bias_true_deg_h.map((x) => x.toFixed(1)).join(" ")} °/h</td></tr>
    </table>
    <table class="plain" style="margin-top:10px">
    <tr><th>Actuator</th><th>X</th><th>Y</th><th>Z</th></tr>
    <tr><td>Wheel state</td><td>${wheel(0)}</td><td>${wheel(1)}</td><td>${wheel(2)}</td></tr>
    <tr><td>Wheel speed [rpm]</td>${a.wheel_rpm.map((x) => `<td class="n">${x.toFixed(0)}</td>`).join("")}</tr>
    <tr><td>Wheel torque [mNm]</td>${a.wheel_torque_mNm.map((x) => `<td class="n">${x.toFixed(3)}</td>`).join("")}</tr>
    <tr><td>Magnetorquer [A m²]</td>${a.mtq_dipole_Am2.map((x) => `<td class="n">${x.toFixed(3)}</td>`).join("")}</tr></table>`;
  const ax = ["X", "Y", "Z"];
  $("#mat-dcm").innerHTML = matrixHTML(a.dcm, { rows: ["body X", "body Y", "body Z"], cols: ["I x", "I y", "I z"], diverging: true, scale: 1, digits: 4 });
  const names = ["θx", "θy", "θz", "bx", "by", "bz"];
  // 1σ: attitude in arcsec, gyro bias in deg/h (rad/s x 206265 = arcsec/s = deg/h)
  const sig = a.cov_log10_diag.map((l) => Math.sqrt(10 ** l) * 206264.8);
  $("#mat-cov").innerHTML = matrixHTML(a.cov_corr, {
    rows: names, cols: names, diverging: true, scale: 1,
    cellText: (v, i, j) => (i === j ? (i < 3 ? `${sig[i].toFixed(1)}″` : `${sig[i].toFixed(2)} °/h`) : v.toFixed(2)),
  });
  const q = (v) => v.map((x) => `<td class="n">${x.toFixed(5)}</td>`).join("");
  $("#quats").innerHTML = `<table class="plain"><tr><th></th><th>x</th><th>y</th><th>z</th><th>w</th></tr>
    <tr><td>True</td>${q(a.q)}</tr><tr><td>Estimated (MEKF)</td>${q(a.q_est)}</tr><tr><td>Reference</td>${q(a.q_ref)}</tr></table>`;
  if (!$("#mat-inertia").innerHTML) $("#mat-inertia").innerHTML = matrixHTML(info.inertia, { rows: ax, cols: ax, digits: 4, diverging: true });
}

function renderEPS(s) {
  const e = s.eps;
  $("#eps-battery").innerHTML = `<div class="gauge-big num">${e.soc_pct.toFixed(1)}<small>%</small></div>
    <div class="bar" style="margin:8px 0 4px"><i style="width:${e.soc_pct}%;background:${e.soc < 0.3 ? C.caution : C.nominal}"></i></div>` +
    kv([["Bus voltage", `${e.bus_voltage_v.toFixed(2)} V`], ["Battery current", `${e.battery_current_a.toFixed(2)} A ${e.battery_current_a > 0.01 ? "charging" : e.battery_current_a < -0.01 ? "discharging" : ""}`],
      ["Capacity", `${e.capacity_wh.toFixed(0)} Wh (2S4P Li-ion)`], ["Solar available / used", `${e.solar_available_w.toFixed(1)} / ${e.solar_used_w.toFixed(1)} W`],
      ["Curtailed by MPPT", `${e.curtailed_w.toFixed(1)} W`], ["Charge inhibit (cold)", e.charge_inhibit ? "yes" : "no"],
      ["Energy in / out", `${e.energy_in_wh.toFixed(0)} / ${e.energy_out_wh.toFixed(0)} Wh`]]);
  const mx = Math.max(30, ...e.panel_w);
  $("#eps-panels").innerHTML = `<table class="plain"><tr><th>Panel</th><th>Power</th><th style="width:40%"></th><th>Health</th></tr>` +
    e.panel_names.map((n, i) => `<tr><td>${n}</td><td class="n">${e.panel_w[i].toFixed(1)} W</td><td><div class="bar"><i style="width:${e.panel_w[i] / mx * 100}%;background:${C.kapton}"></i></div></td><td class="n">${(e.panel_health[i] * 100).toFixed(0)}%</td></tr>`).join("") + "</table>";
  const sw = e.switches;
  const lcl = { camera: "payload", ai_unit: "ai_unit", transmitter: "transmitter", battery_heater: "battery_heater", payload_heater: "payload_heater", propulsion: "propulsion" };
  const label = { obc: "On-board computer", adcs_electronics: "ADCS electronics", reaction_wheels: "Reaction wheels", magnetorquers: "Magnetorquers", star_tracker: "Star tracker", gyro: "Gyro", gnss: "GNSS receiver", radio_rx: "S-band receiver", transmitter: "S-band transmitter", camera: "Camera", ai_unit: "Edge AI processor", battery_heater: "Battery heater", payload_heater: "Payload heater", propulsion: "Thruster valve" };
  const mxl = Math.max(1, ...Object.values(e.loads_w));
  $("#eps-loads").innerHTML = `<table class="plain"><tr><th>Load</th><th>Switch</th><th>Power</th><th style="width:45%"></th></tr>` +
    Object.entries(e.loads_w).map(([k, v]) => `<tr><td>${label[k] || k}</td><td>${lcl[k] ? chip(sw[lcl[k]] ? "on" : "off", sw[lcl[k]] ? "ok" : "") : chip("always on", "")}</td><td class="n">${v.toFixed(2)} W</td><td><div class="bar"><i style="width:${v / mxl * 100}%"></i></div></td></tr>`).join("") + "</table>";
}

function renderTCS(s) {
  const t = s.tcs;
  const lo = -100, hi = 120, X = (v) => `${(v - lo) / (hi - lo) * 100}%`;
  $("#tcs-nodes").innerHTML = `<table class="plain"><tr><th>Node</th><th>Temp.</th><th style="width:50%">Operating range</th><th></th></tr>` +
    t.nodes.map((n, i) => { const [mn, mx] = t.limits_c[i]; const T = t.temps_c[i];
      return `<tr><td>${n.replace("_", " ").toLowerCase()}</td><td class="n">${T.toFixed(1)} °C</td><td><div class="range"><div class="span"></div><div class="ok" style="left:${X(mn)};width:calc(${X(mx)} - ${X(mn)})"></div><div class="mark" style="left:${X(Math.max(lo, Math.min(hi, T)))};background:${t.status[i] === "GREEN" ? "#e4eaf4" : t.status[i] === "YELLOW" ? C.caution : C.alarm}"></div></div></td><td>${chip(t.status[i] === "GREEN" ? "ok" : t.status[i] === "YELLOW" ? "near limit" : "out of range", t.status[i] === "GREEN" ? "ok" : t.status[i] === "YELLOW" ? "warn" : "bad")}</td></tr>`; }).join("") +
    `</table><p class="hint">Battery heater ${t.heaters.battery_heater ? "on" : "off"}${t.heater_stuck ? " (switch stuck)" : ""}, payload heater ${t.heaters.payload_heater ? "on" : "off"}.</p>`;
  const f = t.flows;
  $("#tcs-flows").innerHTML = kv([["Direct solar absorbed", `${f.solar_w.toFixed(1)} W`], ["Earth albedo absorbed", `${f.albedo_w.toFixed(1)} W`],
    ["Earth infrared absorbed", `${f.earth_ir_w.toFixed(1)} W`], ["Internal dissipation", `${f.internal_w.toFixed(1)} W`], ["Radiated to space", `${f.radiated_w.toFixed(1)} W`]]) +
    `<p class="hint">Lumped-parameter network with ${t.nodes.length} nodes. MLI everywhere except the −X radiator, the payload aperture and the solar wings.</p>`;
  if (!$("#mat-g").innerHTML) {
    const short = { STRUCTURE: "struct.", SOLAR_ARRAY: "array", BATTERY: "battery", AVIONICS: "avionics", PAYLOAD: "payload", COMMS: "comms" };
    const n = info.tcs_nodes.map((x) => short[x] || x);
    $("#mat-g").innerHTML = matrixHTML(info.tcs_G, { rows: n, cols: n, digits: 2 });
  }
}

function renderComms(s) {
  const c = s.comms;
  if (!c.visible) {
    const nc = s.obdh.next_contact;
    $("#link-budget").innerHTML = `<p class="waiting">Spacecraft below Bari's 5° elevation mask (now ${c.elevation_deg.toFixed(0)}°). ${nc ? `Next contact at ${utcHM(epochUnix + nc.aos)} UTC, in ${dur(nc.aos - s.t)}, max elevation ${nc.max_el.toFixed(0)}°.` : "No contact in the next four hours."}</p>` +
      kv([["Sent this pass / total", `${gb(c.pass_bytes / 1e9)} / ${gb(c.total_bytes / 1e9)}`], ["Transmitter power amplifier", c.pa_degradation_db ? `degraded by ${c.pa_degradation_db} dB` : "nominal"]]);
    sky.draw(c);
    renderContacts(s);
    return;
  }
  $("#link-budget").innerHTML = `<table class="plain">
    <tr><td>Ground station</td><td class="n">${c.station}</td></tr>
    <tr><td>Elevation / azimuth</td><td class="n">${c.elevation_deg.toFixed(1)}° / ${c.azimuth_deg.toFixed(0)}°</td></tr>
    <tr><td>Slant range</td><td class="n">${c.range_km.toFixed(0)} km</td></tr>
    <tr><td>Doppler at 2245 MHz</td><td class="n">${c.doppler_khz.toFixed(1)} kHz</td></tr>
    <tr><td>Antenna off-boresight</td><td class="n">${c.off_boresight_deg.toFixed(1)}°</td></tr>
    <tr><td>Antenna gain</td><td class="n">${c.antenna_gain_dbi.toFixed(1)} dBi</td></tr>
    <tr><td>EIRP${c.pa_degradation_db ? " (PA degraded)" : ""}</td><td class="n">${c.eirp_dbw.toFixed(1)} dBW</td></tr>
    <tr><td>Free-space loss</td><td class="n">−${c.fspl_db.toFixed(1)} dB</td></tr>
    <tr><td>Atmospheric loss</td><td class="n">−${c.atm_loss_db.toFixed(1)} dB</td></tr>
    <tr><td>Ground G/T</td><td class="n">${(info.stations[0].g_over_t ?? 0).toFixed(1)} dB/K</td></tr>
    <tr><td>C/N₀</td><td class="n">${c.cn0_dbhz.toFixed(1)} dBHz</td></tr>
    <tr><td>Eb/N₀ at selected rate</td><td class="n">${c.ebn0_db.toFixed(1)} dB</td></tr>
    <tr><td>Margin over 2.5 dB required</td><td class="n">${c.margin_db.toFixed(1)} dB</td></tr>
    <tr><td>Adaptive data rate</td><td class="n">${c.rate_bps ? (c.rate_bps / 1e6).toFixed(2) + " Mbit/s" : "link not closed"}</td></tr>
    <tr><td>Transmitter</td><td class="n">${c.transmitting ? chip("transmitting", "ok") : chip("idle", "")}</td></tr>
    <tr><td>Sent this pass / total</td><td class="n">${gb(c.pass_bytes / 1e9)} / ${gb(c.total_bytes / 1e9)}</td></tr></table>`;
  sky.draw(c);
  renderContacts(s);
}

function renderContacts(s) {
  const cs = s.obdh.contacts || [];
  $("#contacts").innerHTML = cs.length ? `<table class="plain"><tr><th>Station</th><th>AOS (UTC)</th><th>LOS (UTC)</th><th>Duration</th><th>Max elevation</th><th>Starts in</th></tr>` +
    cs.map((x) => `<tr><td>${x.station}</td><td class="n">${utcHM(epochUnix + x.aos, true)}</td><td class="n">${utcHM(epochUnix + x.los, true)}</td><td class="n">${dur(x.los - x.aos)}</td><td class="n">${x.max_el.toFixed(0)}°</td><td class="n">${x.aos > s.t ? dur(x.aos - s.t) : "now"}</td></tr>`).join("") + "</table>"
    : `<p class="waiting">No contacts predicted in the next four hours.</p>`;
}

function renderOBDH(s) {
  const o = s.obdh, p = s.payload;
  $("#modes-svg").innerHTML = modesSVG(o.mode, o.activity);
  $("#obc-kv").innerHTML = kv([["Mode", o.mode.toLowerCase()], ["Activity", o.activity.replace("_", " ").toLowerCase()],
    ["Safe-mode reasons", o.safe_reasons.length ? o.safe_reasons.join(", ").toLowerCase() : "none"],
    ["Payload load-shedding", o.power_saving ? "active" : "no"], ["CPU load (est.)", `${o.cpu_load_pct.toFixed(0)}%`],
    ["Uptime since boot", met(o.uptime_s)], ["Watchdog resets", o.reboots],
    ["Mean semi-major axis", o.sma_mean_km ? `${o.sma_mean_km.toFixed(3)} km` : "averaging first orbit"],
    ["Battery heater LCL", o.heater_inhibit ? "opened by FDIR" : "closed"]]);
  const row = (name, m) => `<tr><td>${name}</td><td class="n">${m.used_gb.toFixed(2)} / ${m.capacity_gb.toFixed(0)} GB</td><td style="width:35%"><div class="bar"><i style="width:${m.fill_pct}%;background:${name.startsWith("Edge") ? C.kapton : C.ink3}"></i></div></td><td class="n">${m.counts.FRAME} frames, ${m.counts.ROI} crops, ${m.counts.ALERT} alerts</td></tr>`;
  $("#memory").innerHTML = `<table class="plain">${row("Edge AI", p.memory)}${row("Store everything", p.shadow_memory)}</table>`;
  renderEvents(s, $("#event-full"), 40);
}

function renderProp(s) {
  const p = s.prop, o = s.obdh;
  $("#prop-kv").innerHTML = `<div class="gauge-big num">${p.propellant_g.toFixed(1)}<small> g</small></div>
    <div class="bar" style="margin:8px 0 4px"><i style="width:${p.propellant_pct}%;background:${C.kapton}"></i></div>` +
    kv([["Δv used / remaining", `${(p.dv_used_m_s * 100).toFixed(1)} cm/s / ${p.dv_remaining_m_s.toFixed(1)} m/s`],
      ["Spacecraft mass", `${p.mass_kg.toFixed(3)} kg`], ["Thruster", p.firing ? "firing" : p.burn_remaining_s > 0 ? `burn pending, ${p.burn_remaining_s.toFixed(0)} s` : "idle"],
      ["Burns executed", p.burns], ["Latch valve", p.latch_closed ? "closed by FDIR" : "open"],
      ["Mean SMA vs reference", o.sma_mean_km && o.sma_ref_km ? `${((o.sma_mean_km - o.sma_ref_km) * 1000).toFixed(0)} m` : "—"]]) +
    `<p class="hint">Warm-gas thruster, 25 mN, Isp 105 s. The OBDH plans a prograde burn when the mean semi-major axis drops 150 m below its reference.</p>`;
  const bl = p.burn_log || [];
  $("#burns").innerHTML = bl.length ? `<table class="plain"><tr><th>Planned (UTC)</th><th>Δv commanded</th><th>Δv achieved</th><th>Duration</th></tr>` +
    bl.slice().reverse().map((b) => `<tr><td class="n">${utc(b.t_plan)}</td><td class="n">${(b.dv_cmd * 100).toFixed(1)} cm/s</td><td class="n">${(b.dv_done * 100).toFixed(1)} cm/s</td><td class="n">${b.duration.toFixed(0)} s</td></tr>`).join("") + "</table>"
    : `<p class="waiting">No burns yet. Drag lowers the orbit by roughly 100 m per day at this altitude.</p>`;
}

function renderPayload(s) {
  const p = s.payload;
  const fr = $("#frames");
  fr.innerHTML = s.recent_frames.map((f, i) => {
    const cls = f.decision.startsWith("FIRE") ? "fire" : f.decision.startsWith("KEEP") || f.decision.startsWith("STORED") ? "keep" : "drop";
    const probs = f.probs ? `<div class="probs" title="Cloud, water, land, fire">${f.probs.map((x, k) => `<i class="${k === 3 ? "f" : ""}" style="height:${Math.max(4, x * 100)}%"></i>`).join("")}</div>` : "";
    return `<div class="frame"><canvas width="32" height="32" data-i="${i}"></canvas><div class="dec ${cls}">${f.decision.toLowerCase().replace("fire alert", "Fire alert").replace(/^./, (c) => c.toUpperCase())}</div>
      <div class="meta">truth: ${CLASS_LABEL[f.truth].toLowerCase()}<br>${utcHM(epochUnix + f.t, true)}, ${f.lat}°, ${f.lon}°</div>${probs}</div>`;
  }).join("") || `<p class="waiting">No frames yet. The camera turns on over the imaging areas in daylight.</p>`;
  fr.querySelectorAll("canvas").forEach((cv) => {
    const f = s.recent_frames[+cv.dataset.i];
    const bin = atob(f.rgb), img = cv.getContext("2d").createImageData(32, 32);
    for (let k = 0, j = 0; k < 1024; k++) { img.data[j++] = bin.charCodeAt(3 * k); img.data[j++] = bin.charCodeAt(3 * k + 1); img.data[j++] = bin.charCodeAt(3 * k + 2); img.data[j++] = 255; }
    cv.getContext("2d").putImageData(img, 0, 0);
  });
  const d = p.decisions, tot = Math.max(1, Object.values(d).reduce((a, b) => a + b, 0));
  const lab = { kept: "Kept, ranked by value", alert: "Wildfire alert + crop", discard_cloud: "Discarded: cloud", discard_water: "Discarded: open water", discard_lowvalue: "Discarded: low value", discard_blur: "Discarded: smear (from ADCS)", fallback_store: "Stored without AI" };
  $("#decisions").innerHTML = `<table class="plain">${Object.entries(d).map(([k, v]) => `<tr><td>${lab[k]}</td><td class="n">${v}</td><td style="width:40%"><div class="bar"><i style="width:${v / tot * 100}%;background:${k === "alert" ? C.alarm : k === "kept" ? C.nominal : C.ink3}"></i></div></td></tr>`).join("")}</table>
    <p class="hint">${p.frames_acquired} frames of ${p.frame_mb.toFixed(1)} MB acquired, ${p.blurred} smeared by attitude motion. AI processor ${p.ai_hung ? "hung" : p.ai_powered ? `on, ${(p.ai_busy_frac * 100).toFixed(0)}% busy` : "off"}, queue ${p.ai_queue}.</p>`;
  $("#mat-conf").innerHTML = matrixHTML(p.confusion, { rows: CLASSES.map((c) => CLASS_LABEL[c]), cols: CLASSES.map((c) => CLASS_LABEL[c]), digits: 0, scale: Math.max(1, ...p.confusion.flat()) });
  const m = info.model;
  if (!$("#model-card").innerHTML) $("#model-card").innerHTML = kv([["Architecture", "3 conv layers + GAP/GMP + dense"], ["Parameters", m.params.toLocaleString()],
    ["Size in int8", `${(m.size_bytes_int8 / 1024).toFixed(1)} KB`], ["MACs per 32×32 tile", m.macs_per_tile.toLocaleString()],
    ["Processing per frame", `${info.payload.t_infer_ms.toFixed(0)} ms (64 tiles + I/O)`], ["Validation accuracy float / int8", `${(m.val_accuracy_float * 100).toFixed(1)}% / ${(m.val_accuracy_int8 * 100).toFixed(1)}%`],
    ["Wildfire recall / precision", `${(m.recall_int8.FIRE * 100).toFixed(1)}% / ${(m.precision_int8.FIRE * 100).toFixed(1)}%`]]) +
    `<p class="hint">Trained on procedurally generated B, G, R, NIR scenes. Inference uses integer arithmetic as on a microcontroller.</p>`;
  const al = p.alerts;
  $("#alerts").innerHTML = al.length ? `<table class="plain"><tr><th>Detected (UTC)</th><th>Location</th><th>Confidence</th><th>Real fire</th><th>Alert on ground</th></tr>` +
    al.map((a) => `<tr><td class="n">${utcHM(epochUnix + a.t, true)}</td><td class="n">${a.lat}°, ${a.lon}°</td><td class="n">${(a.conf * 100).toFixed(0)}%</td><td>${a.true ? chip("yes", "ok") : chip("false alarm", "warn")}</td><td>${a.delivered ? chip("delivered", "ok") : chip("queued for next Bari pass", "")}</td></tr>`).join("") + "</table>"
    : `<p class="waiting">No wildfire detected yet. Alerts are queued ahead of all imagery and go down at the next Bari pass.</p>`;
}

// ---------------------------------------------------------------------------------- main loop
function onSnapshot(s) {
  if (lastScenario !== null && (s.t < lastT || s.scenario !== lastScenario)) { tm.clear(); sky.trace = []; }
  lastT = s.t; lastScenario = s.scenario;
  tm.push(s.series_rows || []);
  S = s; fresh = true;
}

function renderAll() {
  if (!S || !fresh) return;
  fresh = false;
  const s = S, st = statuses(s);
  renderHeader(s, st);
  renderEvents(s, $("#events"), 25);
  globe.update(s);
  attSmall.update(s);
  const fns = { overview: renderOverview, adcs: renderADCS, eps: renderEPS, tcs: renderTCS, comms: renderComms, obdh: renderOBDH, prop: renderProp, payload: renderPayload };
  fns[view](s, st);
  if (view === "adcs") { attBig = attBig || new AttitudeView($("#att-big"), info.body_dims); attBig.update(s); }
  for (const el of $$(`#view-${view} [data-chart]`)) charts[el.dataset.chart].update();
}

function frame() {
  renderAll();
  if (view === "overview") { globe.render(); attSmall.render(); }
  if (view === "adcs" && attBig) attBig.render();
  requestAnimationFrame(frame);
}

function show(v) {
  view = v;
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === `view-${v}`));
  fresh = true;
}

function bind() {
  $$(".nav-item").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));
  $("#tiles").addEventListener("click", (e) => { const t = e.target.closest("[data-goto]"); if (t) show(t.dataset.goto); });
  $$(".warp button").forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.warp === "pause") cmd("pause", !(S && S.paused));
    else { cmd("pause", false); cmd("warp", +b.dataset.warp); }
  }));
  $("#ai-toggle").addEventListener("change", (e) => { cmd("ai", e.target.checked); toast(e.target.checked ? "Edge AI enabled" : "Edge AI disabled: frames stored without screening"); });
  $$("[data-reset]").forEach((b) => b.addEventListener("click", () => { cmd("reset", b.dataset.reset); toast(b.dataset.reset === "leop" ? "Restarted after separation from the launcher" : "Restarted in nominal operations"); }));
  $$("[data-cmd]").forEach((b) => b.addEventListener("click", () => { cmd(b.dataset.cmd, b.dataset.arg); if (b.dataset.cmd !== "inject") toast(b.textContent); }));
  $("#burn-btn").addEventListener("click", () => { cmd("burn", 0.05); toast("Burn queued: 5 cm/s prograde"); });
  const ft = $("#follow-toggle");
  ft.addEventListener("click", () => { globe.follow = !globe.follow; ft.setAttribute("aria-pressed", globe.follow); });
  globe.onFollow = (f) => ft.setAttribute("aria-pressed", f);
  $("#faults").innerHTML = Object.entries(info.faults).map(([k, v]) => `<button class="fault-btn" data-fault="${k}">${v}</button>`).join("");
  $("#faults").addEventListener("click", (e) => { const b = e.target.closest("[data-fault]"); if (b) cmd("inject", b.dataset.fault); });
}

async function main() {
  [info, land] = await Promise.all([fetch("/api/info").then((r) => r.json()), fetch("/api/land").then((r) => r.json())]);
  epochUnix = Date.parse(info.mission.epoch_utc + "Z") / 1000;
  tm = new Telemetry(info.series, epochUnix);
  globe = new Globe($("#globe"), info, land);
  attSmall = new AttitudeView($("#att-small"), info.body_dims);
  gmap = new GroundMap($("#map2d"), info, land);
  sky = new SkyPlot($("#skyplot"));
  defineCharts();
  bind();
  const es = new EventSource("/api/stream");
  es.onmessage = (ev) => onSnapshot(JSON.parse(ev.data));
  es.onerror = () => { $("#achieved").textContent = "reconnecting…"; };
  requestAnimationFrame(frame);
  window.__edgesat = { get state() { return S; }, show };
}
main();
