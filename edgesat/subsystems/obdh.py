"""On-Board Data Handling: mode management, autonomous planning, FDIR, housekeeping.

Modes
    LEOP     post-separation: B-dot detumbling, then Sun acquisition
    SAFE     Sun pointing, payload off, telemetry only; automatic recovery
    NOMINAL  autonomous operations (activities below)
    REBOOT   OBC restarting after a single-event upset; ADCS control lost

Activities in NOMINAL (decided every step from an on-board orbit predictor)
    IMAGING      nadir pointing over an area of interest in daylight
    DOWNLINK     ground-station tracking for maximum antenna gain
    ORBIT_MAINT  align thruster with velocity and fire to restore the SMA
    CHARGE       Sun pointing for maximum power

FDIR (Fault Detection, Isolation and Recovery) watches every subsystem and
reacts autonomously; every action is written to the event log.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..core.constants import DEG, MU_EARTH
from ..core.orbit import eci_to_latlon_batch, predict_states

INFO, WARN, ALARM = "INFO", "WARN", "ALARM"


class OBDH:
    def __init__(self, cfg: dict, scenario: str):
        self.cfg = cfg
        self.fdir_cfg = cfg["fdir"]
        self.eps_cfg = cfg["eps"]
        self.mode = "LEOP" if scenario == "leop" else "NOMINAL"
        self.activity = "DETUMBLE" if self.mode == "LEOP" else "CHARGE"
        self.leop_phase = "DETUMBLE"
        self.mode_since = 0.0
        self.events: deque[dict] = deque(maxlen=200)
        self.reboot_until = -1.0
        self.power_saving = False
        self.safe_reasons: set[str] = set()
        self.timers: dict[str, float] = {}
        self.flags: set[str] = set()
        self.prediction: dict = {"t": -1e9}
        self.sma_hist: deque[float] = deque()
        self.sma_ref: float | None = None
        self.sma_mean: float | None = None
        self.hk_accum = 0.0
        self.cmd_override: str | None = None
        self.uptime = 0.0
        self.reboots = 0
        self.heater_inhibit = False
        self.tm: dict = {}
        self.log(0.0, INFO, "OBDH", f"boot complete, mode {self.mode}")

    # ------------------------------------------------------------------ logging
    def log(self, t: float, sev: str, sub: str, msg: str) -> None:
        self.events.appendleft({"t": t, "sev": sev, "sub": sub, "msg": msg})

    def set_mode(self, t: float, mode: str, why: str) -> None:
        if mode != self.mode:
            sev = ALARM if mode in ("SAFE", "REBOOT") else INFO
            self.log(t, sev, "OBDH", f"mode {self.mode} → {mode}: {why}")
            self.mode = mode
            self.mode_since = t

    # ------------------------------------------------------------------ prediction
    def predict(self, t: float, jd: float, r: np.ndarray, v: np.ndarray, sun_eci_u: np.ndarray,
                stations, aois, min_el_deg: float, min_sun_deg: float) -> None:
        horizon, step = 4 * 3600.0, 20.0
        ts, rs, _ = predict_states(r, v, horizon, step)
        jds = jd + ts / 86400.0
        lat, lon, r_ecef = eci_to_latlon_batch(rs, jds)
        from ..core.orbit import gmst
        th = gmst(jds)
        sun_x = np.cos(th) * sun_eci_u[0] + np.sin(th) * sun_eci_u[1]
        sun_y = -np.sin(th) * sun_eci_u[0] + np.cos(th) * sun_eci_u[1]
        la, lo = np.radians(lat), np.radians(lon)
        up = np.column_stack((np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)))
        sun_el = np.degrees(np.arcsin(np.clip(up[:, 0] * sun_x + up[:, 1] * sun_y + up[:, 2] * sun_eci_u[2], -1, 1)))
        in_aoi = np.zeros(len(ts), bool)
        for a in aois:
            in_aoi |= (lat >= a["lat"][0]) & (lat <= a["lat"][1]) & (lon >= a["lon"][0]) & (lon <= a["lon"][1])
        in_aoi &= sun_el >= min_sun_deg
        contacts = []
        best_el = np.full(len(ts), -90.0)
        for gs in stations:
            rho = r_ecef - gs.ecef
            gu = gs.ecef / np.linalg.norm(gs.ecef)
            el = np.degrees(np.arcsin(rho @ gu / np.linalg.norm(rho, axis=1)))
            best_el = np.maximum(best_el, el)
            vis = el >= min_el_deg
            for s, e in _segments(vis):
                contacts.append({"station": gs.name, "aos": t + ts[s], "los": t + ts[e],
                                 "max_el": float(el[s:e + 1].max())})
        aoi_windows = [{"start": t + ts[s], "end": t + ts[e]} for s, e in _segments(in_aoi)]
        self.prediction = {"t": t, "contacts": sorted(contacts, key=lambda c: c["aos"]),
                           "aoi_windows": aoi_windows,
                           "track": {"t": (t + ts[::3]).tolist(), "lat": lat[::3].round(2).tolist(),
                                     "lon": lon[::3].round(2).tolist()}}

    def _next(self, t: float, items: list[dict], key_s: str, key_e: str):
        for it in items:
            if it[key_e] >= t:
                return it
        return None

    # ------------------------------------------------------------------ planning
    def plan(self, t: float, sc) -> dict:
        """Decide mode/activity and command the subsystems. Returns plan dict."""
        e = self.eps_cfg
        soc = sc.eps.soc
        env = sc.env
        # ---- REBOOT
        if self.mode == "REBOOT":
            if t >= self.reboot_until:
                self.set_mode(t, "SAFE", "OBC restarted")
                self.safe_reasons.add("OBC_SEU")
            else:
                return self._apply(sc, "FREE_DRIFT", imaging=False, ai=False, tx=False, activity="REBOOT",
                                   adcs_enabled=False)
        # ---- power saving
        if soc < e["soc_payload_off"] and not self.power_saving:
            self.power_saving = True
            self.log(t, WARN, "EPS", f"SoC {soc * 100:.0f}% < {e['soc_payload_off'] * 100:.0f}%: payload load-shed")
        elif soc > e["soc_recover"] and self.power_saving:
            self.power_saving = False
            self.log(t, INFO, "EPS", "SoC recovered: payload re-enabled")
        if soc < e["soc_safe"] and self.mode == "NOMINAL":
            self.safe_reasons.add("LOW_SOC")
            self.set_mode(t, "SAFE", f"battery SoC {soc * 100:.0f}%")
        # ---- LEOP sequence
        if self.mode == "LEOP":
            rate = sc.adcs.tm.get("rate_norm_deg_s", 10.0) if sc.adcs.tm else 10.0
            if self.leop_phase == "DETUMBLE":
                self._timer("detumbled", rate < 0.35, t, sc.dt)
                if self.timers.get("detumbled", 0) > 120:
                    self.leop_phase = "SUN_ACQ"
                    self.log(t, INFO, "ADCS", "detumbling complete, starting Sun acquisition")
                return self._apply(sc, "DETUMBLE", imaging=False, ai=False, tx=True, activity="DETUMBLE")
            perr = sc.adcs.tm.get("pointing_error_deg", 99)
            self._timer("sun_acq", perr < 3.0, t, sc.dt)
            if self.timers.get("sun_acq", 0) > 300:
                self.set_mode(t, "NOMINAL", "LEOP complete: attitude acquired")
            else:
                return self._apply(sc, "SUN_POINT", imaging=False, ai=False, tx=True, activity="SUN_ACQ")
        # ---- SAFE
        if self.mode == "SAFE":
            ok = (soc >= e["soc_recover"] or "LOW_SOC" not in self.safe_reasons)
            if ok and t - self.mode_since >= self.fdir_cfg["safe_mode_auto_recovery_s"] and sc.adcs.tm.get(
                    "pointing_error_deg", 99) < 5.0:
                self.safe_reasons.clear()
                self.set_mode(t, "NOMINAL", "autonomous recovery")
            else:
                return self._apply(sc, "SUN_POINT", imaging=False, ai=False, tx=soc > 0.12, activity="SAFE")

        # ---- NOMINAL: refresh predictions
        if t - self.prediction["t"] > 300.0:
            self.predict(t, env.jd, env.r, env.v, env.sun_u, sc.comms.stations, sc.cfg["payload"]["aoi"],
                         sc.cfg["comms"]["min_elevation_deg"], sc.cfg["payload"]["min_sun_elevation_deg"])
        lead = 150.0
        in_aoi = sc.in_aoi_now()
        gs_vis = sc.comms.tm.get("visible", False) if sc.comms.tm else False
        nc = self._next(t, self.prediction["contacts"], "aos", "los")
        na = self._next(t, self.prediction["aoi_windows"], "start", "end")
        aos_soon = nc is not None and nc["aos"] - t < lead
        aoi_soon = na is not None and na["start"] - t < lead
        burn_pending = sc.prop.burn_remaining > 0 and not sc.prop.latch_closed
        quiet = (not in_aoi and not gs_vis and (na is None or na["start"] - t > 600)
                 and (nc is None or nc["aos"] - t > 600))
        if self.cmd_override:
            act = self.cmd_override
        elif burn_pending and (quiet or sc.prop.firing):
            act = "ORBIT_MAINT"
        elif (in_aoi or aoi_soon) and not self.power_saving:
            act = "IMAGING"
        elif gs_vis or aos_soon:
            act = "DOWNLINK"
        else:
            act = "CHARGE"
        adcs_mode = {"IMAGING": "NADIR", "DOWNLINK": "GS_TRACK", "ORBIT_MAINT": "DELTA_V",
                     "CHARGE": "SUN_POINT"}[act]
        imaging = act == "IMAGING" and in_aoi and not self.power_saving
        ai_on = (imaging or len(sc.payload.queue) > 0) and not self.power_saving
        return self._apply(sc, adcs_mode, imaging=imaging, ai=ai_on, tx=True, activity=act)

    def _apply(self, sc, adcs_mode, imaging, ai, tx, activity, adcs_enabled=True) -> dict:
        if activity != self.activity:
            self.log(sc.t, INFO, "OBDH", f"activity {self.activity} → {activity}")
        self.activity = activity
        sc.adcs.mode = adcs_mode
        sw = sc.eps.switches
        sw["payload"] = imaging
        sw["ai_unit"] = ai
        sw["transmitter"] = tx
        sc.payload.ai_powered = ai
        return {"imaging": imaging, "adcs_enabled": adcs_enabled}

    def _timer(self, name: str, cond: bool, t: float, dt: float = 1.0) -> float:
        self.timers[name] = self.timers.get(name, 0.0) + dt if cond else 0.0
        return self.timers[name]

    # ------------------------------------------------------------------ orbit maintenance
    def track_sma(self, t: float, r: np.ndarray, v: np.ndarray, period: float, sc) -> None:
        if int(t // 10) == int((t - sc.dt) // 10):
            return
        a = 1.0 / (2.0 / np.linalg.norm(r) - (v @ v) / MU_EARTH)
        self.sma_hist.append(a)
        n = int(period / 10)
        while len(self.sma_hist) > n:
            self.sma_hist.popleft()
        if len(self.sma_hist) == n:
            self.sma_mean = float(np.mean(self.sma_hist))
            if self.sma_ref is None:
                self.sma_ref = self.sma_mean
                self.log(t, INFO, "PROP", f"reference mean SMA set to {self.sma_ref / 1000:.3f} km")
            loss = self.sma_ref - self.sma_mean
            db = self.cfg["propulsion"]["sma_deadband_m"]
            if loss > db and sc.prop.burn_remaining <= 0 and not sc.prop.latch_closed and self.mode == "NOMINAL":
                da = loss + 0.5 * db
                vcirc = np.sqrt(MU_EARTH / self.sma_mean)
                dv = vcirc * da / (2.0 * self.sma_mean)
                dur = sc.prop.plan_burn(dv, t)
                self.log(t, INFO, "PROP", f"SMA decayed {loss:.0f} m: planned {dv * 100:.1f} cm/s burn ({dur:.0f} s)")

    # ------------------------------------------------------------------ FDIR
    def fdir(self, t: float, dt: float, sc) -> None:
        f = self.fdir_cfg
        # reaction wheel anomaly: commanded vs measured momentum
        for i in range(3):
            if sc.adcs.wheel_mismatch_s[i] > 5.0 and not sc.adcs.faults.wheel_isolated[i]:
                sc.adcs.faults.wheel_isolated[i] = True
                sc.adcs.h_expected[i] = sc.adcs.h_w[i]
                self.log(t, ALARM, "FDIR", f"reaction wheel {'XYZ'[i]} anomaly: isolated, axis handed to magnetorquers")
                self.safe_reasons.add("WHEEL")
                self.set_mode(t, "SAFE", f"wheel {'XYZ'[i]} failure")
        # battery over-temperature: open heater LCL
        tb = sc.tcs.temps_c()["BATTERY"]
        heater_on = sc.loads.get("battery_heater", 0.0) > 0.0
        # heater current measured while the thermostat commands it off -> welded switch
        stuck = heater_on and not sc.tcs.heater_cmd["battery_heater"]
        if self._timer("heater_mismatch", stuck, t, dt) > 60 and not self.heater_inhibit:
            self.heater_inhibit = True
            self.log(t, ALARM, "FDIR", "battery heater current with switch commanded OFF: upstream LCL opened")
            self.log(t, WARN, "TCS", "battery heater unavailable: cold-case margin reduced")
        if tb > f["battery_overtemp_C"]:
            if heater_on and not self.heater_inhibit:
                self.heater_inhibit = True
                self.log(t, ALARM, "FDIR", f"battery {tb:.1f} C with heater on: heater LCL opened (switch failure)")
            elif "batt_hot" not in self.flags:
                self.flags.add("batt_hot")
                self.log(t, WARN, "TCS", f"battery temperature {tb:.1f} C above limit")
        elif tb < f["battery_overtemp_C"] - 5:
            self.flags.discard("batt_hot")
        # propellant leak: mass decreasing while valve closed
        if sc.prop.leaking and not sc.prop.latch_closed and not sc.prop.firing:
            if self._timer("leak", True, t, dt) > 20:
                sc.prop.latch_closed = True
                sc.prop.burn_remaining = 0.0
                self.log(t, ALARM, "FDIR", "propellant pressure decay with valve closed: latch valve closed")
        # AI processor watchdog
        if sc.payload.ai_powered and t < sc.payload.ai_hang_until:
            if self._timer("ai_wd", True, t, dt) > f["ai_watchdog_s"]:
                sc.payload.ai_hang_until = t
                self.timers["ai_wd"] = 0.0
                self.log(t, WARN, "FDIR", "AI unit watchdog expired: power-cycled, processing resumed")
        # star tracker long outage
        st = sc.adcs.tm.get("sensors", {}).get("star_tracker", "TRACKING")
        if self._timer("st_out", st in ("FAILED", "BLINDED"), t, dt) > 60.0:
            if "st_out" not in self.flags:
                self.flags.add("st_out")
                self.log(t, WARN, "ADCS", f"star tracker unavailable ({st}): MEKF on Sun sensor + magnetometer")
        else:
            self.flags.discard("st_out")
        # attitude excursion in nominal operations
        perr = sc.adcs.tm.get("pointing_error_deg", 0.0)
        if self.mode == "NOMINAL" and self._timer("perr", perr > 20.0 and sc.adcs.mode != "DETUMBLE", t, dt) > 400:
            self.safe_reasons.add("ATTITUDE")
            self.set_mode(t, "SAFE", f"pointing error {perr:.0f} deg for > 400 s")
        # tumbling
        if self.mode != "LEOP" and sc.adcs.tm.get("rate_norm_deg_s", 0) > 4.0:
            self.log(t, ALARM, "FDIR", "rate > 4 deg/s: re-entering detumble")
            self.mode, self.leop_phase = "LEOP", "DETUMBLE"
            self.timers["detumbled"] = 0.0

    # ------------------------------------------------------------------ housekeeping
    def housekeeping(self, t: float, dt: float, sc) -> None:
        self.uptime += dt
        self.hk_accum += sc.cfg["comms"]["hk_rate_bps"] * dt / 8.0
        if self.hk_accum >= 7500.0:
            sc.payload.add_hk(self.hk_accum, t)
            self.hk_accum = 0.0

    def seu(self, t: float) -> None:
        self.reboots += 1
        self.reboot_until = t + self.fdir_cfg["obc_reboot_s"]
        self.set_mode(t, "REBOOT", "single-event upset in OBC: watchdog reset")
        self.uptime = 0.0

    def telemetry(self, t: float, sc) -> dict:
        nc = self._next(t, self.prediction.get("contacts", []), "aos", "los") if self.prediction.get("contacts") else None
        na = self._next(t, self.prediction.get("aoi_windows", []), "start", "end") if self.prediction.get("aoi_windows") else None
        load = 18.0 + 9.0 * (self.activity == "IMAGING") + 5.0 * (self.activity == "DOWNLINK") \
            + 4.0 * (sc.adcs.tm.get("sensors", {}).get("star_tracker") != "TRACKING")
        self.tm = {
            "mode": self.mode, "activity": self.activity, "leop_phase": self.leop_phase,
            "power_saving": self.power_saving, "safe_reasons": sorted(self.safe_reasons),
            "uptime_s": self.uptime, "reboots": self.reboots, "cpu_load_pct": load,
            "next_contact": nc, "next_aoi": na,
            "contacts": [c for c in self.prediction.get("contacts", []) if c["los"] >= t][:6],
            "aoi_windows": [w for w in self.prediction.get("aoi_windows", []) if w["end"] >= t][:8],
            "sma_mean_km": None if self.sma_mean is None else self.sma_mean / 1000.0,
            "sma_ref_km": None if self.sma_ref is None else self.sma_ref / 1000.0,
            "heater_inhibit": self.heater_inhibit,
        }
        return self.tm


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


__all__ = ["OBDH", "DEG"]
