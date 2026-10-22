"""Simulation driver: configuration loading, stepping, telemetry snapshots, commands."""
from __future__ import annotations

import copy
import time
import tomllib
from pathlib import Path

import numpy as np

from .spacecraft import ROOT, Spacecraft

DEFAULT_CONFIG = ROOT / "config" / "mission.toml"

# scalar channels sampled for the dashboard strip charts
SERIES = [
    "t", "soc", "solar_w", "load_w", "bus_v", "batt_i",
    "point_err", "know_err", "rate", "h_x", "h_y", "h_z",
    "T_STRUCTURE", "T_SOLAR_ARRAY", "T_BATTERY", "T_AVIONICS", "T_PAYLOAD", "T_COMMS",
    "el", "margin", "rate_mbps", "mem_ai", "mem_base", "useful_ai", "useful_base",
    "alt_km", "illum", "sig_x", "sig_y", "sig_z", "tq_gg", "tq_aero", "tq_srp", "tq_mag", "sma_dev",
]


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "rb") as f:
        return tomllib.load(f)


class Simulation:
    def __init__(self, config: dict | str | Path | None = None, scenario: str = "nominal",
                 ai_enabled: bool | None = None, seed: int | None = None):
        self.base_cfg = config if isinstance(config, dict) else load_config(config)
        self.scenario = scenario
        self.seed = seed
        self.ai_override = ai_enabled
        self.reset(scenario)

    def reset(self, scenario: str | None = None) -> None:
        cfg = copy.deepcopy(self.base_cfg)
        if self.ai_override is not None:
            cfg["payload"]["ai"]["enabled"] = self.ai_override
        self.scenario = scenario or self.scenario
        self.sc = Spacecraft(cfg, self.scenario, self.seed)
        self.series_buf: list[list[float]] = []
        self.wall_per_step = 0.0

    @property
    def t(self) -> float:
        return self.sc.t

    def step(self, n: int = 1) -> None:
        t0 = time.perf_counter()
        for _ in range(n):
            self.sc.step()
        self.wall_per_step = (time.perf_counter() - t0) / max(n, 1)

    def run(self, duration_s: float, sample_every_s: float = 0.0, callback=None) -> None:
        steps = int(round(duration_s / self.sc.dt))
        next_sample = self.sc.t
        for _ in range(steps):
            self.sc.step()
            if sample_every_s and self.sc.t >= next_sample:
                self.sample()
                next_sample += sample_every_s
                if callback:
                    callback(self)

    # ------------------------------------------------------------------ telemetry
    def sample(self) -> list[float]:
        sc = self.sc
        a, e, tc, c, p = sc.adcs.tm, sc.eps.tm, sc.tcs.tm, sc.comms.tm, sc.payload.tm
        if not (a and e and tc and c and p):
            return []
        T = tc["temps_c"]
        tq = a["torque_uNm"]
        sig = a["sigma3_arcsec"]
        row = [
            sc.t, e["soc_pct"], e["solar_used_w"], e["load_w"], e["bus_voltage_v"], e["battery_current_a"],
            a["pointing_error_deg"], a["knowledge_error_arcsec"], a["rate_norm_deg_s"],
            *a["wheel_h_mNms"], *T,
            c["elevation_deg"], c["margin_db"], c["rate_bps"] / 1e6 if c["transmitting"] else 0.0,
            p["memory"]["fill_pct"], p["shadow_memory"]["fill_pct"],
            p["delivery_ai"]["useful_bytes"] / 1e9, p["delivery_baseline"]["useful_bytes"] / 1e9,
            sc.env.alt / 1000.0, sc.env.illum, *sig,
            tq["gravity_gradient"], tq["aerodynamic"], tq["solar_pressure"], tq["magnetic"],
            None if sc.obdh.sma_mean is None or sc.obdh.sma_ref is None else sc.obdh.sma_mean - sc.obdh.sma_ref,
        ]
        row = [None if x is None else round(float(x), 5) for x in row]
        self.series_buf.append(row)
        return row

    def pop_series(self) -> list[list[float]]:
        out, self.series_buf = self.series_buf, []
        return out

    def snapshot(self) -> dict:
        sc = self.sc
        env = sc.env
        o = sc.obdh.telemetry(sc.t, sc)
        alerted = sc.payload.store.stats.fire_alert_ground_t
        # compact: [lat, lon, state] with state 0 = burning, 1 = imaged, 2 = alert on the ground
        fires = [[round(f.lat, 2), round(f.lon, 2),
                  2 if f.fid in alerted else 1 if f.first_acq_t is not None else 0]
                 for f in (sc.fires.fires[i] for i in sc.fires.active(sc.t))]
        adcs = dict(sc.adcs.tm)
        comms = {k: v for k, v in sc.comms.tm.items() if k != "gs_eci"}
        jd = env.jd
        return {
            "t": sc.t, "jd": jd, "epoch": sc.cfg["mission"]["epoch_utc"], "scenario": sc.scenario,
            "wall_ms_per_step": self.wall_per_step * 1000.0,
            "orbit": {"lat": float(np.degrees(env.lat)), "lon": float(np.degrees(env.lon)),
                      "alt_km": env.alt / 1000.0, "r_eci": env.r.tolist(), "v_eci": env.v.tolist(),
                      "r_ecef": env.r_ecef.tolist(), "speed_km_s": float(np.linalg.norm(env.v)) / 1000,
                      "illum": env.illum, "sun_eci": env.sun_u.tolist(), "sun_ecef": env.sun_ecef_u.tolist(),
                      "sun_elev_subsat_deg": float(np.degrees(env.sun_elev_subsat)),
                      "density": env.rho, "B_nT": float(np.linalg.norm(env.B_eci) * 1e9),
                      "in_aoi": sc.in_aoi_now(), "track_hist": sc.track_hist[-200:],
                      "prediction": sc.obdh.prediction.get("track")},
            "adcs": adcs, "eps": sc.eps.tm, "tcs": sc.tcs.tm, "comms": comms,
            "prop": sc.prop.tm, "payload": sc.payload.tm, "recent_frames": list(sc.payload.recent),
            "obdh": o, "events": list(sc.obdh.events)[:40], "fires": fires,
            "energy": {"tx_wh": sc.stats_energy["tx_wh"], "ai_wh": sc.stats_energy["ai_wh"],
                       "by_load_wh": sc.eps.energy_by_load_wh},
        }

    def static_info(self) -> dict:
        sc = self.sc
        cfg = sc.cfg
        import json
        model_card = json.loads((ROOT / "edgesat" / "ai" / "model_card.json").read_text())
        return {
            "mission": cfg["mission"], "elements": sc.elements0, "period_s": sc.period,
            "stations": [{"name": g.name, "lat": np.degrees(g.lat), "lon": np.degrees(g.lon), "g_over_t": g.g_over_t}
                         for g in sc.comms.stations],
            "aois": cfg["payload"]["aoi"], "fire_zones": cfg["scenario"]["fire_zone"],
            "inertia": cfg["spacecraft"]["inertia_kgm2"], "body_dims": cfg["spacecraft"]["body_dims_m"],
            "tcs_nodes": sc.tcs.names, "tcs_G": sc.tcs.G.tolist(),
            "tcs_C": sc.tcs.C.tolist(), "panels": [p["name"] for p in sc.eps.panels],
            "faults": Spacecraft.FAULTS, "series": SERIES, "model": model_card,
            "comms": {k: v for k, v in cfg["comms"].items() if k != "ground_stations"},
            "payload": {"frame_mb": sc.payload.frame_bytes / 1e6, "swath_km": sc.payload.swath_km,
                        "t_infer_ms": sc.payload.t_infer * 1000, "memory_gb": cfg["payload"]["mass_memory_GB"]},
            "battery_wh": sc.eps.capacity_wh,
        }

    # ------------------------------------------------------------------ commands
    def command(self, cmd: str, arg=None) -> str:
        sc = self.sc
        if cmd == "inject":
            return sc.inject(arg)
        if cmd == "repair_all":
            sc.repair_all()
            return "hardware restored"
        if cmd == "ai":
            sc.payload.ai_enabled = bool(arg)
            sc.obdh.log(sc.t, "INFO", "GROUND", f"edge AI {'enabled' if arg else 'disabled'}")
            return "ok"
        if cmd == "mode":
            if arg == "SAFE":
                sc.obdh.safe_reasons.add("GROUND")
                sc.obdh.set_mode(sc.t, "SAFE", "ground command")
            elif arg == "NOMINAL":
                sc.obdh.safe_reasons.clear()
                sc.obdh.set_mode(sc.t, "NOMINAL", "ground command")
            return "ok"
        if cmd == "activity":
            sc.obdh.cmd_override = None if arg in (None, "AUTO") else arg
            sc.obdh.log(sc.t, "INFO", "GROUND", f"activity override: {arg}")
            return "ok"
        if cmd == "burn":
            dv = float(arg or 0.05)
            dur = sc.prop.plan_burn(dv, sc.t)
            sc.obdh.log(sc.t, "INFO", "GROUND", f"burn commanded: {dv * 100:.1f} cm/s ({dur:.0f} s)")
            return "ok"
        if cmd == "reset":
            self.reset(arg or self.scenario)
            return "reset"
        raise ValueError(f"unknown command {cmd}")
