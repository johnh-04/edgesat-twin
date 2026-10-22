"""The spacecraft: environment evaluation and the coupled step of all subsystems."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .core.attitude import dcm_from_quat, qmul, qnormalize, quat_from_dcm, quat_from_rotvec, triad
from .core.constants import AU, DEG, MU_EARTH, R_EARTH
from .core.environment import (CloudField, FireScenario, LandMask, atmospheric_density,
                               illumination_fraction, magnetic_field_ecef, sun_position_eci,
                               sun_right_ascension)
from .core.orbit import (OrbitPropagator, coe_to_rv, ecef_to_geodetic, eci_to_ecef_matrix,
                         julian_date, ltan_raan, predict_states, relative_wind, rv_to_coe,
                         sso_inclination)
from .subsystems.adcs import ADCS
from .subsystems.comms import Comms
from .subsystems.eps import EPS
from .subsystems.obdh import OBDH
from .subsystems.payload import Payload
from .subsystems.propulsion import Propulsion
from .subsystems.tcs import TCS

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Env:
    t: float
    jd: float
    r: np.ndarray
    v: np.ndarray
    v_rel: np.ndarray
    R_ei: np.ndarray       # ECI -> ECEF
    r_ecef: np.ndarray
    lat: float
    lon: float
    alt: float
    sun_u: np.ndarray
    sun_dist_au: float
    illum: float
    B_eci: np.ndarray
    rho: float
    sun_elev_subsat: float  # rad
    sun_ecef_u: np.ndarray


class Spacecraft:
    def __init__(self, cfg: dict, scenario: str = "nominal", seed: int | None = None, root: Path = ROOT):
        self.cfg = cfg
        m = cfg["mission"]
        self.dt = float(m["dt"])
        self.seed = m["seed"] if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.epoch = datetime.fromisoformat(m["epoch_utc"])
        self.jd0 = julian_date(self.epoch)
        self.t = 0.0
        self.scenario = scenario

        o = cfg["orbit"]
        a = R_EARTH + o["altitude_km"] * 1000.0
        inc = sso_inclination(a, o["eccentricity"])
        raan = ltan_raan(self.jd0, o["ltan_hours"], sun_right_ascension(self.jd0))
        r0, v0 = coe_to_rv(a, o["eccentricity"], inc, raan, o["arg_perigee_deg"] * DEG, o["true_anomaly_deg"] * DEG)
        # The configured altitude is a *mean* altitude: J2 short-period terms make the osculating
        # semi-major axis differ from the orbit-averaged one by ~10 km, so correct the initial state.
        for _ in range(2):
            period = 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
            _, rs, vs = predict_states(r0, v0, period, 10.0)
            a_osc = 1.0 / (2.0 / np.linalg.norm(rs, axis=1) - np.einsum("ij,ij->i", vs, vs) / MU_EARTH)
            a_mean = float(a_osc[:-1].mean())
            a_target = R_EARTH + o["altitude_km"] * 1000.0
            a0 = 1.0 / (2.0 / np.linalg.norm(r0) - (v0 @ v0) / MU_EARTH)
            a_new = a0 + (a_target - a_mean)
            inc = sso_inclination(a_target, o["eccentricity"])
            r0, v0 = coe_to_rv(a_new, o["eccentricity"], inc, raan, o["arg_perigee_deg"] * DEG,
                               o["true_anomaly_deg"] * DEG)
            a = a_new
        self.orbit = OrbitPropagator(r0, v0)
        self.elements0 = {"a_km": (R_EARTH / 1000 + o["altitude_km"]), "e": o["eccentricity"], "i_deg": inc / DEG, "raan_deg": raan / DEG,
                          "ltan_h": o["ltan_hours"]}

        # world
        self.land = LandMask()
        self.clouds = CloudField(self.seed)
        self.fires = FireScenario(cfg["scenario"]["fire_zone"], self.land, self.seed)

        # subsystems
        self.env = self._environment()
        A0 = triad(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), self.env.sun_u,
                   -self.env.r / np.linalg.norm(self.env.r))
        if scenario == "leop":
            q0 = self.rng.normal(size=4)
            q0 /= np.linalg.norm(q0)
            w0 = self.rng.normal(size=3)
            w0 = w0 / np.linalg.norm(w0) * 5.0 * DEG
        else:
            q0 = quat_from_dcm(A0)
            w0 = self.rng.normal(size=3) * cfg["adcs"]["initial_rate_deg_s"] * DEG
        self.adcs = ADCS(cfg, self.rng, q0, w0)
        if scenario == "leop":
            self.adcs.mode = "DETUMBLE"
            # coarse initialisation (TRIAD from Sun + magnetometer) good to ~10 deg
            self.adcs.ekf.q = qnormalize(qmul(quat_from_rotvec(self.rng.normal(0, 6 * DEG, 3)), q0))
            self.adcs.ekf.P = np.diag([(10 * DEG) ** 2] * 3 + [(0.1 * DEG) ** 2] * 3)
        self.eps = EPS(cfg)
        self.tcs = TCS(cfg)
        self.comms = Comms(cfg)
        self.prop = Propulsion(cfg)
        self.payload = Payload(cfg, self.rng, self.land, self.clouds, self.fires, root)
        self.obdh = OBDH(cfg, scenario)
        self.loads: dict[str, float] = {}
        self.period = rv_to_coe(self.orbit.r, self.orbit.v)["period"]
        self.track_hist: list[tuple[float, float]] = []
        self.stats_energy = {"tx_wh": 0.0, "ai_wh": 0.0}

    # ------------------------------------------------------------------ environment
    def _environment(self) -> Env:
        t = self.t
        jd = self.jd0 + t / 86400.0
        r, v = self.orbit.r, self.orbit.v
        R_ei = eci_to_ecef_matrix(jd)
        r_ecef = R_ei @ r
        lat, lon, alt = ecef_to_geodetic(r_ecef)
        sun = sun_position_eci(jd)
        sd = float(np.linalg.norm(sun))
        sun_u = sun / sd
        illum = illumination_fraction(r, sun)
        B_eci = R_ei.T @ magnetic_field_ecef(r_ecef)
        rho = atmospheric_density(alt, self.cfg["orbit"]["density_scale"])
        sun_ecef_u = R_ei @ sun_u
        up = r_ecef / np.linalg.norm(r_ecef)
        sun_elev = float(np.arcsin(np.clip(up @ sun_ecef_u, -1, 1)))
        return Env(t, jd, r.copy(), v.copy(), relative_wind(r, v), R_ei, r_ecef, lat, lon, alt, sun_u,
                   sd / AU, illum, B_eci, rho, sun_elev, sun_ecef_u)

    def in_aoi_now(self) -> bool:
        lat, lon = np.degrees(self.env.lat), np.degrees(self.env.lon)
        if np.degrees(self.env.sun_elev_subsat) < self.cfg["payload"]["min_sun_elevation_deg"]:
            return False
        return any(a["lat"][0] <= lat <= a["lat"][1] and a["lon"][0] <= lon <= a["lon"][1]
                   for a in self.cfg["payload"]["aoi"])

    # ------------------------------------------------------------------ main step
    def step(self) -> None:
        dt = self.dt
        self.env = env = self._environment()
        t = self.t

        # GS target for the ADCS (ECI now and one step ahead)
        gs = self.comms.stations[0] if self.comms.visible_station is None else self.comms.visible_station
        R_ie = env.R_ei.T
        self.adcs.gs_eci = R_ie @ gs.ecef
        self.adcs.gs_eci_next = eci_to_ecef_matrix(env.jd + dt / 86400.0).T @ gs.ecef

        plan = self.obdh.plan(t, self)

        # propulsion
        aligned = self.adcs.mode == "DELTA_V" and self.adcs.tm.get("pointing_error_deg", 99) < 3.0
        f_body, tq, valve = self.prop.step(dt, aligned, self.obdh.mode == "NOMINAL")
        # ADCS
        self.adcs.step(t, dt, env, tq, enabled=plan["adcs_enabled"])
        A = dcm_from_quat(self.adcs.q)
        # orbit: drag + thrust
        mass = self.prop.mass
        vr = env.v_rel
        vrn = np.linalg.norm(vr)
        a_drag = -0.5 * env.rho * vrn * self.cfg["spacecraft"]["drag_cd"] * self.adcs.area_proj / mass * vr
        a_thr = A.T @ f_body / mass
        self.orbit.step(dt, a_drag + a_thr)

        # payload
        ctx = {"lat": np.degrees(env.lat), "lon": np.degrees(env.lon), "alt_m": env.alt,
               "sun_elev_deg": np.degrees(env.sun_elev_subsat),
               "pointing_error_deg": self.adcs.tm["pointing_error_deg"],
               "rate_err_rad_s": self.adcs.tm["rate_err_rad_s"]}
        self.payload.step(t, dt, plan["imaging"], ctx)

        # comms + downlink (both pipelines get the same capacity)
        sw = self.eps.switches
        has = self.payload.store.has_data() or self.payload.shadow.has_data()
        ctm = self.comms.step(t, dt, env.r_ecef, env.r, A, R_ie, sw["transmitter"], has)
        cap = ctm["capacity_bytes"]
        if cap > 0:
            sent = self.payload.store.transmit(cap, t, {})
            self.payload.shadow.transmit(cap, t, {})
            self.comms.account(sent)

        # power
        L = self.cfg["eps"]["loads"]
        heaters = self.tcs.thermostats()
        hb = (heaters["battery_heater"] or self.tcs.battery_heater_stuck) and not self.obdh.heater_inhibit
        loads = {"obc": L["obc"], **self.adcs.power, "radio_rx": L["radio_rx"],
                 "transmitter": L["radio_tx_dc"] if self.comms.transmitting else 0.0,
                 "camera": (L["camera_imaging"] if self.payload.imaging else L["camera_standby"]) if sw["payload"] else 0.0,
                 "ai_unit": self.payload.ai_power(L),
                 "battery_heater": L["battery_heater"] if hb else 0.0,
                 "payload_heater": L["payload_heater"] if heaters["payload_heater"] else 0.0,
                 "propulsion": L["prop_valve"] * valve}
        if self.obdh.mode == "REBOOT":
            loads["obc"] = 0.3
        sw["battery_heater"], sw["payload_heater"] = hb, heaters["payload_heater"]
        self.loads = loads
        sun_b = A @ env.sun_u
        temps = self.tcs.temps_c()
        self.eps.step(dt, sun_b, env.illum, env.sun_dist_au, loads, temps)
        self.stats_energy["tx_wh"] += loads["transmitter"] * dt / 3600
        self.stats_energy["ai_wh"] += loads["ai_unit"] * dt / 3600
        self.payload.ai_energy_j += loads["ai_unit"] * dt

        # thermal (TX: RF power leaves the spacecraft)
        heat = dict(loads)
        if self.comms.transmitting:
            heat["transmitter"] = max(loads["transmitter"] - self.cfg["comms"]["tx_power_w"], 0.0)
        heat["battery_heater"] = heat["battery_heater"] + self.eps.battery_heat
        nadir_b = A @ (-env.r / np.linalg.norm(env.r))
        self.tcs.step(dt, sun_b, nadir_b, float(np.linalg.norm(env.r)), env.illum, env.sun_dist_au,
                      float(np.sin(env.sun_elev_subsat)), heat, self.eps.gen_by_node)

        # data handling, FDIR, orbit maintenance
        self.obdh.housekeeping(t, dt, self)
        self.obdh.track_sma(t, self.orbit.r, self.orbit.v, self.period, self)
        self.obdh.fdir(t, dt, self)
        if int(t) % 20 == 0:
            self.track_hist.append((round(np.degrees(env.lat), 2), round(np.degrees(env.lon), 2)))
            if len(self.track_hist) > 300:
                self.track_hist.pop(0)
        self.t = t + dt

    # ------------------------------------------------------------------ faults
    FAULTS = {
        "wheel_x_seize": "Reaction wheel X seizes (momentum dumped into the body)",
        "wheel_y_seize": "Reaction wheel Y seizes",
        "wheel_z_seize": "Reaction wheel Z seizes",
        "star_tracker_blind": "Star tracker blinded for 20 min (stray light)",
        "star_tracker_fail": "Star tracker permanent failure",
        "gyro_bias_jump": "Gyro bias step of 0.05 deg/s on all axes",
        "solar_string_loss": "Wing -X loses half of its strings",
        "battery_cell_loss": "One of four parallel battery strings opens",
        "transponder_degradation": "S-band PA output drops by 6 dB",
        "battery_heater_stuck": "Battery heater switch welded closed",
        "obc_seu": "Single-event upset in the OBC (watchdog reset)",
        "thruster_leak": "Thruster valve leak",
        "ai_hang": "AI processor hangs (no heartbeat)",
    }

    def inject(self, name: str) -> str:
        t = self.t
        o = self.obdh
        if name.startswith("wheel_") and name.endswith("_seize"):
            i = "xyz".index(name[6])
            self.adcs.seize_wheel(i)
        elif name == "star_tracker_blind":
            self.adcs.faults.star_tracker_blind_until = t + 1200.0
        elif name == "star_tracker_fail":
            self.adcs.faults.star_tracker_failed = True
        elif name == "gyro_bias_jump":
            self.adcs.gyro_bias = self.adcs.gyro_bias + 0.05 * DEG
        elif name == "solar_string_loss":
            self.eps.panels[0]["health"] = 0.5
        elif name == "battery_cell_loss":
            self.eps.capacity_ah = self.eps.nominal_capacity_ah * 0.75
        elif name == "transponder_degradation":
            self.comms.pa_degradation_db = 6.0
        elif name == "battery_heater_stuck":
            self.tcs.battery_heater_stuck = True
        elif name == "obc_seu":
            o.seu(t)
        elif name == "thruster_leak":
            self.prop.leaking = True
        elif name == "ai_hang":
            self.payload.ai_hang_until = t + 1e9
        else:
            raise ValueError(f"unknown fault {name}")
        o.log(t, "WARN", "TEST", f"fault injected: {self.FAULTS[name]}")
        return self.FAULTS[name]

    def repair_all(self) -> None:
        """Simulator-only: restore all hardware (not something a real spacecraft can do)."""
        a = self.adcs
        a.faults.wheel_seized = [False] * 3
        a.faults.wheel_isolated = [False] * 3
        a.faults.star_tracker_failed = False
        a.faults.star_tracker_blind_until = -1
        a.h_expected = a.h_w.copy()
        a.wheel_mismatch_s[:] = 0
        for p in self.eps.panels:
            p["health"] = 1.0
        self.eps.capacity_ah = self.eps.nominal_capacity_ah
        self.comms.pa_degradation_db = 0.0
        self.tcs.battery_heater_stuck = False
        self.obdh.heater_inhibit = False
        self.prop.leaking = False
        self.prop.latch_closed = False
        self.payload.ai_hang_until = -1
        self.obdh.log(self.t, "INFO", "TEST", "all hardware faults cleared (simulator command)")
