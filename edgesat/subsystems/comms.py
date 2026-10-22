"""Communications subsystem: S-band downlink with adaptive coding & modulation.

The link budget is evaluated every step from the true geometry and attitude:
EIRP (PA power, attitude-dependent antenna gain, losses) - free-space loss
- atmospheric loss + ground-station G/T - k  ->  C/N0, and the highest
data rate whose Eb/N0 meets the requirement plus margin is selected.
"""
from __future__ import annotations

import numpy as np

from ..core.constants import BOLTZMANN_DBW, C_LIGHT, DEG
from ..core.orbit import geodetic_to_ecef, look_angles


class GroundStation:
    def __init__(self, d: dict):
        self.name = d["name"]
        self.lat = d["lat_deg"] * DEG
        self.lon = d["lon_deg"] * DEG
        self.alt = d.get("alt_m", 0.0)
        self.g_over_t = d["g_over_t_dbk"]
        self.ecef = geodetic_to_ecef(self.lat, self.lon, self.alt)


class Comms:
    def __init__(self, cfg: dict):
        c = cfg["comms"]
        self.cfg = c
        self.stations = [GroundStation(g) for g in c["ground_stations"]]
        self.bore = np.array(c["antenna_boresight_body"], float)
        self.rates = sorted(c["rates_bps"])
        self.pa_degradation_db = 0.0
        self.prev_range: float | None = None
        self.pass_bytes = 0.0
        self.total_bytes = 0.0
        self.passes: list[dict] = []
        self.in_pass = False
        self.tm: dict = {}
        self.visible_station: GroundStation | None = None
        self.transmitting = False

    def geometry(self, r_ecef: np.ndarray):
        best = None
        for gs in self.stations:
            el, az, rng = look_angles(r_ecef, gs.ecef, gs.lat, gs.lon)
            if best is None or el > best[1]:
                best = (gs, el, az, rng)
        return best

    def budget(self, el: float, rng: float, off_bore: float, g_over_t: float) -> dict:
        c = self.cfg
        cos_t = max(np.cos(off_bore), 1e-3)
        g_ant = c["antenna_peak_gain_dbi"] + 10.0 * c["antenna_pattern_n"] * np.log10(cos_t)
        g_ant = max(g_ant, -20.0)
        p_dbw = 10.0 * np.log10(c["tx_power_w"]) - self.pa_degradation_db
        eirp = p_dbw + g_ant - c["tx_losses_db"]
        fspl = 20.0 * np.log10(4.0 * np.pi * rng * c["frequency_hz"] / C_LIGHT)
        l_atm = c["atmo_losses_db"] / max(np.sin(el), 0.087)
        cn0 = eirp - fspl - l_atm - c["polarization_loss_db"] + g_over_t - BOLTZMANN_DBW
        need = c["required_ebn0_db"] + c["link_margin_db"]
        rate = 0.0
        for r in self.rates:
            if cn0 - 10.0 * np.log10(r) >= need:
                rate = r
        ebn0 = cn0 - 10.0 * np.log10(rate) if rate > 0 else cn0 - 10.0 * np.log10(self.rates[0])
        margin = ebn0 - c["required_ebn0_db"]
        return {"antenna_gain_dbi": float(g_ant), "eirp_dbw": float(eirp), "fspl_db": float(fspl),
                "atm_loss_db": float(l_atm), "cn0_dbhz": float(cn0), "ebn0_db": float(ebn0),
                "margin_db": float(margin), "rate_bps": float(rate)}

    def step(self, t: float, dt: float, r_ecef: np.ndarray, r_eci: np.ndarray, A_bi: np.ndarray,
             R_ie: np.ndarray, tx_enabled: bool, has_data: bool) -> dict:
        """R_ie: ECEF->ECI rotation. Returns telemetry; capacity in bytes available this step
        is in tm['capacity_bytes']."""
        c = self.cfg
        gs, el, az, rng = self.geometry(r_ecef)
        visible = el >= c["min_elevation_deg"] * DEG
        gs_eci = R_ie @ gs.ecef
        los_b = A_bi @ ((gs_eci - r_eci) / rng)
        off = float(np.arccos(np.clip(los_b @ self.bore, -1.0, 1.0)))
        lb = self.budget(el, rng, off, gs.g_over_t)
        rr = 0.0 if self.prev_range is None else (rng - self.prev_range) / dt
        self.prev_range = rng
        doppler = -rr / C_LIGHT * c["frequency_hz"]
        link_up = visible and lb["rate_bps"] > 0.0
        self.transmitting = link_up and tx_enabled and has_data
        cap = lb["rate_bps"] * c["protocol_efficiency"] * dt / 8.0 if self.transmitting else 0.0
        if visible and not self.in_pass:
            self.in_pass = True
            self.pass_bytes = 0.0
            self.passes.append({"station": gs.name, "aos": t, "los": None, "max_el": 0.0, "bytes": 0.0})
        if self.in_pass:
            p = self.passes[-1]
            p["max_el"] = max(p["max_el"], float(np.degrees(el)))
            if not visible:
                self.in_pass = False
                p["los"] = t
        self.visible_station = gs if visible else None
        self.tm = {
            "station": gs.name, "visible": bool(visible), "link_up": bool(link_up),
            "transmitting": bool(self.transmitting),
            "elevation_deg": float(np.degrees(el)), "azimuth_deg": float(np.degrees(az) % 360),
            "range_km": rng / 1000.0, "range_rate_km_s": rr / 1000.0, "doppler_khz": doppler / 1000.0,
            "off_boresight_deg": float(np.degrees(off)), **lb,
            "pa_degradation_db": self.pa_degradation_db,
            "capacity_bytes": cap, "pass_bytes": self.pass_bytes, "total_bytes": self.total_bytes,
            "gs_eci": gs_eci,
        }
        return self.tm

    def account(self, sent: float) -> None:
        self.pass_bytes += sent
        self.total_bytes += sent
        if self.passes and self.in_pass:
            self.passes[-1]["bytes"] = self.pass_bytes
