"""Electrical Power Subsystem.

Solar panels with temperature-dependent efficiency and per-panel string
health, MPPT with charge-current and full-battery curtailment, a Li-ion
battery pack (OCV(SoC) curve, internal resistance, coulombic efficiency,
cold-charge inhibit) and a switched load bus with latching current limiters
(LCLs) commanded by the OBDH.
"""
from __future__ import annotations

import numpy as np

from ..core.constants import SOLAR_CONSTANT

# Li-ion cell open-circuit voltage vs state of charge
_SOC = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00])
_OCV = np.array([3.00, 3.30, 3.45, 3.56, 3.63, 3.72, 3.86, 4.03, 4.18])

SWITCHED = ("payload", "ai_unit", "transmitter", "battery_heater", "payload_heater", "propulsion",
            "star_tracker", "gnss")


class EPS:
    def __init__(self, cfg: dict):
        e = cfg["eps"]
        self.cfg = e
        self.panels = [dict(p, normal=np.array(p["normal"], float), health=1.0) for p in e["panels"]]
        self.n_series = e["battery_series"]
        self.n_par = e["battery_parallel"]
        self.cell_ah = e["cell_capacity_Ah"]
        self.capacity_ah = self.n_par * self.cell_ah
        self.nominal_capacity_ah = self.capacity_ah
        self.soc = e["initial_soc"]
        self.R = e["battery_resistance_ohm"]
        self.switches = {k: True for k in SWITCHED}
        self.switches.update({"payload": False, "ai_unit": False, "transmitter": False,
                              "propulsion": False, "battery_heater": False, "payload_heater": False})
        self.energy_in_wh = 0.0
        self.energy_out_wh = 0.0
        self.energy_by_load_wh: dict[str, float] = {}
        self.tm: dict = {}
        self.gen_by_node: dict[str, float] = {}
        self.battery_heat = 0.0

    def ocv(self) -> float:
        return float(np.interp(self.soc, _SOC, _OCV)) * self.n_series

    @property
    def capacity_wh(self) -> float:
        return self.capacity_ah * 3.7 * self.n_series

    def step(self, dt: float, sun_b: np.ndarray, illum: float, sun_dist_au: float,
             loads: dict[str, float], temps: dict[str, float]) -> dict:
        e = self.cfg
        flux = SOLAR_CONSTANT / (sun_dist_au ** 2) * illum
        p_avail = 0.0
        per_panel = []
        self.gen_by_node = {}
        for p in self.panels:
            c = float(p["normal"] @ sun_b)
            T = temps.get(p["node"], 28.0)
            eff = e["cell_efficiency"] * (1.0 + e["temp_coeff_per_K"] * (T - 28.0))
            pw = max(c, 0.0) * flux * p["area"] * eff * e["packing_factor"] * e["degradation"] * p["health"]
            per_panel.append(pw)
            p_avail += pw

        p_load = float(sum(loads.values()))
        ocv = self.ocv()
        t_bat = temps.get("BATTERY", 20.0)
        charge_allowed = t_bat > 0.0 and self.soc < 0.995
        p_charge_max = e["max_charge_current_A"] * ocv if charge_allowed else 0.0
        p_used = min(p_avail, p_load + p_charge_max)
        p_bat = p_used - p_load              # >0 charging
        # V*I = P with V = OCV + I R  ->  R I^2 + OCV I - P = 0
        disc = ocv * ocv + 4.0 * self.R * p_bat
        i_bat = (-ocv + np.sqrt(max(disc, 0.0))) / (2.0 * self.R)
        v_bus = ocv + i_bat * self.R
        eta = e["charge_efficiency"] if i_bat > 0 else 1.0
        self.soc = float(np.clip(self.soc + eta * i_bat * dt / 3600.0 / self.capacity_ah, 0.0, 1.0))
        self.battery_heat = i_bat * i_bat * self.R
        # electrical power extracted from each thermal node (removed from absorbed heat)
        scale = p_used / p_avail if p_avail > 1e-9 else 0.0
        for p, pw in zip(self.panels, per_panel):
            self.gen_by_node[p["node"]] = self.gen_by_node.get(p["node"], 0.0) + pw * scale

        hours = dt / 3600.0
        self.energy_in_wh += p_used * hours
        self.energy_out_wh += p_load * hours
        for k, v in loads.items():
            self.energy_by_load_wh[k] = self.energy_by_load_wh.get(k, 0.0) + v * hours

        self.tm = {
            "soc": self.soc, "soc_pct": 100.0 * self.soc,
            "solar_available_w": p_avail, "solar_used_w": p_used,
            "curtailed_w": p_avail - p_used, "load_w": p_load,
            "battery_power_w": p_bat, "battery_current_a": float(i_bat),
            "bus_voltage_v": float(v_bus), "ocv_v": ocv,
            "capacity_wh": self.capacity_wh,
            "panel_w": per_panel, "panel_names": [p["name"] for p in self.panels],
            "panel_health": [p["health"] for p in self.panels],
            "loads_w": {k: round(v, 3) for k, v in loads.items()},
            "switches": dict(self.switches),
            "charge_inhibit": not charge_allowed and self.soc < 0.995,
            "energy_in_wh": self.energy_in_wh, "energy_out_wh": self.energy_out_wh,
        }
        return self.tm
