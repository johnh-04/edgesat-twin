"""Thermal Control Subsystem: lumped-parameter thermal network.

Each node i obeys  C_i dT_i/dt = Q_ext,i + Q_int,i - P_elec,i + sum_j G_ij (T_j - T_i) - Q_rad,i

* Q_ext: direct solar, Earth albedo and Earth infrared on every external
  surface (orientation from the true attitude), radiation to deep space.
* Q_int: dissipation of the electrical loads hosted by the node (the AI unit
  heats the avionics node, the transmitter's PA losses heat the comms node).
* P_elec: electrical power extracted by the solar cells.
* Thermostatic heaters on battery and payload, with a stuck-on failure mode.
Integration: explicit Euler (time constants >> dt).
"""
from __future__ import annotations

import numpy as np

from ..core.constants import KELVIN, R_EARTH, SIGMA_SB, SOLAR_CONSTANT

# where each electrical load dissipates its heat
LOAD_NODE = {
    "obc": "AVIONICS", "adcs_electronics": "AVIONICS", "reaction_wheels": "AVIONICS",
    "magnetorquers": "STRUCTURE", "star_tracker": "STRUCTURE", "gyro": "AVIONICS",
    "gnss": "AVIONICS", "radio_rx": "COMMS", "transmitter": "COMMS", "camera": "PAYLOAD",
    "ai_unit": "AVIONICS", "battery_heater": "BATTERY", "payload_heater": "PAYLOAD",
    "propulsion": "STRUCTURE",
}


class TCS:
    def __init__(self, cfg: dict):
        c = cfg["tcs"]
        self.cfg = c
        self.names = [n["name"] for n in c["nodes"]]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.C = np.array([n["C"] for n in c["nodes"]])
        self.T = np.array([n["T0"] for n in c["nodes"]]) + KELVIN
        self.limits = np.array([[n["min"], n["max"]] for n in c["nodes"]])
        n = len(self.names)
        self.G = np.zeros((n, n))
        for a, b, g in c["links"]:
            i, j = self.idx[a], self.idx[b]
            self.G[i, j] += g
            self.G[j, i] += g
        self.L = self.G - np.diag(self.G.sum(axis=1))  # conductance Laplacian
        self.surf = c["surfaces"]
        self.s_normal = np.array([s["normal"] for s in self.surf], float)
        self.s_area = np.array([s["area"] for s in self.surf])
        self.s_alpha = np.array([s["alpha"] for s in self.surf])
        self.s_eps = np.array([s["eps"] for s in self.surf])
        self.s_node = np.array([self.idx[s["node"]] for s in self.surf])
        self.battery_heater_stuck = False
        self.heater_cmd = {"battery_heater": False, "payload_heater": False}
        self.tm: dict = {}

    def temps_c(self) -> dict[str, float]:
        return {n: float(self.T[i] - KELVIN) for i, n in enumerate(self.names)}

    def thermostats(self) -> dict[str, bool]:
        """Bang-bang heater logic, evaluated by the OBDH."""
        c, t = self.cfg, self.temps_c()
        hb = self.heater_cmd["battery_heater"]
        if t["BATTERY"] < c["battery_heater_on_C"]:
            hb = True
        elif t["BATTERY"] > c["battery_heater_off_C"]:
            hb = False
        hp = self.heater_cmd["payload_heater"]
        if t["PAYLOAD"] < c["payload_heater_on_C"]:
            hp = True
        elif t["PAYLOAD"] > c["payload_heater_off_C"]:
            hp = False
        self.heater_cmd = {"battery_heater": hb, "payload_heater": hp}
        return dict(self.heater_cmd)

    def step(self, dt: float, sun_b: np.ndarray, nadir_b: np.ndarray, r_norm: float, illum: float,
             sun_dist_au: float, cos_sun_zenith_subsat: float, internal: dict[str, float],
             elec_extracted: dict[str, float]) -> dict:
        c = self.cfg
        S = SOLAR_CONSTANT / sun_dist_au ** 2
        cos_s = self.s_normal @ sun_b
        cos_n = self.s_normal @ nadir_b
        vf = 0.5 * (1.0 + cos_n) * (R_EARTH / r_norm) ** 2       # view factor to Earth (approx.)
        q_sun = self.s_alpha * self.s_area * S * illum * np.clip(cos_s, 0.0, None)
        q_alb = self.s_alpha * self.s_area * S * c["albedo"] * max(cos_sun_zenith_subsat, 0.0) * vf
        q_ir = self.s_eps * self.s_area * c["earth_ir_W_m2"] * vf
        q_rad = self.s_eps * self.s_area * SIGMA_SB * self.T[self.s_node] ** 4
        n = len(self.names)
        q_ext = np.zeros(n)
        np.add.at(q_ext, self.s_node, q_sun + q_alb + q_ir - q_rad)
        q_int = np.zeros(n)
        for load, p in internal.items():
            node = LOAD_NODE.get(load)
            if node is not None:
                q_int[self.idx[node]] += p
        for node, p in elec_extracted.items():
            q_int[self.idx[node]] -= p
        q_cond = self.L @ self.T
        dT = (q_ext + q_int + q_cond) / self.C
        self.T = self.T + dT * dt
        tc = self.T - KELVIN
        status = []
        for i in range(n):
            lo, hi = self.limits[i]
            status.append("RED" if (tc[i] < lo or tc[i] > hi) else
                          "YELLOW" if (tc[i] < lo + 5 or tc[i] > hi - 5) else "GREEN")
        flows = {
            "solar_w": float(q_sun.sum()), "albedo_w": float(q_alb.sum()), "earth_ir_w": float(q_ir.sum()),
            "radiated_w": float(q_rad.sum()), "internal_w": float(sum(internal.values())),
        }
        self.tm = {"nodes": self.names, "temps_c": tc.tolist(), "limits_c": self.limits.tolist(),
                   "status": status, "heaters": dict(self.heater_cmd),
                   "heater_stuck": self.battery_heater_stuck, "flows": flows,
                   "node_q_ext_w": q_ext.tolist(), "node_q_int_w": q_int.tolist()}
        return self.tm
