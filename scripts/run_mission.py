"""Headless mission run: Edge AI vs a conventional store-everything satellite.

Two independent simulations with the same seed (same orbit, weather, wildfires,
ground station) are flown for N days; the only difference is the on-board AI.

    python scripts/run_mission.py --days 3        # ~10 min on 2 cores

Writes docs/results.json, docs/results.md and the figures in docs/images/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edgesat.ai.scenes import CLASSES  # noqa: E402
from edgesat.sim import Simulation  # noqa: E402


def fly(args):
    ai, days, seed = args
    sim = Simulation(ai_enabled=ai, seed=seed)
    sc = sim.sc
    hourly, fine = [], []
    steps = int(days * 86400 / sc.dt)
    t0 = time.time()
    for k in range(steps):
        sc.step()
        t = sc.t
        if int(t) % 30 == 0:
            a, e, c = sc.adcs.tm, sc.eps.tm, sc.comms.tm
            fine.append([t, e["soc_pct"], e["solar_used_w"], e["load_w"], a["pointing_error_deg"],
                         a["knowledge_error_arcsec"], *sc.tcs.tm["temps_c"],
                         c["rate_bps"] / 1e6 if c["transmitting"] else 0.0, c["elevation_deg"],
                         sc.env.alt / 1000.0, {"IMAGING": 1, "DOWNLINK": 2, "ORBIT_MAINT": 3}.get(sc.obdh.activity, 0)])
        if int(t) % 3600 == 0:
            d = sc.payload.store.stats
            hourly.append([t, d.useful_bytes / 1e9, d.frames_delivered, d.value_delivered,
                           sc.payload.store.used / sc.payload.store.capacity * 100,
                           len(d.fire_alert_ground_t), len(d.fire_image_ground_t), sc.payload.acq_count])
            if int(t) % (6 * 3600) == 0:
                print(f"[{'AI ' if ai else 'BASE'}] day {t / 86400:4.2f}  useful {d.useful_bytes / 1e9:6.2f} GB  "
                      f"wall {time.time() - t0:5.0f} s", flush=True)
    p = sc.payload
    st = p.store.stats
    fires = sc.fires.fires
    imaged = {f.fid: f.first_acq_t for f in fires if f.first_acq_t is not None}
    alert_lat = [(st.fire_alert_ground_t[i] - imaged[i]) / 3600 for i in st.fire_alert_ground_t if i in imaged]
    image_lat = [(st.fire_image_ground_t[i] - imaged[i]) / 3600 for i in st.fire_image_ground_t if i in imaged]
    first_info = [min(st.fire_alert_ground_t.get(i, np.inf), st.fire_image_ground_t.get(i, np.inf)) for i in imaged]
    info_lat = [(x - imaged[i]) / 3600 for x, i in zip(first_info, imaged) if np.isfinite(x)]
    energy = sc.eps.energy_by_load_wh
    res = {
        "ai": ai, "days": days, "seed": seed,
        "frames_acquired": p.acq_count, "truth_counts": dict(zip(CLASSES, p.truth_counts)),
        "frames_smeared": p.blurred, "decisions": p.decisions, "confusion": p.confusion.tolist(),
        "downlinked_gb": sc.comms.total_bytes / 1e9, "bytes_by_kind_gb": {k: v / 1e9 for k, v in st.bytes_by_kind.items()},
        "frames_delivered": st.frames_delivered, "useful_frames_delivered": st.useful_frames_delivered,
        "useful_gb": st.useful_bytes / 1e9, "value_delivered": st.value_delivered,
        "median_frame_age_h": float(np.median(st.frame_age_s) / 3600) if st.frame_age_s else None,
        "dropped_memory_full": st.dropped_memory_full, "evicted": st.evicted,
        "memory_fill_pct": p.store.used / p.store.capacity * 100,
        "fires_imaged": len(imaged), "fires_alerted_on_ground": len(st.fire_alert_ground_t),
        "false_alerts_delivered": st.false_alerts_delivered,
        "fires_with_imagery_on_ground": len(st.fire_image_ground_t),
        "fires_with_any_info_on_ground": len(info_lat),
        "alert_latency_h": alert_lat, "image_latency_h": image_lat, "info_latency_h": info_lat,
        "energy_wh": {k: round(v, 2) for k, v in energy.items()}, "energy_total_wh": sum(energy.values()),
        "passes": len(sc.comms.passes), "burns": len(sc.prop.burns), "dv_used_m_s": sc.prop.dv_used,
        "min_soc_pct": float(min(r[1] for r in fine)), "events": list(sc.obdh.events)[:60],
        "hourly": hourly, "fine": fine if ai else [], "wall_s": time.time() - t0,
    }
    return res


def plots(ai, base, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager  # noqa: F401

    plt.rcParams.update({
        "figure.facecolor": "#0f1828", "axes.facecolor": "#15213a", "axes.edgecolor": "#2a3b5c",
        "axes.labelcolor": "#c9d3e4", "xtick.color": "#9aa9c2", "ytick.color": "#9aa9c2",
        "text.color": "#e4eaf4", "grid.color": "#2a3b5c", "axes.grid": True, "grid.linewidth": 0.6,
        "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    })
    KAP, BASE, CELL, CYAN, RED, GREEN = "#d4a84b", "#8c9bb3", "#5b8def", "#4fc3d9", "#ec4f55", "#3dbb8c"
    out.mkdir(parents=True, exist_ok=True)
    ha, hb = np.array(ai["hourly"]), np.array(base["hourly"])

    # 1 -- the headline: useful data on the ground and memory
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    ax[0].plot(ha[:, 0] / 86400, ha[:, 1], color=KAP, lw=2.2, label="Edge AI on board")
    ax[0].plot(hb[:, 0] / 86400, hb[:, 1], color=BASE, lw=1.8, label="Store everything")
    ax[0].set(title="Useful imagery on the ground", xlabel="mission day", ylabel="GB")
    ax[0].legend()
    ua = ai["useful_frames_delivered"] / max(ai["frames_delivered"], 1) * 100
    ub = base["useful_frames_delivered"] / max(base["frames_delivered"], 1) * 100
    ax[1].bar(["Edge AI", "Store\neverything"], [ua, ub], color=[KAP, BASE], width=0.55)
    for i, v in enumerate([ua, ub]):
        ax[1].text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=12)
    ax[1].set(title="Share of downlinked frames that are useful", ylim=(0, 110), ylabel="%")
    ax[2].plot(ha[:, 0] / 86400, ha[:, 4], color=KAP, lw=2.2, label="Edge AI on board")
    ax[2].plot(hb[:, 0] / 86400, hb[:, 4], color=BASE, lw=1.8, label="Store everything")
    ax[2].set(title="Mass memory (16 GB)", xlabel="mission day", ylabel="% full", ylim=(0, 105))
    fig.tight_layout()
    fig.savefig(out / "ai_vs_baseline.png", dpi=140)
    plt.close(fig)

    # 2 -- wildfire latency
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
    la, lb = ai["info_latency_h"], base["info_latency_h"]
    bins = np.linspace(0, max([*la, *lb, 24]), 25)
    ax[0].hist(la, bins=bins, color=KAP, alpha=0.9, label=f"Edge AI alert (n={len(la)})")
    ax[0].hist(lb, bins=bins, color=BASE, alpha=0.7, label=f"Store everything (n={len(lb)})")
    ax[0].set(title="Wildfire: imaged → information on the ground", xlabel="hours", ylabel="fires")
    ax[0].legend()
    cats = ["imaged", "information\non ground"]
    ax[1].bar(np.arange(2) - 0.18, [ai["fires_imaged"], ai["fires_with_any_info_on_ground"]], 0.36, color=KAP, label="Edge AI")
    ax[1].bar(np.arange(2) + 0.18, [base["fires_imaged"], base["fires_with_any_info_on_ground"]], 0.36, color=BASE, label="Store everything")
    ax[1].set_xticks(range(2), cats)
    ax[1].set(title="Wildfires", ylabel="count", ylim=(0, 1.35 * max(ai["fires_imaged"], 1)))
    ax[1].legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "wildfire_latency.png", dpi=140)
    plt.close(fig)

    # 3 -- one day in the life of every subsystem
    f = np.array(ai["fine"])
    m = f[:, 0] <= 86400
    f = f[m]
    th = f[:, 0] / 3600
    fig, ax = plt.subplots(5, 1, figsize=(13, 11), sharex=True)
    act = f[:, -1]
    for a in ax:
        for val, col in ((1, KAP), (2, CYAN), (3, RED)):
            on = act == val
            a.fill_between(th, 0, 1, where=on, transform=a.get_xaxis_transform(), color=col, alpha=0.22, lw=0)
    ax[0].plot(th, f[:, 1], color=GREEN, lw=1.4)
    ax[0].set(ylabel="SoC %", title="One day of EDGESAT-1 (shaded: imaging gold, downlink cyan, burn red)")
    ax[1].plot(th, f[:, 2], color=KAP, lw=1.0, label="solar used")
    ax[1].plot(th, f[:, 3], color=CELL, lw=1.0, label="load")
    ax[1].set(ylabel="W")
    ax[1].legend(loc="upper right", ncol=2)
    names = ["structure", "solar array", "battery", "avionics", "payload", "comms"]
    cols = [BASE, KAP, GREEN, RED, CELL, "#a78bfa"]
    for i, (n, c) in enumerate(zip(names, cols)):
        ax[2].plot(th, f[:, 6 + i], color=c, lw=1.0, label=n)
    ax[2].set(ylabel="°C")
    ax[2].legend(loc="upper right", ncol=6, fontsize=9)
    ax[3].semilogy(th, np.maximum(f[:, 4], 1e-3), color=KAP, lw=0.9, label="pointing error [deg]")
    ax[3].semilogy(th, np.maximum(f[:, 5], 1e-1) / 3600, color=CELL, lw=0.9, label="knowledge error [deg]")
    ax[3].set(ylabel="deg")
    ax[3].legend(loc="upper right", ncol=2)
    ax[4].plot(th, f[:, 12], color=CYAN, lw=1.2)
    ax[4].set(ylabel="downlink Mbit/s", xlabel="hours since epoch (2027-07-15 04:00 UTC)")
    fig.tight_layout()
    fig.savefig(out / "day_in_the_life.png", dpi=130)
    plt.close(fig)

    # 4 -- confusion matrix in flight
    cm = np.array(ai["confusion"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="cividis", vmin=0, vmax=1)
    lab = ["cloud", "water", "land", "fire"]
    ax.set_xticks(range(4), lab)
    ax.set_yticks(range(4), lab)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] < cm.max() / 2 else "black")
    ax.set(xlabel="on-board prediction", ylabel="truth", title="In-flight confusion matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion_in_flight.png", dpi=140)
    plt.close(fig)


def write_markdown(ai, base, path: Path):
    def med(x):
        return f"{np.median(x):.1f} h" if x else "n/a"

    ua = ai["useful_frames_delivered"] / max(ai["frames_delivered"], 1) * 100
    ub = base["useful_frames_delivered"] / max(base["frames_delivered"], 1) * 100
    rows = [
        ("Frames acquired", ai["frames_acquired"], base["frames_acquired"]),
        ("Data downlinked (same passes)", f"{ai['downlinked_gb']:.2f} GB", f"{base['downlinked_gb']:.2f} GB"),
        ("Full frames delivered", ai["frames_delivered"], base["frames_delivered"]),
        ("Useful frames delivered", ai["useful_frames_delivered"], base["useful_frames_delivered"]),
        ("Useful share of delivered frames", f"{ua:.0f}%", f"{ub:.0f}%"),
        ("Useful imagery on the ground", f"{ai['useful_gb']:.2f} GB", f"{base['useful_gb']:.2f} GB"),
        ("Median age of a frame when it reaches the ground", f"{ai['median_frame_age_h']:.1f} h", f"{base['median_frame_age_h']:.1f} h"),
        ("Frames lost to full memory", ai["dropped_memory_full"], base["dropped_memory_full"]),
        ("Memory fill at end", f"{ai['memory_fill_pct']:.0f}%", f"{base['memory_fill_pct']:.0f}%"),
        ("Wildfires imaged", ai["fires_imaged"], base["fires_imaged"]),
        ("Wildfires with information on the ground", ai["fires_with_any_info_on_ground"], base["fires_with_any_info_on_ground"]),
        ("Median time, imaged → on the ground", med(ai["info_latency_h"]), med(base["info_latency_h"])),
        ("False fire alerts delivered", ai["false_alerts_delivered"], "n/a"),
        ("Energy used by the AI processor", f"{ai['energy_wh'].get('ai_unit', 0):.1f} Wh", "0 Wh"),
        ("Total energy consumed", f"{ai['energy_total_wh']:.0f} Wh", f"{base['energy_total_wh']:.0f} Wh"),
        ("Minimum battery state of charge", f"{ai['min_soc_pct']:.0f}%", f"{base['min_soc_pct']:.0f}%"),
        ("Ground-station passes / orbit-maintenance burns", f"{ai['passes']} / {ai['burns']}", f"{base['passes']} / {base['burns']}"),
    ]
    md = [f"# Mission results: {ai['days']} days, seed {ai['seed']}", "",
          "Generated by `python scripts/run_mission.py`. Both satellites fly the same orbit through the same",
          "weather and wildfires and downlink to the same ground station; only the on-board AI differs.", "",
          "| Metric | Edge AI on board | Store everything |", "|---|---:|---:|"]
    md += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    md += ["", "![AI vs baseline](images/ai_vs_baseline.png)", "", "![Wildfire latency](images/wildfire_latency.png)", "",
           "![A day in the life](images/day_in_the_life.png)", ""]
    path.write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot-only", action="store_true", help="redraw figures from the last raw run")
    a = ap.parse_args()
    raw = ROOT / "out" / "mission_raw.json"
    if a.plot_only:
        ai, base = json.loads(raw.read_text())
    else:
        with Pool(2) as pool:
            ai, base = pool.map(fly, [(True, a.days, a.seed), (False, a.days, a.seed)])
        raw.parent.mkdir(exist_ok=True)
        raw.write_text(json.dumps([ai, base], default=float))
    docs = ROOT / "docs"
    slim = {k: {kk: vv for kk, vv in r.items() if kk not in ("fine", "hourly")} for k, r in (("edge_ai", ai), ("baseline", base))}
    (docs / "results.json").write_text(json.dumps(slim, indent=1, default=float))
    plots(ai, base, docs / "images")
    write_markdown(ai, base, docs / "results.md")
    print((docs / "results.md").read_text())


if __name__ == "__main__":
    main()
