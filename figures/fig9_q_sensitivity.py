#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning-rate (Q) sensitivity — figures. Reads outputs/qsweep/qscale_<c>.csv
(q_sensitivity.py) and plots the trade-off vs EFFECTIVE MEMORY.

Effective memory is computed ANALYTICALLY (the empirical regression gain in the
sweep driver is unstable at low Q): treat each driver as a local-level filter,
signal-to-noise r = Q_total / R with Q_total = (Q_lam+Q_mag)*c and R = variance
of the log annual exposure on the historical record; steady-state EWMA smoothing
alpha = (-r + sqrt(r^2+4r))/2; memory = (1-alpha)/alpha years. Averaged over the
pilot reservoirs. Baseline c_Q=1 marked.

Two figures:
  q_sensitivity_mechanism.png -- policy SWITCHING rate vs memory (the churn a fast
    learner buys), per reservoir.
  q_sensitivity_outcome.png   -- cost reduction vs frozen (mean) and cost SPREAD
    (p90 of per-trajectory cost ratio) vs memory; the net resolution.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))
sys.path.insert(0, str(REPO / "bdps"))
import deploy as bd                                              # noqa: E402
from belief import fit_gamma_shape, fit_hyper                      # noqa: E402

OUT = REPO / "results"; SW = OUT / "qsweep"
GRID = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RES = ["NML", "EXC", "ORO"]
K = ["res", "clim", "lulc"]


def memory_years():
    """analytic EWMA-equivalent memory per c_Q, averaged over pilot reservoirs."""
    base = {}
    for res in RES:
        n, V = bd.historical_obs(res)
        a = fit_gamma_shape([(n, V)]); ql, qm, phi = fit_hyper([(n, V)], a, n_part=600)
        z = np.log(np.maximum(V.astype(float), 1.0))
        base[res] = (ql + qm, max(float(np.var(z)), 1e-6))
    mem = {}
    for c in GRID:
        ms = []
        for res in RES:
            Qt, R = base[res]; r = Qt * c / R
            alpha = (-r + np.sqrt(r * r + 4 * r)) / 2
            ms.append((1 - alpha) / alpha)
        mem[c] = float(np.mean(ms))
    return mem


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench = pd.read_csv(OUT / "bench_baselines.csv")[K + ["J_frozen2020"]]
    mem = memory_years()
    print("c_Q -> memory(yr):", {c: round(m, 1) for c, m in mem.items()})

    # gather per-(c) aggregates
    agg = {m: {} for m in ("bayes", "blend", "plugin")}
    spread = {m: {} for m in ("bayes", "blend", "plugin")}
    switch = {}
    per_res = {r: {"switch": {}, "bayes": {}} for r in RES}
    for c in GRID:
        f = SW / f"qscale_{c}.csv"
        if not f.exists():
            print("missing", f); continue
        d = pd.read_csv(f).merge(bench, on=K)
        switch[c] = d.switch_rate.mean()
        for m in agg:
            col = f"J_{m}"
            agg[m][c] = 100.0 * (d.J_frozen2020.sum() - d[col].sum()) / d.J_frozen2020.sum()
            ratio = d[col] / d.J_frozen2020.replace(0, np.nan)
            spread[m][c] = float(ratio.quantile(0.90))
        for r in RES:
            dr = d[d.res == r]
            per_res[r]["switch"][c] = dr.switch_rate.mean()
            per_res[r]["bayes"][c] = 100.0 * (dr.J_frozen2020.sum() - dr.J_bayes.sum()) / dr.J_frozen2020.sum()

    xm = [mem[c] for c in GRID]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#444"})

    # ---- FIG 1: mechanism (switching) ----
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for r in RES:
        ax.plot(xm, [per_res[r]["switch"][c] for c in GRID], "o-", lw=1.8, label=r)
    ax.plot(xm, [switch[c] for c in GRID], "k--o", lw=2.2, label="pilot mean")
    ax.axvline(mem[1.0], color="0.5", ls=":", lw=1.5)
    ax.text(mem[1.0], ax.get_ylim()[1], " baseline Q", va="top", fontsize=9, color="0.4")
    ax.set_xscale("log"); ax.invert_xaxis()          # short memory (fast learner) on the right
    ax.set_xlabel("effective belief memory (years)  —  faster learner →")
    ax.set_ylabel("policy switching rate  (fraction of years the deployed policy changes)")
    ax.set_title("A fast learner (short memory) switches policy more", fontweight="bold")
    ax.grid(alpha=0.3, which="both"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "q_sensitivity_mechanism.png", dpi=160)
    print("wrote q_sensitivity_mechanism.png")

    # ---- FIG 2: outcome (cost mean + spread) ----
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    col = {"bayes": "#e8833a", "blend": "#d1495b", "plugin": "#00a3a3"}
    for m in agg:
        a1.plot(xm, [agg[m][c] for c in GRID], "o-", lw=2, color=col[m], label=m)
        a2.plot(xm, [spread[m][c] for c in GRID], "o-", lw=2, color=col[m], label=m)
    for a in (a1, a2):
        a.axvline(mem[1.0], color="0.5", ls=":", lw=1.5)
        a.set_xscale("log"); a.invert_xaxis()
        a.set_xlabel("effective belief memory (years)  —  faster learner →")
        a.grid(alpha=0.3, which="both"); a.legend(frameon=False)
    a1.set_ylabel("cost reduction vs frozen 2020  (%)")
    a1.set_title("Net cost (mean)", fontweight="bold")
    a2.set_ylabel("p90 of per-trajectory cost / frozen")
    a2.set_title("Cost spread (p90 tail)", fontweight="bold")
    fig2.suptitle("Learning-rate outcome: mean cost is flat, the tail is where it moves",
                  fontweight="bold")
    fig2.tight_layout(); fig2.savefig(OUT / "q_sensitivity_outcome.png", dpi=160)
    print("wrote q_sensitivity_outcome.png")


if __name__ == "__main__":
    main()
