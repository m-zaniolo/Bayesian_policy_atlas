#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3x3 per-reservoir barplot of REALISED (absolute) cost, each bar split into its
FLOOD component (solid) and DEMAND/shortage component (hatched). Same method
colours as plot_blend_figs.py, plus the True-exposure (truelookup) policy.

Costs are cost-weighted totals (sum over the 240 trajectories) of J_flood and
J_shortage, from the component-instrumented deployment runs:
    outputs/bench_components.csv       (frozen, ff, truelookup, oracle)
    outputs/bayes_deploy_components.csv (plugin, bayes, blend)
"""
from pathlib import Path
import argparse
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results"

_ap = argparse.ArgumentParser()
_ap.add_argument("--freeopt", action="store_true",
                 help="replace the atlas oracle with the FREE-PARAMETER oracle "
                      "(outputs/freeparam_oracle.csv); writes a _freeopt figure")
ARGS = _ap.parse_args()

K = ["res", "clim", "lulc"]
m = pd.read_csv(OUT / "bayes_deploy_components.csv")   # plugin/bayes/blend
fz = pd.read_csv(OUT / "frozen2020_components.csv")    # DE-frozen-2020 (matches the barplot baseline)
m = m.merge(fz, on=K)
for extra in ("ff_components.csv", "oracle_components.csv"):   # ff / oracle, if present
    if (OUT / extra).exists():
        e = pd.read_csv(OUT / extra)
        cols = [c for c in e.columns if c.startswith(("J_ff_15_50", "J_oracle"))]
        m = m.merge(e[K + cols], on=K, how="left")

# key, display label, J-column base, colour  (same as the barplot, minus truelookup)
METHODS = [
    ("frozen", "frozen 2020",               "J_frozen2020", "#9aa0a6"),
    ("ff",     "ff 15/50",                   "J_ff_15_50",   "#5b6cff"),
    ("plugin", "Mean-exposure policy",       "J_plugin",     "#00a3a3"),
    ("bayes",  "Posterior-optimal policy",   "J_bayes",      "#e8833a"),
    ("blend",  "Posterior-averaged action",  "J_blend",      "#d1495b"),
    ("oracle", "oracle",                     "J_oracle",     "#3f8f4f"),
]

if ARGS.freeopt:
    # swap the atlas oracle for the free-parameter oracle (same 7-param shape,
    # continuous parameters). Needs the component columns written by the patched
    # freeparam_oracle.py -- re-run it if they are missing.
    fp = pd.read_csv(OUT / "freeparam_oracle.csv")
    need = ["J_oracle_free_flood", "J_oracle_free_short"]
    missing = [c for c in need if c not in fp.columns]
    if missing:
        raise SystemExit(
            f"freeparam_oracle.csv lacks {missing}. Re-run:\n"
            "  python scripts/freeparam_oracle.py --jobs 24 --popsize 25 "
            "--maxiter 120 --out outputs/freeparam_oracle.csv")
    m = m.merge(fp[K + need], on=K, how="left")
    METHODS = [t for t in METHODS if t[0] != "oracle"]
    METHODS.append(("oracle_free", "free-param oracle", "J_oracle_free", "#3f8f4f"))
RES = ["NML", "EXC", "NHG", "BUL", "FOL", "SHA", "MIL", "ORO", "DNP"]

# keep only methods whose flood/demand split is actually present
METHODS = [t for t in METHODS
           if f"{t[2]}_flood" in m.columns and not m[f"{t[2]}_flood"].isna().all()]
print("methods in figure:", [t[0] for t in METHODS])

# per-reservoir MEAN flood/demand cost per trajectory (bars).
tot = {}
for res, g in m.groupby("res"):
    tot[res] = {k: (float(g[f"{col}_flood"].values.mean()),
                    float(g[f"{col}_short"].values.mean())) for k, _, col, _ in METHODS}


# worst-10%-YEARS tail (CVaR_0.9 of per-YEAR cost) -- only for the deployed methods
# that have a per-year stream (cost_over_time.csv.gz); ff/oracle/frozen2020 don't.
def cvar(x, alpha=0.10):
    thr = np.quantile(x, 1 - alpha); return float(x[x >= thr].mean())


ct = pd.read_csv(OUT / "cost_over_time.csv.gz"); ct["J"] = ct.J_flood + ct.J_shortage
year_tail = {}   # year_tail[res][method key] = worst-10%-years mean cost
for res, g in ct.groupby("res"):
    year_tail[res] = {mth: cvar(g[g.method == mth].J.values)
                      for mth in ("plugin", "bayes", "blend")}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#444", "axes.linewidth": 0.8})
fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.4))
keys = [k for k, _, _, _ in METHODS]
colors = [c for _, _, _, c in METHODS]
x = np.arange(len(keys))

TAILK = [mth for mth in ("plugin", "bayes", "blend") if mth in keys]   # have per-year
for ax, res in zip(axes.flat, RES):
    flood = np.array([tot[res][k][0] for k in keys])
    short = np.array([tot[res][k][1] for k in keys])
    scL = 10 ** np.floor(np.log10(max((flood + short).max(), 1.0)))   # left unit (bars)
    fl, sh = flood / scL, short / scL
    ax.bar(x, fl, color=colors, edgecolor="#333", linewidth=0.6, width=0.76, zorder=3)  # flood
    ax.bar(x, sh, bottom=fl, color=colors, edgecolor="#333", linewidth=0.6,
           width=0.76, hatch="////", zorder=3)             # hatched = demand
    ax.set_ylim(0, (fl + sh).max() * 1.12)
    # worst-10%-YEARS tail (right axis) -- only the deployed methods
    tv = {keys.index(mth): year_tail[res][mth] for mth in TAILK}
    scR = 10 ** np.floor(np.log10(max(max(tv.values()), 1.0)))
    ax2 = ax.twinx()
    for idx, val in tv.items():
        t = val / scR
        ax2.plot([idx - 0.42, idx + 0.42], [t, t], color="#111", lw=2.2, zorder=6)
        ax2.plot(idx, t, "D", color="#111", ms=5, zorder=7)
    ax2.set_ylim(0, max(tv.values()) / scR * 1.25)
    ax2.set_ylabel(f"worst-10% yr  (×{scR:.0e})", fontsize=7, color="#333")
    ax2.tick_params(labelsize=7)
    ax.set_xticks([])
    ax.grid(axis="y", color="#ddd", lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_title(res, fontsize=11, fontweight="bold")
    ax.set_ylabel(f"mean cost / traj  (×{scL:.0e})", fontsize=8)

from matplotlib.lines import Line2D
handles = [Patch(facecolor=c, edgecolor="#333", label=d) for _, d, _, c in METHODS]
handles += [Patch(facecolor="#bbb", edgecolor="#333", label="flood cost (solid)"),
            Patch(facecolor="#bbb", edgecolor="#333", hatch="////",
                  label="demand cost (hatched)"),
            Line2D([0], [0], color="#111", marker="D", lw=2.0,
                   label="worst-10% YEARS (right axis, deployed methods)")]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Realised cost by reservoir: mean per trajectory (flood solid, demand "
             "hatched) with the worst-10%-YEARS tail (right axis)", y=1.06,
             fontsize=12, fontweight="bold")
fig.text(0.5, -0.01,
         "Bars (left axis) = MEAN cost per trajectory (240 = 30 GCM x 8 demand), flood + "
         "demand.  Tick (right axis) = mean per-YEAR cost over the worst 10% of years "
         "(CVaR), for the deployed methods only (ff/oracle have no per-year stream).  "
         "Posterior-optimal's tick is lowest — it hedges the tail.",
         ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.005, 1, 1.0])
out = OUT / ("cost_components_3x3_freeopt.png" if ARGS.freeopt
             else "cost_components_3x3.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
print("wrote", out)

# quick text summary: flood share of total realised cost per reservoir (frozen)
print("\nflood share of realised cost (frozen policy):")
for res in RES:
    fl, sh = tot[res]["frozen"]
    print(f"  {res}: flood {100*fl/(fl+sh):4.0f}%   demand {100*sh/(fl+sh):4.0f}%")
