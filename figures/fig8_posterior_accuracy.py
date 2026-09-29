#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw posterior accuracy of the deployed belief, as heatmaps over time.

Metric = RAW accuracy, NOT calibration:  |posterior_mean - true| / sigma_clim,
the belief's forward-window prediction error as a FRACTION of that driver's own
climatological STANDARD DEVIATION (sigma_clim = std of the realised exposure
across all scenarios/years for that driver at that horizon). Comparable flood-vs-
demand and unit-free:
    0 = perfect tracking;  1 = error of one climatological sigma.  (Green good -> red bad.)

Reads outputs/posterior_accuracy.csv.gz (analyze_posterior_accuracy.py). Two
figures, both at a chosen forward horizon:
  * posterior_accuracy_by_reservoir.png -- flood per RESERVOIR (mean over GCMs)
    + demand per LULC (demand is reservoir-independent).
  * posterior_accuracy_by_scenario.png  -- flood per GCM (mean over reservoirs)
    + demand per LULC.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results"
FIGDIR = OUT / "all_reservoir_figures"
RES = ["NML", "EXC", "NHG", "BUL", "FOL", "SHA", "MIL", "ORO", "DNP"]


def grid(df, rowcol, years):
    """median rawnorm as a (row x year) matrix."""
    piv = (df.groupby([rowcol, "year"])["rawnorm"].median()
           .unstack("year").reindex(columns=years))
    return piv


def heat(ax, M, years, cmap, vmax, title, ylabel):
    rows = np.arange(M.shape[0])
    im = ax.pcolormesh(years, rows, M.values, cmap=cmap, vmin=0, vmax=vmax,
                       shading="nearest")
    ax.set_yticks(rows)
    ax.set_yticklabels(M.index, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.invert_yaxis()
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--vmax", type=float, default=1.5)
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(OUT / "posterior_accuracy.csv.gz")
    data_end = int(df[df.H == 1].year.max())          # last realised water year
    df = df[df.H == args.horizon]
    # only decision years with a FULL forward-H realised window (match plot_belief_multi);
    # the stored file allowed partial windows down to H//2 near the end -> drop them.
    df = df[df.year <= data_end - args.horizon + 1]
    # recompute the accuracy normaliser over the full-window subset only
    sd_clim = df.groupby("driver")["true"].transform("std").clip(lower=1e-9)
    df["rawnorm"] = (df["mean"] - df["true"]).abs() / sd_clim
    fl = df[df.driver == "flood"].copy()
    dm = df[df.driver == "demand"].copy()
    years = np.array(sorted(df.year.unique()))
    cmap = "RdYlGn_r"
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # ---- FIG 1: per reservoir (flood) + per LULC (demand) --------------------
    flg = grid(fl, "unit", years).reindex(RES)
    dmg = grid(dm, "unit", years)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [len(flg), len(dmg)]})
    im = heat(axes[0], flg, years, cmap, args.vmax,
              f"FLOOD  ·  raw prediction error / climatological σ  "
              f"(forward {args.horizon}-yr window)", "reservoir\n(median over 30 GCM)")
    heat(axes[1], dmg, years, cmap, args.vmax,
         "DEMAND  ·  same metric (reservoir-independent)", "LULC scenario")
    axes[1].set_xlabel("decision year", fontsize=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, extend="max")
    cb.set_label("|mean − true| / climatological σ\n(0 = perfect; 1 = one climatological σ of error)",
                 fontsize=9)
    fig.suptitle("Posterior raw accuracy over time, per reservoir",
                 fontsize=13, fontweight="bold", x=0.44)
    for d in (OUT, FIGDIR):
        fig.savefig(d / "posterior_accuracy_by_reservoir.png", dpi=160, bbox_inches="tight")
    print("wrote posterior_accuracy_by_reservoir.png")

    # ---- FIG 2: per GCM (flood) + per LULC (demand) --------------------------
    order = (fl.groupby("scen")["rawnorm"].median().sort_values().index)
    flg2 = grid(fl, "scen", years).reindex(order)
    flg2.index = [s.replace("_r1i1p1", "") for s in flg2.index]
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 12),
                               gridspec_kw={"height_ratios": [len(flg2), len(dmg)]})
    im2 = heat(axes2[0], flg2, years, cmap, args.vmax,
               f"FLOOD  ·  per GCM (median over 9 reservoirs), forward {args.horizon}-yr window",
               "GCM (best→worst)")
    heat(axes2[1], dmg, years, cmap, args.vmax,
         "DEMAND  ·  per LULC scenario", "LULC scenario")
    axes2[1].set_xlabel("decision year", fontsize=10)
    cb2 = fig2.colorbar(im2, ax=axes2, fraction=0.03, pad=0.02, extend="max")
    cb2.set_label("|mean − true| / climatological spread\n(0 perfect · 1 = climatology)",
                  fontsize=9)
    fig2.suptitle("Posterior raw accuracy over time, per scenario",
                  fontsize=13, fontweight="bold", x=0.44)
    for d in (OUT, FIGDIR):
        fig2.savefig(d / "posterior_accuracy_by_scenario.png", dpi=160, bbox_inches="tight")
    print("wrote posterior_accuracy_by_scenario.png")


if __name__ == "__main__":
    main()
