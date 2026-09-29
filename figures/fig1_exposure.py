#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exposure-space trajectories of the deployed ensemble, 9 panels (one per reservoir).

Every GCM x demand scenario traces a path over time through the 2-D exposure
space the atlas is built on:
    x = flood exposure  = per-water-year max 7-day inflow volume (af)  [belief signal]
    y = demand exposure = per-water-year mean delivery multiplier      [from LULC]
Colour = water year, so climate/land-use DRIFT is visible: paths sweep right
(bigger floods) and up (higher demand) as the century progresses.

The exposure coordinate and demand coordinate are the exact per-year signals the
belief tracks (src/belief.annual_flood_demand). Flood is drawn on a log axis
(heavy-tailed). Each per-year path is smoothed with an 11-year centred rolling
mean so the drift is legible rather than a scribble; demand is reservoir-
independent, so the y-range is identical across panels.

USAGE
    python scripts/plot_exposure_trajectories.py
    python scripts/plot_exposure_trajectories.py --smooth 11 --start-year 2020
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))
from data_io import load_inflow, load_demand   # noqa: E402
from belief import annual_flood_demand          # noqa: E402

CFSD_TO_AF = 1.98347
RES = ["NML", "EXC", "NHG", "BUL", "FOL", "SHA", "MIL", "ORO", "DNP"]


def water_year_bounds(dowy):
    """(start, end) index pairs for each water year (dowy resets to 0 at Oct 1)."""
    starts = np.where(dowy == 0)[0]
    return [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]


def rollmean(a, w):
    if w <= 1 or a.size < w:
        return a
    k = np.ones(w) / w
    out = np.convolve(a, k, "same")
    h = w // 2                      # trim convolution edge bias
    out[:h] = a[:h]; out[-h:] = a[-h:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--smooth", type=int, default=11, help="rolling-mean window (yrs)")
    ap.add_argument("--out", default=str(REPO / "results" / "exposure_trajectories.png"))
    args = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    b = pd.read_csv(REPO / "results" / "bench_baselines.csv")
    clim = sorted(b.clim.unique())                     # 30 deployed GCM scenarios
    lulc = sorted(b.lulc.unique())                     # 8 deployed demand scenarios
    print(f"{len(clim)} GCM x {len(lulc)} demand = {len(clim)*len(lulc)} "
          f"trajectories / reservoir, from {args.start_year}")

    # --- demand: reservoir-independent, load each LULC once -------------------
    dem_by_lulc, yb_ref, years_ref = {}, None, None
    _, dowy0, dates0 = load_inflow(clim[0], key=RES[0])
    yb_ref = water_year_bounds(dowy0)
    years_ref = np.array([dates0.year[a] for a, _ in yb_ref])   # WY start calendar year
    for ls in lulc:
        dm, _ = load_demand(ls)
        _, dem = annual_flood_demand(np.zeros(len(dm)), dm, yb_ref)
        dem_by_lulc[ls] = dem
    print("demand scenarios loaded")

    # --- flood: per (clim, res); each clim file holds all reservoirs ----------
    flood = {r: {} for r in RES}                       # flood[res][clim] -> per-WY af
    for i, cs in enumerate(clim):
        for r in RES:
            q, dowy, _ = load_inflow(cs, key=r)
            yb = water_year_bounds(dowy)
            fl, _ = annual_flood_demand(q, np.zeros(len(q)), yb)
            flood[r][cs] = fl * CFSD_TO_AF
        print(f"  climate {i+1}/{len(clim)} {cs}")

    # --- plot -----------------------------------------------------------------
    keep = years_ref >= args.start_year
    yrs = years_ref[keep]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(yrs.min(), yrs.max())

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#444", "axes.linewidth": 0.8})
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.5))
    for ax, r in zip(axes.flat, RES):
        segs, cvals = [], []
        for cs in clim:
            fl = rollmean(flood[r][cs][keep], args.smooth)
            for ls in lulc:
                dm = rollmean(dem_by_lulc[ls][keep], args.smooth)
                pts = np.column_stack([fl, dm])
                s = np.stack([pts[:-1], pts[1:]], axis=1)
                segs.append(s); cvals.append(yrs[:-1])
        lc = LineCollection(np.concatenate(segs), cmap=cmap, norm=norm,
                            linewidths=0.5, alpha=0.28)
        lc.set_array(np.concatenate(cvals))
        ax.add_collection(lc)
        ax.set_xscale("log")
        allf = np.concatenate([flood[r][cs][keep] for cs in clim])
        ax.set_xlim(np.percentile(allf, 0.5), np.percentile(allf, 99.8))
        ax.set_ylim(min(d[keep].min() for d in dem_by_lulc.values()) * 0.99,
                    max(d[keep].max() for d in dem_by_lulc.values()) * 1.01)
        ax.set_title(r, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("flood exposure — max 7-day inflow volume (af, log)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("demand exposure — mean multiplier", fontsize=9)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("water year", fontsize=10)
    fig.suptitle("Exposure-space trajectories of the deployed ensemble "
                 f"({len(clim)} GCM × {len(lulc)} demand), {args.start_year}–{int(yrs.max())}",
                 fontsize=14, fontweight="bold", x=0.44)
    fig.text(0.44, 0.055,
             "Each faint path = one GCM×demand scenario stepping through the exposure "
             "space over time (11-yr smoothed). Colour = water year: paths drift right "
             "(bigger floods) and up (higher demand) as the century progresses — the "
             "non-stationarity the atlas + belief must track.",
             ha="center", fontsize=8.5, color="#555", wrap=True)
    for d in (Path(args.out).parent, REPO / "results" / "all_reservoir_figures"):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "exposure_trajectories.png", dpi=160, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
