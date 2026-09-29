#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/plot_policy_atlas_3panel.py -- the policy atlas, all three views in ONE
row per reservoir (merges the two figures of plot_policy_atlas.py):

    [ x4 flood-pool depth ] [ release · flood-season state ] [ release · demand-season state ]

Panel 1 is the raw policy parameter x4 over the (kappa, demand) atlas.
Panels 2-3 are the PRESCRIBED RELEASE ACTION at two physically-chosen states,
one isolating the flood response (high storage, flood window -> release rises
with kappa) and one isolating the demand response (moderate storage, supply
season -> release rises with demand). The states are searched exactly as in
plot_policy_atlas.py.

USAGE
    python scripts/plot_policy_atlas_3panel.py                # all reservoirs
    python scripts/plot_policy_atlas_3panel.py --res NML
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))
sys.path.insert(0, str(REPO / "bdps"))
from data_io import reservoir_config                       # noqa: E402
from plot_policy_atlas import release_cfs, dowy_label       # noqa: E402

OUTDIR = REPO / "results" / "all_reservoir_figures"


def _one(res, atlas_all, plt, fixed=None):
    cfg = reservoir_config(res); K = float(cfg.capacity_af)
    at = atlas_all[atlas_all.res == res]
    if at.empty:
        print(f"skip {res}: no atlas"); return
    P = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]
    g = at.groupby(["kappa", "demand"])[P].median().reset_index()
    kappas = np.array(sorted(g.kappa.unique()))
    demands = np.array(sorted(g.demand.unique()))
    KK = np.empty((len(demands), len(kappas)))
    X = {p: np.full(KK.shape, np.nan) for p in P}
    for _, r in g.iterrows():
        i = np.where(demands == r.demand)[0][0]
        j = np.where(kappas == r.kappa)[0][0]
        for p in P:
            X[p][i, j] = r[p]

    def release_grid(Sf, d):
        S = Sf * K; S_avg_prev = float(cfg.S_avg[d - 1])
        R = np.full(KK.shape, np.nan)
        for i, dem in enumerate(demands):
            dem_cfs = float(cfg.R_avg[d]) * dem
            for j in range(len(kappas)):
                xv = np.array([X[p][i, j] for p in P])
                if not np.isnan(xv).any():
                    R[i, j] = release_cfs(xv, S, d, dem_cfs, K, S_avg_prev)
        return R, np.nanmean(np.nanstd(R, axis=1)), np.nanmean(np.nanstd(R, axis=0))

    if fixed is not None:
        # SAME (storage, day-of-year) for every reservoir, for direct comparison
        SfK, dK = fixed["flood_stor"], fixed["flood_dowy"]
        SfD, dD = fixed["demand_stor"], fixed["demand_dowy"]
        RK, _, _ = release_grid(SfK, dK)
        RD, _, _ = release_grid(SfD, dD)
    else:
        best_k = best_d = None
        for Sf in np.linspace(0.75, 0.92, 5):
            for d in range(80, 200, 6):
                R, v_k, _ = release_grid(Sf, d)
                if not np.isnan(R).all() and (best_k is None or v_k > best_k[0]):
                    best_k = (v_k, Sf, d, R)
        for Sf in np.linspace(0.30, 0.60, 7):
            for d in range(200, 330, 6):
                R, _, v_d = release_grid(Sf, d)
                if not np.isnan(R).all() and (best_d is None or v_d > best_d[0]):
                    best_d = (v_d, Sf, d, R)
        _, SfK, dK, RK = best_k
        _, SfD, dD, RD = best_d

    # 3x3 NaN-aware median smoothing of the release panels -- removes isolated
    # atlas-fit outliers (e.g. a single cell with a mis-placed TOCS breakpoint)
    # without shifting the overall gradient.
    from scipy.ndimage import generic_filter

    def smooth(R):
        if np.isnan(R).all():
            return R
        return generic_filter(R, np.nanmedian, size=3, mode="nearest")

    RK, RD = smooth(RK), smooth(RD)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    lk = np.log(kappas)
    dk = np.diff(lk).mean() if len(lk) > 1 else 0.2
    ddm = np.diff(demands).mean() if len(demands) > 1 else 0.1
    xlim = (np.exp(lk[0] - dk / 2), np.exp(lk[-1] + dk / 2))
    ylim = (demands[0] - ddm / 2, demands[-1] + ddm / 2)

    def heat(ax, Z, title, cbar_label, cmap):
        pc = ax.pcolormesh(kappas, demands, Z, shading="nearest", cmap=cmap)
        ax.set_xscale("log")
        ax.set_xticks([t for t in (0.25, 0.5, 1, 2, 5) if xlim[0] <= t <= xlim[1]])
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)   # tight: no white margin
        ax.set_xlabel("flood magnitude  κ"); ax.set_ylabel("demand multiplier")
        ax.set_title(title, fontsize=11, fontweight="bold")
        fig.colorbar(pc, ax=ax, label=cbar_label)

    heat(axes[0], X["x4"], "policy atlas: flood-pool depth  x4",
         "x4  (fraction of capacity drawn down)", "viridis")
    heat(axes[1], RK, f"release · flood-season state\n"
         f"(storage {SfK:.0%} cap, {dowy_label(dK)})", "release (cfs)", "magma")
    heat(axes[2], RD, f"release · demand-season state\n"
         f"(storage {SfD:.0%} cap, {dowy_label(dD)})", "release (cfs)", "magma")

    fig.suptitle(f"{res} — policy atlas: flood-pool parameter and the "
                 f"prescribed release in each regime", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"policy_atlas_3panel_{res}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out.name}  (flood {dowy_label(dK)}, demand {dowy_label(dD)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=None, help="reservoir; default = all")
    # SAME (day-of-water-year, storage) for every reservoir -> panels are
    # directly comparable across reservoirs. If any is set, the per-reservoir
    # state search is skipped and these fixed values are used everywhere.
    ap.add_argument("--flood-dowy", type=int, default=None,
                    help="fixed flood-season day-of-water-year (0=Oct 1)")
    ap.add_argument("--flood-stor", type=float, default=0.85,
                    help="fixed flood-season storage (fraction of capacity)")
    ap.add_argument("--demand-dowy", type=int, default=None,
                    help="fixed demand-season day-of-water-year (0=Oct 1)")
    ap.add_argument("--demand-stor", type=float, default=0.50,
                    help="fixed demand-season storage (fraction of capacity)")
    args = ap.parse_args()
    fixed = None
    if args.flood_dowy is not None or args.demand_dowy is not None:
        fixed = {"flood_dowy": args.flood_dowy, "flood_stor": args.flood_stor,
                 "demand_dowy": args.demand_dowy, "demand_stor": args.demand_stor}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    atlas_all = pd.read_csv(REPO / "results" / "find_atlas_fits.csv")
    reservoirs = ([args.res] if args.res else sorted(atlas_all.res.unique()))
    for res in reservoirs:
        _one(res, atlas_all, plt, fixed=fixed)


if __name__ == "__main__":
    main()
