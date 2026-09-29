#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/plot_three_strategies.py -- belief landscape + the THREE action-selection
strategies, one figure per reservoir (reconstructs three_strategies_<res>.png).

Top row  : posterior weights p(cell|data) at three snapshot years, with the
           belief mean (+) and the true position (x).
Bottom row: expected-cost landscape J·w, with the three decision rules marked:
    (1) mean-exposure policy      -- atlas cell nearest the posterior MEAN   (o)
    (2) posterior-optimal policy  -- argmin of E[cost] over the posterior     (*)
    (3) posterior-averaged action -- integrates over the 90% belief support (box)

Decisions are pinned to the belief's demand row (demand is confidently known and
the cost is ~flat along it). Uses the CANONICAL horizon H=1 (the fixed ~annual
design); the belief width therefore carries no multi-year forecast inflation.

USAGE
    python scripts/plot_three_strategies.py                # all reservoirs
    python scripts/plot_three_strategies.py --res NML,DNP
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps")); sys.path.insert(0, str(REPO / "bdps"))
from data_io import load_inflow, load_demand, reservoir_config      # noqa: E402
from belief import (annual_events, fit_gamma_shape, fit_hyper,   # noqa: E402
                        particle_filter)
from deploy import (historical_obs, calibrate_belief_width,    # noqa: E402
                          demand_llt, weights, CACHE, ATLAS_YEARS, START_YEAR)

H = 10  # belief horizon for the landscape display (matches belief-tracking fig)
DEFAULT_GCM = ("cnrm-cm5_rcp85_r1i1p1", "GCAM-26")
OUTDIR = REPO / "results" / "all_reservoir_figures"


def support_box(w, kap, dem, frac=0.90):
    """Bounding box (in kappa, demand) of the smallest set of cells holding
    `frac` of the posterior weight -- what the posterior-averaged action
    effectively integrates over."""
    order = np.argsort(w)[::-1]
    csum = np.cumsum(w[order]) / max(w.sum(), 1e-12)
    keep = order[:np.searchsorted(csum, frac) + 1]
    kk, dd = kap[keep], dem[keep]
    ks = np.array(sorted(np.unique(kap))); ds = np.array(sorted(np.unique(dem)))
    dk = np.diff(np.log(ks)).mean() if ks.size > 1 else 0.2
    dd_ = np.diff(ds).mean() if ds.size > 1 else 0.1
    return (np.exp(np.log(kk.min()) - dk / 2), np.exp(np.log(kk.max()) + dk / 2),
            dd.min() - dd_ / 2, dd.max() + dd_ / 2)


def one(res, plt):
    from matplotlib.patches import FancyBboxPatch
    cs, ls = DEFAULT_GCM
    cfg = reservoir_config(res); z = np.load(CACHE / f"{res}.npz")
    J, coord, dem, kap = z["J"], z["coord"], z["demand"], z["kappa"]
    c = np.log(np.maximum(coord, 1.0))
    ks = np.array(sorted(np.unique(kap))); ds = np.array(sorted(np.unique(dem)))
    ck = np.array([c[np.isclose(kap, k)].mean() for k in ks])
    to_k = lambda v: float(np.interp(v, ck, ks))

    def grid(v):
        G = np.full((len(ds), len(ks)), np.nan)
        for i in range(len(v)):
            G[np.argmin(np.abs(ds - dem[i])), np.argmin(np.abs(ks - kap[i]))] = v[i]
        return G

    nH, VH = historical_obs(res); a = fit_gamma_shape([(nH, VH)])
    ql, qm, phi = fit_hyper([(nH, VH)], a, n_part=1200)
    cw = calibrate_belief_width(res, ql, qm, a, phi, H)
    q, _, dts = load_inflow(cs, key=res); m = dts.year.values >= START_YEAR
    q, dts = q[m], dts[m]
    yrs, nev, V = annual_events(q, dts, cfg.r_max_cfs)
    fm, fsd, _ = particle_filter(nev, V, ql, qm, a, 1500,
                                 np.random.default_rng(1), phi=phi)
    fm = fm + np.log(ATLAS_YEARS)
    fsd = cw * np.sqrt(fsd ** 2 + max(H - 1, 0) * (ql + qm))
    dmf, dd = load_demand(ls); md = dd.year.values >= START_YEAR
    wy = dd.year.values[md]
    annd = np.array([dmf[md][wy == y].mean() for y in np.unique(wy)])
    T = min(len(fm), len(annd), len(yrs)); dmu, dsd = demand_llt(annd[:T], H=H)
    tf = np.array([np.log(max(V[k:k + H].mean() * ATLAS_YEARS, 1))
                   for k in range(T)])

    ys = [15, 30, T - 1] if T > 31 else [8, 15, T - 1]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    def hm(a_, G, cmap, title, cbar):
        pc = a_.pcolormesh(ks, ds, G, shading="nearest", cmap=cmap)
        a_.set_xscale("log"); a_.set_xticks([0.25, 0.5, 1, 2, 5])
        a_.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        a_.set_xlabel("flood magnitude  κ"); a_.set_ylabel("demand multiplier")
        a_.set_title(title, fontsize=10, fontweight="bold")
        fig.colorbar(pc, ax=a_, label=cbar)

    for col, t in enumerate(ys):
        w = weights(coord, dem, fm[t], fsd[t], dmu[t], dsd[t]); Jw = J @ w
        d0 = ds[np.argmin(np.abs(ds - dmu[t]))]
        row = np.isclose(dem, d0)
        ib = int(np.where(row)[0][np.argmin(Jw[row])])              # argmin
        ip = int(np.where(row)[0][np.argmin(np.abs(c[row] - fm[t]))])  # mean

        a0 = ax[0, col]
        hm(a0, grid(w), "Blues", f"{int(yrs[t])} — posterior weights", "weight")
        a0.scatter([to_k(fm[t])], [dmu[t]], marker="+", s=150, color="k", lw=2,
                   zorder=5, label="belief mean")
        a0.scatter([to_k(tf[t])], [annd[t]], marker="x", s=95, color="#D62728",
                   lw=2.5, zorder=6, label="true position")
        if col == 0:
            a0.legend(fontsize=8, loc="lower left")

        a1 = ax[1, col]
        hm(a1, grid(Jw / 1e9), "viridis_r",
           f"{int(yrs[t])} — expected-cost landscape  J·w",
           "E[cost] of cell's policy (×10⁹)")
        # (3) posterior-averaged action: outline of the 90% belief support
        x0, x1, y0, y1 = support_box(w, kap, dem)
        a1.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0,rounding_size=0.02",
            transform=a1.transData, fill=False, edgecolor="#E8833A", lw=2.4,
            zorder=5, label="(3) posterior-averaged: 90% belief support"))
        # (2) posterior-optimal policy
        a1.scatter([kap[ib]], [dem[ib]], marker="*", s=340, facecolor="none",
                   edgecolor="#D62728", lw=2.4, zorder=6,
                   label="(2) posterior-optimal policy")
        # (1) mean-exposure policy
        a1.scatter([kap[ip]], [dem[ip]], marker="o", s=130, facecolor="none",
                   edgecolor="k", lw=1.8, zorder=6,
                   label="(1) mean-exposure policy")
        if col == 0:
            a1.legend(fontsize=8, loc="lower left")

    fig.suptitle(f"{res} — belief landscape and the three action-selection "
                 f"strategies (H={H};  orange = cells the averaged action "
                 f"integrates over)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"three_strategies_{res}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="NML,EXC,NHG,BUL,FOL,SHA,MIL,ORO,DNP")
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for r in [x.strip() for x in args.res.split(",")]:
        try:
            one(r, plt)
        except Exception as e:
            print(f"FAILED {r}: {e}")


if __name__ == "__main__":
    main()
