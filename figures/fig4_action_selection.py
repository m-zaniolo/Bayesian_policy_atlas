#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined action-selection figure: 3 reservoirs (New Bullards Bar=BUL, Shasta=SHA,
New Melones=NML) x 2 snapshot years (near 2035, far 2099), as a 2x6 grid.
Top row  = posterior weights (+ belief mean, x true position).
Bottom row = expected-cost landscape J.w (o mean-exposure, * posterior-optimal,
             box posterior-averaged 90% support).
NO titles/row labels (overlaid in PowerPoint). Bigger axis ticks/labels. A single
standalone legend is written separately (legend_action_selection.png) to slot in.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps")); sys.path.insert(0, str(REPO / "bdps"))
from data_io import load_inflow, load_demand, reservoir_config      # noqa: E402
from belief import annual_events, fit_gamma_shape, fit_hyper, particle_filter  # noqa: E402
from deploy import (historical_obs, calibrate_belief_width,    # noqa: E402
                          demand_llt, weights, CACHE, ATLAS_YEARS, START_YEAR)
from fig4b_three_strategies import support_box                        # noqa: E402

H = 10
RESV = [("BUL", "cnrm-cm5_rcp85_r1i1p1", "GCAM-26"),
        ("SHA", "cnrm-cm5_rcp85_r1i1p1", "GCAM-26"),
        ("NML", "cnrm-cm5_rcp85_r1i1p1", "GCAM-26")]
YEARS = [2035, 2099]
LBL, TICK = 14, 12                                                    # bigger fonts
OUT = REPO / "results"


def setup(res, cs, ls):
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
    fm, fsd, _ = particle_filter(nev, V, ql, qm, a, 1500, np.random.default_rng(1), phi=phi)
    fm = fm + np.log(ATLAS_YEARS); fsd = cw * np.sqrt(fsd ** 2 + max(H - 1, 0) * (ql + qm))
    dmf, dd = load_demand(ls); md = dd.year.values >= START_YEAR
    wy = dd.year.values[md]; annd = np.array([dmf[md][wy == y].mean() for y in np.unique(wy)])
    T = min(len(fm), len(annd), len(yrs)); dmu, dsd = demand_llt(annd[:T], H=H)
    tf = np.array([np.log(max(V[k:k + H].mean() * ATLAS_YEARS, 1)) for k in range(T)])
    return dict(J=J, coord=coord, dem=dem, kap=kap, c=c, ks=ks, ds=ds, to_k=to_k,
                grid=grid, fm=fm, fsd=fsd, dmu=dmu, dsd=dsd, annd=annd, tf=tf,
                yrs=yrs[:T], T=T)


def _edges_log(v):
    lv = np.log(v); e = np.empty(len(v) + 1)
    e[1:-1] = (lv[:-1] + lv[1:]) / 2
    e[0] = lv[0] - (lv[1] - lv[0]) / 2; e[-1] = lv[-1] + (lv[-1] - lv[-2]) / 2
    return np.exp(e)


def _edges_lin(v):
    e = np.empty(len(v) + 1); e[1:-1] = (v[:-1] + v[1:]) / 2
    e[0] = v[0] - (v[1] - v[0]) / 2; e[-1] = v[-1] + (v[-1] - v[-2]) / 2
    return e


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.ticker as mticker

    S = [setup(*r) for r in RESV]
    # precompute belief weights / costs and global norms
    for s in S:
        s["idx"] = [int(np.argmin(np.abs(s["yrs"] - y))) for y in YEARS]
        s["Ws"] = [weights(s["coord"], s["dem"], s["fm"][t], s["fsd"][t], s["dmu"][t], s["dsd"][t])
                   for t in s["idx"]]
        s["Jws"] = [s["J"] @ w for w in s["Ws"]]
        jj = np.concatenate([jw for jw in s["Jws"]])
        s["cmin"], s["cmax"] = float(jj.min()), float(jj.max())          # per-reservoir cost norm
    wvmax = max(w.max() for s in S for w in s["Ws"])                     # ONE weight scale (global)

    fig, ax = plt.subplots(2, 6, figsize=(21, 7.0), sharey=True, constrained_layout=True)
    pc0ref = pc1ref = None
    for ri, s in enumerate(S):
        for k, (t, w, Jw) in enumerate(zip(s["idx"], s["Ws"], s["Jws"])):
            col = 2 * ri + k
            a0, a1 = ax[0, col], ax[1, col]
            pc0ref = a0.pcolormesh(s["ks"], s["ds"], s["grid"](w), shading="nearest",
                                   cmap="Blues", vmin=0, vmax=wvmax)
            a0.scatter([s["to_k"](s["fm"][t])], [s["dmu"][t]], marker="+", s=170, color="k",
                       lw=2.2, zorder=5)
            a0.scatter([s["to_k"](s["tf"][t])], [s["annd"][t]], marker="x", s=110,
                       color="#D62728", lw=2.8, zorder=6)
            # cost normalised to [0,1] per reservoir -> ONE shared colour scale
            crel = (s["grid"](Jw) - s["cmin"]) / max(s["cmax"] - s["cmin"], 1e-9)
            pc1ref = a1.pcolormesh(s["ks"], s["ds"], crel, shading="nearest",
                                   cmap="viridis_r", vmin=0, vmax=1)
            d0 = s["ds"][np.argmin(np.abs(s["ds"] - s["dmu"][t]))]
            rowm = np.isclose(s["dem"], d0)
            ib = int(np.where(rowm)[0][np.argmin(Jw[rowm])])
            ip = int(np.where(rowm)[0][np.argmin(np.abs(s["c"][rowm] - s["fm"][t]))])
            x0, x1, y0, y1 = support_box(w, s["kap"], s["dem"])
            a1.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                         boxstyle="round,pad=0,rounding_size=0.02", transform=a1.transData,
                         fill=False, edgecolor="#E8833A", lw=2.6, zorder=5))
            a1.scatter([s["kap"][ib]], [s["dem"][ib]], marker="*", s=360, facecolor="none",
                       edgecolor="#D62728", lw=2.6, zorder=6)
            a1.scatter([s["kap"][ip]], [s["dem"][ip]], marker="o", s=150, facecolor="none",
                       edgecolor="k", lw=2.0, zorder=6)
            xe, ye = _edges_log(s["ks"]), _edges_lin(s["ds"])
            for a_ in (a0, a1):
                a_.set_xscale("log")
                a_.set_xlim(xe[0], xe[-1]); a_.set_ylim(ye[0], ye[-1])   # tight: no white space
                ticks = [v for v in (0.25, 0.5, 1, 2, 5) if xe[0] <= v <= xe[-1]]
                a_.xaxis.set_major_locator(mticker.FixedLocator(ticks))
                a_.xaxis.set_major_formatter(mticker.ScalarFormatter())
                a_.xaxis.set_minor_locator(mticker.NullLocator())        # no cluttered minor ticks
                a_.tick_params(labelsize=TICK)
            a0.tick_params(labelbottom=False)                            # share x within column
            a1.set_xlabel("flood magnitude  κ", fontsize=LBL)
            if col == 0:
                a0.set_ylabel("demand multiplier", fontsize=LBL)
                a1.set_ylabel("demand multiplier", fontsize=LBL)

    cb0 = fig.colorbar(pc0ref, ax=ax[0, :], location="right", fraction=0.02, pad=0.01)
    cb0.set_label("posterior weight", fontsize=LBL); cb0.ax.tick_params(labelsize=TICK)
    cb1 = fig.colorbar(pc1ref, ax=ax[1, :], location="right", fraction=0.02, pad=0.01)
    cb1.set_label("expected cost (relative, per reservoir)", fontsize=LBL - 1)
    cb1.ax.tick_params(labelsize=TICK)

    fig.savefig(OUT / "action_selection_fig.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "action_selection_fig.pdf", bbox_inches="tight")
    print("wrote action_selection_fig.png/.pdf")

    # ---- standalone legend (slot in wherever) ----
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    figL, axL = plt.subplots(figsize=(4.6, 2.2)); axL.axis("off")
    handles = [
        Line2D([0], [0], marker="+", color="k", lw=0, mew=2.2, ms=13, label="belief mean"),
        Line2D([0], [0], marker="x", color="#D62728", lw=0, mew=2.8, ms=11, label="true position"),
        Line2D([0], [0], marker="o", color="k", lw=0, mfc="none", mew=2.0, ms=12,
               label="(1) mean-exposure policy"),
        Line2D([0], [0], marker="*", color="#D62728", lw=0, mfc="none", mew=2.2, ms=17,
               label="(2) posterior-optimal policy"),
        Patch(facecolor="none", edgecolor="#E8833A", lw=2.6,
              label="(3) posterior-averaged (90% belief support)"),
    ]
    axL.legend(handles=handles, loc="center", frameon=True, fontsize=13, handlelength=1.6)
    figL.savefig(OUT / "legend_action_selection.png", dpi=200, bbox_inches="tight")
    figL.savefig(OUT / "legend_action_selection.pdf", bbox_inches="tight")
    print("wrote legend_action_selection.png/.pdf")


if __name__ == "__main__":
    main()
