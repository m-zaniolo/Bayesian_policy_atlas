#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/plot_belief_multi.py -- up to 4 GCM trajectories on one atlas, each in
its own colour: solid line = the REAL (true forward-window) path, matching
shaded cloud = the Bayesian belief.

DESIGN
    * No policy heatmap. With several coloured trajectories overlaid, a coloured
      background competes with them; the atlas is shown only as faint grey
      sample points so the reader sees where it has support.
    * The belief is drawn ONLY as shading -- no posterior-mean line. The shade
      is a proper Gaussian density accumulated over time: at each year the joint
      posterior N(mean, sd) is added onto a grid, so the cloud is DARK along the
      path (where the mean sits) and fades toward the edges, instead of being a
      flat blob with a hard 95% rim.
    * Each GCM gets one hue. Belief cloud and true line share it, so "red
      shading = red line's belief" needs no legend entry.

USAGE
    python scripts/plot_belief_multi.py --res NML
    python scripts/plot_belief_multi.py --res DNP --scens a,b,c,d --lulc GCAM-26
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))
sys.path.insert(0, str(REPO / "bdps"))

from data_io import (load_inflow, load_demand, reservoir_config,       # noqa: E402
                     scenario_names)
from belief import (annual_events, fit_gamma_shape, fit_hyper,     # noqa: E402
                        particle_filter)
from plot_belief_trajectory import demand_llt                          # noqa: E402

ATLAS_YEARS = 50.0
COORD = "vol_above_safe_af"
# distinguishable at low alpha, and colour-blind-safe enough as a set
COLOURS = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD"]   # red, blue, green, purple


def belief_cloud(GX, GY, fm, fsd, dmu, dsd, sub=6, floor=0.18):
    """Continuous posterior TUNNEL along the trajectory.

    Sum_t N(fm[t], fsd[t]) x N(dmu[t], dsd[t]), but the path is INTERPOLATED to
    `sub` steps per year first. One Gaussian per YEAR leaves visible gaps: the
    demand axis moves quickly through mid-century and then plateaus, so
    consecutive annual posteriors land far enough apart that they read as
    separate ellipses instead of a moving belief. Sub-year interpolation makes
    successive kernels overlap, giving a continuous tunnel.

    `floor` clips the diffuse outer tail to 0, which BOUNDS the tunnel (a
    visible edge instead of an unbounded haze) and stops four overlaid clouds
    from bleeding into each other.
    """
    n = len(fm)
    if n < 2:
        return np.zeros_like(GX)
    t0 = np.arange(n)
    t1 = np.linspace(0, n - 1, (n - 1) * sub + 1)
    fmi = np.interp(t1, t0, fm)
    fsi = np.interp(t1, t0, fsd)
    dmi = np.interp(t1, t0, dmu)
    dsi = np.interp(t1, t0, dsd)

    # vectorised: min squared Mahalanobis distance to the path, in chunks
    gx1 = GX[0, :]                      # grid is a meshgrid -> axes are separable
    gy1 = GY[:, 0]
    best = np.full((gy1.size, gx1.size), np.inf)
    CH = 64
    for s0 in range(0, len(t1), CH):
        sl = slice(s0, s0 + CH)
        dx = (gx1[None, :] - fmi[sl, None]) / np.maximum(fsi[sl, None], 1e-3)
        dy = (gy1[None, :] - dmi[sl, None]) / np.maximum(dsi[sl, None], 1e-4)
        d2 = dy[:, :, None] ** 2 + dx[:, None, :] ** 2      # (k, ny, nx)
        best = np.minimum(best, d2.min(axis=0))
    Z = np.exp(-0.5 * best)
    # max (not sum) keeps the tunnel a constant-density band along its length,
    # so it does not go dark merely where the path happens to dwell
    Z[Z < floor] = 0.0
    return Z


def true_path(V, dann, horizon, burn, true_window=None):
    """True exposure and demand (the oracle position).

    Default: forward `horizon`-year mean (the decision target). If `true_window`
    is set, the exposure LINE is instead a `true_window`-year CENTRED rolling
    mean, defined for every year (clamped at the edges) -- this de-noises the
    line for display when horizon=1 (a single year's flood volume whipsaws
    between ~0 and huge) WITHOUT changing the belief, which stays at `horizon`."""
    if true_window and true_window > 1:
        n = len(V); half = true_window // 2
        tf, td = [], []
        for t in range(burn, n):
            lo, hi = max(0, t - half), min(n, t + true_window - half)
            tf.append(np.log(max(V[lo:hi].mean() * ATLAS_YEARS, 1.0)))
            td.append(float(dann[lo:hi].mean()))
        return np.array(tf), np.array(td)
    tf, td = [], []
    for t in range(burn, len(V)):
        wv = V[t:t + horizon]
        wd = dann[t:t + horizon]
        if len(wv) < horizon:              # full window only (see handoff)
            break
        tf.append(np.log(max(wv.mean() * ATLAS_YEARS, 1.0)))
        td.append(float(wd.mean()))
    return np.array(tf), np.array(td)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="NML")
    ap.add_argument("--pairs", default=None,
                    help="explicit 'gcm:lulc,gcm:lulc,...' (up to 4). Default "
                         "is automatic farthest-point selection over BOTH axes.")
    ap.add_argument("--lulc", default=None,
                    help="force one demand scenario for every trajectory "
                         "(default: let the auto-selection vary demand too)")
    ap.add_argument("--exclude-gcm", default=None,
                    help="comma-separated substrings; GCMs whose name contains "
                         "any are dropped from auto-selection (e.g. 'miroc', "
                         "whose exposure collapses to zero).")
    ap.add_argument("--x-kappa", action="store_true",
                    help="plot atlas-equivalent kappa on x (via the synthetic "
                         "kappa<->vol_above_safe curve) instead of log volume, "
                         "to match the policy-atlas / three-strategies figures.")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--true-window", type=int, default=None,
                    help="smooth the TRUE-exposure line with a centred rolling "
                         "mean of this many years (belief/demand stay at "
                         "--horizon). Use with --horizon 1 to de-noise the "
                         "single-year truth line for display.")
    ap.add_argument("--burn-in", type=int, default=8)
    ap.add_argument("--n-particles", type=int, default=2000)
    ap.add_argument("--grid", type=int, default=200)
    ap.add_argument("--all-gcms", action="store_true",
                    help="allow trajectories whose exposure falls OUTSIDE atlas "
                         "support. Off by default: farthest-point sampling "
                         "otherwise picks the extremes, and at FOL/ORO the "
                         "extreme is the zero-exposure corner where no atlas "
                         "cell exists and the filter has no events to condition "
                         "on -- selecting the pathological case by construction.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb, LinearSegmentedColormap

    res = args.res
    cfg = reservoir_config(res)
    atlas = pd.read_csv(REPO / "results" / "find_atlas_fits.csv")
    atlas = atlas[atlas.res == res]
    if atlas.empty:
        print(f"{res}: no atlas fits"); return
    lx = np.log(np.maximum(atlas[COORD].to_numpy(), 1.0))
    ly = atlas["demand"].to_numpy()

    clim = scenario_names()
    lulc_all = sorted(p.name[:-8] for p in
                      (REPO / "scenario_data" / "lulc").glob("*.csv.zip"))

    if args.pairs:
        pairs = []
        for tok in args.pairs.split(",")[:4]:
            g, _, l = tok.strip().partition(":")
            pairs.append((g, l or (args.lulc or "GCAM-26")))
    else:
        # MEASURE both axes, then farthest-point sample. Picking scenarios by
        # index in the alphabetical list (the previous default) is arbitrary:
        # it gave four mid-range GCMs on ONE demand scenario, which is why the
        # clouds piled on top of each other. Exposure is cached per reservoir
        # because scanning every GCM is the slow part.
        import json
        ef = REPO / "results" / f"gcm_exposure_{res}.json"
        if ef.is_file():
            expo = json.loads(ef.read_text())
            # older caches stored a bare mean; recompute those
            if expo and not isinstance(next(iter(expo.values())), list):
                expo = {}
        else:
            expo = {}
            for cs in clim:
                q, _, d = load_inflow(cs, key=res)
                m = d.year.values >= 2020
                _, _, V = annual_events(q[m], d[m], cfg.r_max_cfs)
                tf = [np.log(max(V[t:t + args.horizon].mean() * ATLAS_YEARS, 1.0))
                      for t in range(args.burn_in, len(V) - args.horizon)]
                expo[cs] = [float(np.mean(tf)), float(np.min(tf)),
                            float(np.max(tf))]
            ef.parent.mkdir(exist_ok=True)
            ef.write_text(json.dumps(expo, indent=1))
        dem = {}
        for l in lulc_all:
            dm_, dt_ = load_demand(l)
            dem[l] = float(dm_[dt_.year.values >= 2020].mean())

        cand = [(g, l, expo[g][0], dem[l], expo[g][1], expo[g][2])
                for g in expo for l in
                ([args.lulc] if args.lulc else lulc_all)]
        if args.exclude_gcm:
            toks = [t.strip().lower() for t in args.exclude_gcm.split(",") if t.strip()]
            cand = [c for c in cand if not any(t in c[0].lower() for t in toks)]
            print(f"  excluded GCMs matching {toks}; {len(cand)} candidates remain")
        if not args.all_gcms:
            # keep only trajectories the atlas can actually serve. At FOL 63/97
            # and at ORO 90/97 GCMs are in support, so this is a scope
            # restriction, not cherry-picking -- but it IS a restriction and
            # belongs in the caption.
            sup = lx[lx > 0]
            if sup.size:
                lo_s, hi_s = float(sup.min()), float(lx.max())
                # require the WHOLE trajectory in support, not just its mean.
                # A mean-in-support trajectory can still spend years at exactly
                # zero exposure (ORO miroc-esm_rcp45: mean 4.45 but true
                # forward exposure 0 at both ends of the century), which the
                # log-scale belief cannot represent at all.
                keep = [c for c in cand if lo_s <= c[4] and c[5] <= hi_s]
                n_drop = len(cand) - len(keep)
                if keep:
                    if n_drop:
                        print(f"  ({n_drop}/{len(cand)} candidate pairs dropped: "
                              f"exposure outside atlas support "
                              f"[{lo_s:.2f}, {hi_s:.2f}])")
                    cand = keep
        ev = np.array([c[2] for c in cand]); dv = np.array([c[3] for c in cand])
        P = np.column_stack([(ev - ev.mean()) / max(ev.std(), 1e-9),
                             (dv - dv.mean()) / max(dv.std(), 1e-9)])
        # greedy farthest-point: start from the most extreme, then repeatedly
        # add the candidate furthest from everything already chosen
        sel = [int(np.argmax(np.linalg.norm(P, axis=1)))]
        while len(sel) < 4:
            d2 = np.min(np.linalg.norm(P[:, None, :] - P[None, sel, :], axis=2),
                        axis=1)
            sel.append(int(np.argmax(d2)))
        pairs = [(cand[i][0], cand[i][1]) for i in sel]
        print("auto-selected (flood exposure, mean demand):")
        for i in sel:
            print(f"  {cand[i][0]:32s} x {cand[i][1]:22s} "
                  f"({cand[i][2]:.2f}, {cand[i][3]:.3f})")

    # belief hyperparameters: fit once on OTHER trajectories, then CACHED.
    # fit_hyper is a particle-filter marginal-likelihood search and dominates
    # the runtime of this script; caching makes figure iteration interactive.
    import json
    hp_file = REPO / "results" / "belief_hyper.json"
    store = json.loads(hp_file.read_text()) if hp_file.is_file() else {}
    if res in store:
        a, q_lam, q_mag, phi = store[res]
        print(f"{res}: cached a={a:.2f} Q=({q_lam:.4f},{q_mag:.4f}) phi={phi:.1f}")
    else:
        train = []
        for cs in clim[:4]:
            q, _, d = load_inflow(cs, key=res)
            m = d.year.values >= 2020
            _, n, V = annual_events(q[m], d[m], cfg.r_max_cfs)
            train.append((n, V))
        a = fit_gamma_shape(train)
        q_lam, q_mag, phi = fit_hyper(train, a, n_part=1200)
        store[res] = [float(a), float(q_lam), float(q_mag), float(phi)]
        hp_file.parent.mkdir(exist_ok=True)
        hp_file.write_text(json.dumps(store, indent=1))
        print(f"{res}: fitted a={a:.2f} Q=({q_lam:.4f},{q_mag:.4f}) phi={phi:.1f}")

    runs = []
    for cs, ls in pairs:
        dm_full, ddates = load_demand(ls)
        md = ddates.year.values >= 2020
        wyl = ddates.year.values[md]
        ann_d = np.array([dm_full[md][wyl == y].mean() for y in np.unique(wyl)])
        q, _, dates = load_inflow(cs, key=res)
        m = dates.year.values >= 2020
        yrs, n, V = annual_events(q[m], dates[m], cfg.r_max_cfs)
        fm, fsd, _ = particle_filter(n, V, q_lam, q_mag, a, args.n_particles,
                                     np.random.default_rng(1), phi=phi)
        fm = fm + np.log(ATLAS_YEARS)
        T = min(len(fm), len(ann_d))
        dmu, dsd = demand_llt(ann_d[:T], H=args.horizon)
        b = args.burn_in
        tf, td = true_path(V[:T], ann_d[:T], args.horizon, b, args.true_window)
        runs.append(dict(scen=cs, lulc=ls, fm=fm[b:T], fsd=fsd[b:T],
                         dmu=dmu[b:T], dsd=dsd[b:T], tf=tf, td=td,
                         yrs=yrs[b:T]))
        print(f"  {cs} x {ls}: belief {fm[b]:.2f}->{fm[T-1]:.2f}, "
              f"true {tf[0]:.2f}->{tf[-1]:.2f}, demand {ann_d[b]:.2f}->"
              f"{ann_d[T-1]:.2f}")

    # ---- plotting grid spanning atlas + every belief ----------------------
    # Exclude the vol=0 atlas cells from the plotting range. FOL has 140 and
    # ORO 175 of them, all at log-coordinate 0, which stretched the axis from 0
    # to ~17 and left the left two-thirds of the panel empty. They are real
    # atlas points (the lower boundary of the validity band) but they carry no
    # information about where the trajectories live.
    lx_pos = lx[lx > 0]
    if lx_pos.size == 0:
        lx_pos = lx
    # Trim isolated low-exposure outliers too. At ORO the positive support
    # starts at 3.37 but the bulk of the atlas sits above ~9, so keying the
    # axis to the minimum still left a third of the panel empty. Use a low
    # percentile of the atlas AND never clip any trajectory: the axis is the
    # union of the populated atlas and everything actually plotted.
    traj = np.concatenate([r["fm"] for r in runs] + [r["tf"] for r in runs])
    lo_atlas = float(np.percentile(lx_pos, 2))

    # optional x-axis reparametrisation log(volume) -> log(equivalent kappa),
    # via the atlas kappa<->vol_above_safe curve (monotone; matches atlas figs).
    if args.x_kappa:
        kv = atlas.groupby("kappa")[COORD].median()
        ok = np.argsort(kv.index.values)
        ks_a = kv.index.values[ok].astype(float)
        lv_a = np.log(np.maximum(kv.values[ok], 1.0))
        kmn, kmx = float(ks_a.min()), float(ks_a.max())
        def xt(lv):                       # log-vol -> log(equiv kappa)
            return np.log(np.clip(np.interp(lv, lv_a, ks_a), kmn, kmx))
        def lv_of_logk(logk):             # display log-kappa -> log-vol
            return np.interp(logk, np.log(ks_a), lv_a)
    else:
        def xt(lv):
            return lv

    ally = np.concatenate([ly] + [r["dmu"] for r in runs])
    allxd = np.concatenate([xt(lx_pos[lx_pos >= lo_atlas]), xt(traj)])
    pad = 0.1 if args.x_kappa else 0.3
    gxd = np.linspace(allxd.min() - pad, allxd.max() + pad, args.grid)
    gy = np.linspace(ally.min() - 0.05, ally.max() + 0.05, args.grid)
    # cloud is a density in LOG-VOL; evaluate it at the log-vol matching each
    # display-x column so the picture is a faithful reparametrisation.
    gx_lv = lv_of_logk(gxd) if args.x_kappa else gxd
    GX, GY = np.meshgrid(gx_lv, gy)

    fig, ax = plt.subplots(figsize=(10.5, 7))
    # atlas support: faint grey points only, no heatmap
    ax.scatter(xt(lx), ly, s=7, c="0.55", alpha=0.35, edgecolors="none",
               zorder=1, label="atlas support")

    for r, col in zip(runs, COLOURS):
        Z = belief_cloud(GX, GY, r["fm"], r["fsd"], r["dmu"], r["dsd"])
        rgb = to_rgb(col)
        cmap = LinearSegmentedColormap.from_list(
            "c", [(rgb[0], rgb[1], rgb[2], 0.0), (rgb[0], rgb[1], rgb[2], 0.62)])
        Zm = np.ma.masked_where(Z <= 0, Z)
        ax.imshow(Zm, origin="lower", aspect="auto", cmap=cmap,
                  vmin=0.18, vmax=1.0, interpolation="bilinear",
                  extent=[gxd.min(), gxd.max(), gy.min(), gy.max()], zorder=2)
        tfx = xt(r["tf"])
        ax.plot(tfx, r["td"], "-", color="white", lw=3.6, zorder=3, alpha=0.75)
        ax.plot(tfx, r["td"], "-", color=col, lw=2.2, zorder=4,
                label=f'{r["scen"]}  x  {r["lulc"]}')
        ax.plot(tfx[0], r["td"][0], "o", color="white", mec=col, mew=2,
                ms=8, zorder=5)
        ax.plot(tfx[-1], r["td"][-1], "s", color=col, mec="k", ms=8, zorder=5)

    ax.set_xlim(gxd.min(), gxd.max())
    if args.x_kappa:
        tk = [t for t in (0.25, 0.5, 1, 2, 5) if gxd.min() <= np.log(t) <= gxd.max()]
        ax.set_xticks([np.log(t) for t in tk])
        ax.set_xticklabels([f"{t:g}" for t in tk])
        ax.set_xlabel("atlas-equivalent flood magnitude  κ  "
                      "(via synthetic κ↔volume curve)")
    else:
        ax.set_xlabel("log flood exposure  —  log(volume above safe release, af / 50 yr)")
    ax.set_ylabel(f"mean demand multiplier over the next {args.horizon} yr")
    line_desc = (f"true exposure ({args.true_window}-yr centred mean)"
                 if args.true_window and args.true_window > 1
                 else f"true forward-{args.horizon}yr exposure")
    ax.set_title(f"{res}: four GCMs on the policy atlas\n"
                 f"line = {line_desc},  shading = Bayesian belief "
                 f"(H={args.horizon})",
                 fontweight="bold")
    ax.grid(alpha=0.2, lw=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.text(0.985, 0.015, "○ start   ■ finish\nshaded tunnel = joint posterior "
            "(darker = higher density)",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            bbox=dict(fc="white", ec="0.7", alpha=0.9, boxstyle="round,pad=0.35"))

    out = Path(args.out or (REPO / "results" / f"belief_multi_{res}.png"))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
