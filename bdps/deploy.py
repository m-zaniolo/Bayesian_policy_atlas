#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/bayes_deploy.py -- deploy the WHOLE policy vector by Bayesian atlas
lookup, across every reservoir x GCM x demand combination, and score the
realized cost.

WHY THE WHOLE VECTOR, NOT x4
    The learnability screen (fit_find_atlas.py) asks whether each parameter is
    INDIVIDUALLY predictable as a smooth function of the exposure coordinate.
    Only x4 (and sometimes x5/x6) passes. That is NOT the same question as
    "which parameters should be deployed".

    Leave-one-out over the fitted atlas, median excess cost vs the true optimum:
        predict all 7 independently, then assemble .... 10.2%
        copy the WHOLE optimal vector from the atlas ...  3.1%
        nearest-neighbour vector, x4 overwritten ........  8.0%

    Assembling 7 independently-predicted parameters is 3x worse than keeping
    the vector intact, because the parameters INTERACT (x1/x2/x3 are coupled
    day-of-year breakpoints). A vector built from 7 marginally-correct
    predictions can be optimal for no exposure at all -- structurally the same
    error as hybrid_atlas.py's coordinate-wise median (HANDOFF §7.1).

    So the candidate actions here are WHOLE atlas policy vectors. A parameter
    that is individually "non-monotone" is still part of a coherent optimum at
    its own atlas point; we never predict it, we look it up alongside the rest.

METHODS COMPARED (all deployed on the same realized GCM trajectory)
    bayes    Bayes action over whole vectors:
                 x* = argmin_x  E_{theta~p(theta|data)} [ J(x, theta) ]
             The posterior is the calibrated particle filter over flood regime
             (bayes_rate.py) x the projected demand belief. Expectation is taken
             against a precomputed transfer matrix, so it is a real integral
             over the posterior, not a plug-in.
    plugin   whole vector at the atlas cell nearest the posterior MEAN
             (continuous adaptation, but discards the posterior spread)
    x4only   frozen initial vector with ONLY x4 replaced by the atlas value
             -- the project's original "predict x4, fix the rest"
    frozen   initial policy held for the whole century (no adaptation)

    bayes vs plugin isolates the value of the posterior; bayes vs x4only
    isolates the value of updating the whole vector; any vs frozen is the value
    of adapting at all.

PHASES
    A  per reservoir: candidate policies (one representative whole vector per
       (kappa, demand) atlas cell) + transfer matrix J[i,j] = cost of candidate
       i on atlas cell j. Computed ONCE per reservoir and cached to npz, since
       every trajectory reuses it.
    B  per (reservoir, GCM, demand): run the belief forward, choose a policy
       each year by each method, simulate with storage carried across the
       switch, accumulate realized cost.

USAGE
    python scripts/bayes_deploy.py --phase A -j 22
    python scripts/bayes_deploy.py --phase B -j 22
    python scripts/bayes_deploy.py --phase both -j 22          # all reservoirs
    python scripts/bayes_deploy.py --phase both -j 22 --res NML,DNP --n-climate 8
"""
from __future__ import annotations
import os
import sys
import time
import functools
import argparse
import multiprocessing as _mp
try:                              # 'fork' spawns workers reliably in detached
    _mp.set_start_method("fork")  # background jobs; macOS default 'spawn' hangs
except RuntimeError:              # there (worker re-exec stalls on Box/stdin).
    pass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))
sys.path.insert(0, str(REPO / "bdps"))

from data_io import (load_inflow, load_demand, reservoir_config,      # noqa: E402
                     scenario_names, CMIP5_DIR, LULC_DIR,
                     DEFAULT_NODES, DEFAULT_MEDIANS)
from model import simulate, simulate_blended, PARAM_NAMES             # noqa: E402
from belief import (annual_events, fit_gamma_shape, fit_hyper,    # noqa: E402
                        particle_filter)

from functools import lru_cache as _lru
# read GCM/LULC/config from a LOCAL mirror ($BPA_DATA_ROOT) -- Box file-opens are ~1s
# and occasionally hang, stalling the run (see bench_baselines note).
from paths import DATA_ROOT as _LOCAL   # configurable; see bdps/paths.py
# use the local mirror when present (fast), else fall back to the real Box paths
_HAVE_LOCAL = (_LOCAL / "reference" / "nodes.json").exists()
_CMIP5 = (_LOCAL / "cmip5") if _HAVE_LOCAL else CMIP5_DIR
_LULC = (_LOCAL / "lulc") if _HAVE_LOCAL else LULC_DIR
_NODES = (_LOCAL / "reference" / "nodes.json") if _HAVE_LOCAL else DEFAULT_NODES
_MEDIANS = (_LOCAL / "reference" / "historical_medians.csv") if _HAVE_LOCAL else DEFAULT_MEDIANS
# same multiplier applied to BOTH flood Q (set at task build) and demand Q (here),
# for the learning-rate sweep. 1.0 = calibrated baseline (no change).
DEMAND_Q_SCALE = 1.0
# when True, _run also returns per-year (year, method, J_flood, J_shortage) records
# under row["yearly"] -- for the cost-over-time analysis. Off by default.
EMIT_YEARLY = False
@_lru(maxsize=None)
def _cinflow(cs, res):
    return load_inflow(cs, key=res, cmip5_dir=_CMIP5)
@_lru(maxsize=None)
def _cdemand(ls):
    return load_demand(ls, lulc_dir=_LULC)
@_lru(maxsize=None)
def _ccfg(res):
    return reservoir_config(res, nodes_json=_NODES, medians_csv=_MEDIANS)

CFSD_TO_AF = 1.98347
COORD = "vol_above_safe_af"
ATLAS_YEARS = 50.0            # atlas scenarios are 50-yr records
CACHE = REPO / "results" / "deploy_cache"
START_YEAR = 2020


# ===========================================================================
# PHASE A -- candidates + transfer matrix (once per reservoir)
# ===========================================================================
def build_cells(atlas_res):
    """One representative WHOLE policy vector per (kappa, demand) atlas cell.

    The representative is the chronology with the MEDIAN objective, not the
    best: the best is the luckiest flood sequence, which is an optimistic
    outlier. Median is the typical optimum for that exposure.
    """
    rows = []
    for (kap, dem), g in atlas_res.groupby(["kappa", "demand"]):
        g = g.sort_values("J")
        rep = g.iloc[len(g) // 2]
        rows.append(dict(kappa=kap, demand=dem,
                         coord=float(g[COORD].mean()),
                         file=rep["file"],
                         x=np.array([rep[n] for n in PARAM_NAMES], float)))
    return rows


def _build_J_stationary(res, cfg, cells, X):
    """Original transfer matrix: policy i run for the WHOLE 50-yr record of cell
    j, from a full reservoir. J[i,j] is a stationary-world total cost. This is
    what shipped; kept for comparison. Its defect (HANDOFF §7.8): deployment
    holds a policy ~1 yr from an INHERITED storage state, never 50 yr from full,
    so this mis-ranks policies exactly where realised storage strays from full
    (i.e. after big flood years). See _build_J_deployment for the fix."""
    n = len(cells)
    J = np.empty((n, n))
    qcache = {}
    for j, c in enumerate(cells):
        f = str(REPO / "scenario_data" / "find_flood" / res / c["file"])
        if f not in qcache:
            qcache[f] = pd.read_csv(f)["inflow_cfs"].to_numpy(float)
        q = qcache[f]
        dowy = np.arange(q.size, dtype=np.int64) % 365   # FIND: exact, 0-based
        dm = np.full(q.size, c["demand"])
        for i in range(n):
            J[i, j] = simulate(q, X[i], cfg, dowy, DM=dm,
                               S0=cfg.S_avg[-1])["J"]
    return J


def _build_J_deployment(res, cfg, cells, X, window, nwin, burn):
    """Deployment-conditioned transfer matrix.

    J[i,j] = mean PER-YEAR cost of candidate i, applied for a `window`-year
    stretch on cell j's climate, STARTED FROM the storage that a reasonable
    operator would have left entering that stretch -- not a full reservoir.

    Two changes from the stationary matrix, both to match how a policy is
    actually used at deployment (HANDOFF §7.8):

      1. HORIZON: `window` years, not 50. A policy is re-selected roughly yearly
         by the belief, so it should be graded on a short outcome, not a 50-yr
         equilibrium it never reaches. A single year would be biased -- it
         rewards hoarding water this year and flooding next -- so the window is
         a few years to price that continuation.

      2. ENTERING STORAGE: sampled from a realistic distribution instead of
         fixed-full. We spin up cell j's OWN optimal policy (X[j]) through its
         record and record the storage entering each water year; those are the
         states the system genuinely visits under this climate, including the
         depleted post-flood states the stationary matrix never saw. Candidate i
         is then dropped into those states -- deployment inherits storage from
         whatever ran before, it does not get to start full.

    Cost is divided by `window` to stay on a per-year scale (a global constant,
    so it does not affect argmin selection; it just keeps J readable).
    """
    n = len(cells)
    J = np.empty((n, n))
    qcache = {}
    for j, c in enumerate(cells):
        f = str(REPO / "scenario_data" / "find_flood" / res / c["file"])
        if f not in qcache:
            qcache[f] = pd.read_csv(f)["inflow_cfs"].to_numpy(float)
        q = qcache[f]
        nyr = int(q.size // 365)
        dm_lvl = c["demand"]

        # spin-up: realistic entering-storage under the cell's own optimum
        S_enter = np.empty(nyr)
        S = cfg.S_avg[-1]
        for t in range(nyr):
            S_enter[t] = S
            yq = q[t * 365:(t + 1) * 365]
            dwy = np.arange(yq.size, dtype=np.int64) % 365
            out = simulate(yq, X[j], cfg, dwy,
                           DM=np.full(yq.size, dm_lvl), S0=S)
            S = float(out["storage"][-1])

        # window starts after burn-in that fit a full `window` stretch
        starts = np.arange(burn, max(burn + 1, nyr - window + 1))
        starts = starts[starts + window <= nyr]
        if starts.size == 0:
            starts = np.array([max(0, nyr - window)])
        if nwin and starts.size > nwin:
            starts = starts[np.linspace(0, starts.size - 1, nwin).astype(int)]

        wins = []
        for t in starts:
            wq = q[t * 365:(t + window) * 365]
            wd = np.arange(wq.size, dtype=np.int64) % 365
            wins.append((wq, wd, np.full(wq.size, dm_lvl), float(S_enter[t])))

        for i in range(n):
            tot = 0.0
            for wq, wd, wdm, S0 in wins:
                tot += float(simulate(wq, X[i], cfg, wd, DM=wdm, S0=S0)["J"])
            J[i, j] = tot / (len(wins) * window)
    return J


def phase_a(res, cache_dir=None, jmode="stationary",
            window=3, nwin=12, burn=3):
    """Candidate policies + transfer matrix J[i,j], cached per reservoir.

    jmode='stationary' reproduces the shipped matrix; jmode='deployment' scores
    each policy under deployment conditions (see _build_J_deployment)."""
    cfg = _ccfg(res)
    atlas = pd.read_csv(REPO / "results" / "find_atlas_fits.csv")
    atlas = atlas[atlas.res == res]
    if atlas.empty:
        return None
    cells = build_cells(atlas)
    n = len(cells)
    X = np.array([c["x"] for c in cells])
    if jmode == "deployment":
        J = _build_J_deployment(res, cfg, cells, X, window, nwin, burn)
    else:
        J = _build_J_stationary(res, cfg, cells, X)
    cdir = Path(cache_dir) if cache_dir else CACHE
    cdir.mkdir(parents=True, exist_ok=True)
    np.savez(cdir / f"{res}.npz", J=J, X=X,
             coord=np.array([c["coord"] for c in cells]),
             demand=np.array([c["demand"] for c in cells]),
             kappa=np.array([c["kappa"] for c in cells]))
    return res, n, jmode


# ===========================================================================
# PHASE B -- belief + deployment on a real trajectory
# ===========================================================================
def demand_llt(y, H=20, q_level=1e-6, q_slope=1e-6, R=1e-5):
    """Filtered level/slope + H-yr projection of mean demand (see handoff §1.2)."""
    T = len(y)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Hm = np.array([[1.0, 0.0]])
    Q = np.diag([q_level, q_slope])
    m = np.array([y[0], 0.0]); P = np.diag([1e-3, 1e-2])
    mu, sd = np.empty(T), np.empty(T)
    for t in range(T):
        m = F @ m; P = F @ P @ F.T + Q
        s = float(Hm @ P @ Hm.T) + R
        K = (P @ Hm.T) / s
        m = m + (K * (y[t] - float(Hm @ m))).ravel()
        P = P - K @ Hm @ P
        Fh = np.array([[1.0, H / 2.0], [0.0, 1.0]])
        mh = Fh @ m; Ph = Fh @ P @ Fh.T + Q * H
        mu[t] = mh[0]; sd[t] = np.sqrt(max(float(Hm @ Ph @ Hm.T), 1e-12))
    return mu, sd


def weights(coord, dem_cells, mu, sd, dnow, dsd):
    """p(atlas cell | data): Gaussian on log-coordinate x Gaussian on demand."""
    c = np.log(np.maximum(coord, 1.0))
    w = np.exp(-0.5 * ((c - mu) / max(sd, 1e-6)) ** 2)
    w = w * np.exp(-0.5 * ((dem_cells - dnow) / max(dsd, 1e-3)) ** 2)
    s = w.sum()
    return w / s if s > 0 else np.full(len(w), 1.0 / len(w))


def _run(task, cache_dir=None, switch_margin=0.0):
    res, cs, ls, hp, a, phi, horizon, cwidth = task
    try:
        cdir = Path(cache_dir) if cache_dir else CACHE
        z = np.load(cdir / f"{res}.npz")
        J, X = z["J"], z["X"]
        coord, dem_cells = z["coord"], z["demand"]
        cfg = _ccfg(res)

        q, dowy_all, dates = _cinflow(cs, res)
        m = dates.year.values >= START_YEAR
        # dowy_all MUST be masked with q/dates -- `sel` below is built on the
        # truncated length, so leaving it full-length raises IndexError.
        q, dowy_all, dates = q[m], dowy_all[m], dates[m]
        yrs, nev, V = annual_events(q, dates, cfg.r_max_cfs)
        fm, fsd, _ = particle_filter(nev, V, hp[0], hp[1], a, 1500,
                                     np.random.default_rng(1), phi=phi)
        fm = fm + np.log(ATLAS_YEARS)          # annual -> atlas 50-yr scale
        # HORIZON-CONSISTENT flood uncertainty. The flood regime is a driftless
        # random walk, so the mean forecast at any horizon = the filtered mean
        # (a martingale has no trend to extrapolate -- this is why the flood mean
        # is NOT projected forward, unlike demand). But a random walk's VARIANCE
        # grows ~linearly with the horizon, so the forecast sd must widen:
        #   horizon=1  -> extra=0  -> pure nowcast (matches demand at horizon=1)
        #   horizon=H  -> add (H-1)*(Q_lam+Q_mu) to the variance of log(lambda*mu)
        # A wider belief makes the Bayes action average over more atlas cells,
        # i.e. hedge more -- the honest treatment for a multi-year decision, and
        # the mechanism that pulls selection toward robust policies where the
        # atlas surface is flat (see DNP, handoff 7.10).
        fsd = np.sqrt(fsd ** 2 + max(horizon - 1, 0) * (hp[0] + hp[1]))
        # WIDTH RECALIBRATION: the filtered belief is under-dispersed (z_std>1
        # on historical). cwidth (fit on the pre-2020 record) inflates fsd so the
        # belief is honestly calibrated -> less mean-chasing, less switching,
        # tighter cost spread. cwidth=1.0 disables it (--no-calib-width).
        fsd = cwidth * fsd

        dm_full, ddates = _cdemand(ls)
        md = ddates.year.values >= START_YEAR
        wy = ddates.year.values[md]
        ann_d = np.array([dm_full[md][wy == y].mean() for y in np.unique(wy)])
        T = min(len(fm), len(ann_d), len(yrs))
        dmu, dsd = demand_llt(ann_d[:T], H=horizon,
                              q_level=1e-6 * DEMAND_Q_SCALE,
                              q_slope=1e-6 * DEMAND_Q_SCALE)

        # per-year index into the daily arrays
        wyd = dates.year.values + (dates.month.values >= 10)
        # choose a policy each year by each method, carry storage across.
        # 'blend' = belief-weighted AVERAGE ACTION (posterior-expected release);
        # smooth in the belief so no policy hops -> much lower cost variance than
        # the argmin 'bayes' at moderate-signal reservoirs (NML/EXC), while also
        # improving the median. The proper Bayesian decision (see model.simulate_blended).
        methods = ["bayes", "blend", "plugin", "x4only", "frozen"]
        S = {k: cfg.S_avg[-1] for k in methods}
        cost = {k: 0.0 for k in methods}
        cost_fl = {k: 0.0 for k in methods}     # flood component
        cost_sh = {k: 0.0 for k in methods}     # shortage/demand component
        x_frozen = None
        i_dep = None                            # currently-deployed bayes cell
        n_switch = 0                            # count of deployed-policy changes
        yearly = []                             # per-year cost stream (if EMIT_YEARLY)
        burn = 8                                # filter burn-in (see handoff)
        for t in range(burn, T):
            yr = yrs[t]
            sel = wyd == yr
            if not sel.any():
                continue
            qq = q[sel]
            dowy = dowy_all[sel]          # leap-aware; NOT (arange % 365)
            dmv = np.full(qq.size, ann_d[t])

            w = weights(coord, dem_cells, fm[t], fsd[t], dmu[t], dsd[t])
            Jw = J @ w
            i_star = int(np.argmin(Jw))                        # Bayes action
            # SWITCHING HYSTERESIS: adapt continuously but only ACT when the best
            # candidate beats the currently-deployed policy in EXPECTED cost by
            # more than `switch_margin`. Damps the mean-jitter-driven policy churn
            # that manufactures cost variance (NML 43%/yr) without blocking real
            # regime shifts. switch_margin=0 recovers the plain Bayes action.
            if i_dep is None or Jw[i_star] < Jw[i_dep] * (1.0 - switch_margin):
                if i_dep is not None and i_star != i_dep:
                    n_switch += 1                      # deployed-policy change
                i_dep = i_star
            i_bayes = i_dep
            c = np.log(np.maximum(coord, 1.0))
            i_plug = int(np.argmin(np.abs(c - fm[t])
                                   + 10.0 * np.abs(dem_cells - dmu[t])))
            if x_frozen is None:
                x_frozen = X[i_plug].copy()                     # deploy-at-start

            pick = {"bayes": X[i_bayes], "plugin": X[i_plug],
                    "x4only": None, "frozen": x_frozen}
            xv = x_frozen.copy(); xv[4] = X[i_plug][4]
            pick["x4only"] = xv

            for k in methods:
                if k == "blend":
                    out = simulate_blended(qq, X, w, cfg, dowy, DM=dmv, S0=S[k])
                else:
                    out = simulate(qq, pick[k], cfg, dowy, DM=dmv, S0=S[k])
                cost[k] += float(out["J"])
                cost_fl[k] += float(out["J_flood"])
                cost_sh[k] += float(out["J_shortage"])
                S[k] = float(out["storage"][-1])
                if EMIT_YEARLY:
                    yearly.append((int(yr), k, float(out["J_flood"]),
                                   float(out["J_shortage"])))

        row = dict(res=res, clim=cs, lulc=ls, years=T - burn,
                   post_sd=float(np.mean(fsd[burn:T])),
                   n_switch=n_switch, switch_rate=n_switch / max(T - burn, 1))
        for k in methods:
            row[f"J_{k}"] = cost[k]
            row[f"J_{k}_flood"] = cost_fl[k]
            row[f"J_{k}_short"] = cost_sh[k]
        if EMIT_YEARLY:
            row["yearly"] = yearly
        return row
    except Exception as e:
        return dict(res=res, clim=cs, lulc=ls, error=repr(e)[:300])


def lulc_names(limit):
    d = REPO / "scenario_data" / "lulc"
    n = sorted(p.name[:-8] for p in d.glob("*.csv.zip"))
    return n[:limit]


def historical_obs(res, before=START_YEAR):
    """Flood observations (n_y, V_y) from the PRE-`before` reference record.

    Used to calibrate the belief hyperparameters (a, phi, Q) WITHOUT touching the
    GCM projection ensemble that is used for evaluation -- no data leakage. The
    reference record data/find_input_<res>.csv is a single trajectory disjoint
    from the 97 CMIP5 members (its 1951-2005 statistics differ from every
    member); its pre-2020 span is the history an operator would actually hold at
    deployment time. This is the honest, leak-free calibration source.
    """
    cfg = _ccfg(res)
    r = pd.read_csv(REPO / "data" / f"find_input_{res}.csv")
    dates = pd.DatetimeIndex(pd.to_datetime(r["date"]))
    q = r["inflow_cfs"].to_numpy(float)
    m = dates.year.values < before
    _, n, V = annual_events(q[m], dates[m], cfg.r_max_cfs)
    return n, V


def calibrate_belief_width(res, ql, qm, a, phi, H, before=START_YEAR):
    """Empirical width recalibration factor c for the flood belief.

    Diagnosis (spread analysis, 2026-07-25): the particle-filter posterior on the
    flood coordinate is systematically UNDER-DISPERSED -- the true forward-H-yr
    exposure lands outside fm +- fsd too often (std of the standardised error
    z=(fm-true)/fsd is 1.2-5.0, i.e. the belief is 1.2-5x too confident). An
    over-confident belief makes argmin(J w) chase the jittery posterior mean and
    switch policies too often, manufacturing ~2x the cost variance of a single
    fixed policy (NML switches 43%/yr, spread 215 vs 105).

    Fix: on the PRE-2020 reference record (leak-free, same source as the
    hyperparameters), run the belief, compare its mean to the realised
    forward-H-yr coordinate, and return c = std(z). Deployment then uses
    fsd -> c * fsd, making the belief honestly calibrated (z_std ~ 1). A wider,
    calibrated belief averages over more atlas cells, so it stops chasing the
    jitter -> less switching -> tighter cost distribution, and it pins itself
    toward the robust policy where data is too sparse to localise (no gate).

    Clamped to c >= 1: only ever WIDEN (the belief is never too wide in practice,
    and narrowing would sharpen the switching).
    """
    cfg = _ccfg(res)
    r = pd.read_csv(REPO / "data" / f"find_input_{res}.csv")
    dates = pd.DatetimeIndex(pd.to_datetime(r["date"]))
    q = r["inflow_cfs"].to_numpy(float)
    m = dates.year.values < before
    q, dates = q[m], dates[m]
    _, nev, V = annual_events(q, dates, cfg.r_max_cfs)
    fm, fsd, _ = particle_filter(nev, V, ql, qm, a, 1500,
                                 np.random.default_rng(1), phi=phi)
    fm = fm + np.log(ATLAS_YEARS)
    fsd = np.sqrt(fsd ** 2 + max(H - 1, 0) * (ql + qm))
    T = len(fm)
    tf = np.full(T, np.nan)                       # realised forward-H-yr coord
    for k in range(T):
        w = V[k:k + H]
        if len(w) >= max(3, H // 2):
            tf[k] = np.log(max(w.mean() * ATLAS_YEARS, 1.0))
    ok = np.isfinite(tf)
    z = (fm[ok] - tf[ok]) / np.maximum(fsd[ok], 1e-6)
    z = z[3:] if len(z) > 6 else z                # drop filter burn-in
    c = float(np.std(z)) if len(z) > 2 else 1.0
    return max(c, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="both", choices=["A", "B", "both"])
    ap.add_argument("--res", default=None)
    ap.add_argument("--n-climate", type=int, default=30)
    ap.add_argument("--n-demand", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("-j", "--workers", type=int, default=8)
    ap.add_argument("--out", default=str(REPO / "results" / "bayes_deploy.csv"))
    # --- transfer-matrix (phase A) controls ---
    ap.add_argument("--cache-dir", default=None,
                    help="where to read/write the J cache (default: "
                         "outputs/deploy_cache). Use a separate dir for the "
                         "deployment-conditioned J so the stationary one is kept.")
    ap.add_argument("--j-mode", default="stationary",
                    choices=["stationary", "deployment"],
                    help="how J is built. 'deployment' = per-year cost over a "
                         "multi-year window from realistic entering storage.")
    ap.add_argument("--window", type=int, default=3,
                    help="deployment J: window length in years")
    ap.add_argument("--n-windows", type=int, default=12,
                    help="deployment J: window starts sampled per cell")
    ap.add_argument("--spinup-burn", type=int, default=3,
                    help="deployment J: spin-up years discarded before sampling")
    ap.add_argument("--calib", default="historical",
                    choices=["historical", "gcm"],
                    help="belief hyperparameter calibration source. 'historical' "
                         "(default) = pre-2020 reference record, NO GCM leakage; "
                         "'gcm' = 4 test GCMs (leaky, comparison only).")
    ap.add_argument("--calib-width", dest="calib_width", action="store_true",
                    default=True,
                    help="recalibrate the flood-belief width on historical so it "
                         "is honestly dispersed (default ON; fixes over-switching).")
    ap.add_argument("--no-calib-width", dest="calib_width", action="store_false",
                    help="disable width recalibration (cwidth=1; for comparison).")
    ap.add_argument("--switch-margin", type=float, default=0.0,
                    help="hysteresis: only switch the deployed policy when a new "
                         "one beats it in expected cost by this fraction (e.g. "
                         "0.05). Damps mean-jitter policy churn. 0 = plain Bayes.")
    args = ap.parse_args()

    atlas = pd.read_csv(REPO / "results" / "find_atlas_fits.csv")
    reservoirs = ([r.strip() for r in args.res.split(",")] if args.res
                  else sorted(atlas.res.unique()))

    cdir = args.cache_dir or str(CACHE)
    if args.phase in ("A", "both"):
        print(f"PHASE A ({args.j_mode}): transfer matrices for "
              f"{len(reservoirs)} reservoirs on {args.workers} workers "
              f"-> {cdir}", flush=True)
        if args.j_mode == "deployment":
            print(f"  window={args.window}yr  n_windows={args.n_windows}  "
                  f"spinup_burn={args.spinup_burn}", flush=True)
        pa = functools.partial(phase_a, cache_dir=args.cache_dir,
                               jmode=args.j_mode, window=args.window,
                               nwin=args.n_windows, burn=args.spinup_burn)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=min(args.workers,
                                                 len(reservoirs))) as pool:
            for r in pool.map(pa, reservoirs):
                if r:
                    print(f"  {r[0]}: {r[1]}x{r[1]} matrix ({r[2]})", flush=True)
        print(f"PHASE A done in {time.time()-t0:.0f}s", flush=True)

    if args.phase in ("B", "both"):
        clim = scenario_names()[:args.n_climate]
        lul = lulc_names(args.n_demand)
        print(f"\nPHASE B: {len(reservoirs)} res x {len(clim)} GCM x "
              f"{len(lul)} demand = {len(reservoirs)*len(clim)*len(lul)} "
              f"trajectories on {args.workers} workers", flush=True)

        # belief hyperparameters: fit ONCE per reservoir, then applied to every
        # (held-out) trajectory. calib='historical' (default) fits on the pre-2020
        # reference record -- NO leakage from the GCM evaluation ensemble.
        # calib='gcm' reproduces the old behaviour (fit on 4 test GCMs) for
        # comparison only.
        print(f"  belief calibration source: {args.calib.upper()}"
              + ("  (pre-2020 reference record, leak-free)"
                 if args.calib == "historical" else
                 "  (4 GCMs -- has test-set leakage, comparison only)"), flush=True)
        hyper = {}
        for res in reservoirs:
            cfg = _ccfg(res)
            if args.calib == "historical":
                train = [historical_obs(res)]
            else:
                train = []
                for cs in clim[:4]:
                    q, dowy_all, dates = _cinflow(cs, res)
                    m = dates.year.values >= START_YEAR
                    _, n, V = annual_events(q[m], dates[m], cfg.r_max_cfs)
                    train.append((n, V))
            a = fit_gamma_shape(train)
            ql, qm, phi = fit_hyper(train, a, n_part=1200)
            cwidth = (calibrate_belief_width(res, ql, qm, a, phi, args.horizon)
                      if args.calib_width else 1.0)
            hyper[res] = (ql, qm, a, phi, cwidth)
            print(f"  {res}: a={a:.2f} Q=({ql:.4f},{qm:.4f}) phi={phi:.1f} "
                  f"cwidth={cwidth:.2f}  (n_hist_years={len(train[0][0])})",
                  flush=True)

        tasks = [(res, cs, ls, (hyper[res][0], hyper[res][1]), hyper[res][2],
                  hyper[res][3], args.horizon, hyper[res][4])
                 for res in reservoirs for cs in clim for ls in lul]
        rn = functools.partial(_run, cache_dir=args.cache_dir,
                               switch_margin=args.switch_margin)
        rows, t0 = [], time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, r in enumerate(pool.map(rn, tasks), 1):
                if "error" in r:
                    print(f"  FAIL {r['res']}/{r['clim']}: {r['error']}",
                          flush=True)
                else:
                    rows.append(r)
                if i % 50 == 0:
                    print(f"  {i}/{len(tasks)} ({time.time()-t0:.0f}s)",
                          flush=True)

        D = pd.DataFrame(rows)
        # --- non-destructive write (see memory/bayes-deploy-destructive-write) ---
        # A partial run (--res subset, or a smaller --n-demand/--n-climate) must
        # NOT silently drop the reservoirs it didn't compute. Back up the prior
        # file and MERGE: rows this run produced replace their (res,clim,lulc)
        # keys, everything else is preserved.
        out_path = Path(args.out)
        if out_path.exists():
            bak = out_path.with_suffix(out_path.suffix + ".bak")
            try:
                prev = pd.read_csv(out_path)
                prev.to_csv(bak, index=False)
                key = ["res", "clim", "lulc"]
                new_keys = set(map(tuple, D[key].values))
                keep = prev[~prev[key].apply(tuple, axis=1).isin(new_keys)]
                if len(keep):
                    D = pd.concat([keep, D], ignore_index=True)
                    print(f"  merged {len(keep)} preserved rows from prior "
                          f"{out_path.name} (backup -> {bak.name})", flush=True)
            except Exception as e:
                print(f"  WARNING: could not merge prior {out_path.name}: {e}",
                      flush=True)
        D.to_csv(args.out, index=False)
        print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {args.out}\n")

        print("Realized cost vs FROZEN (negative = adaptation is cheaper), "
              "cost-weighted per reservoir\n")
        print(f"{'res':>4} {'n':>5} {'bayes':>9} {'plugin':>9} {'x4only':>9}"
              f"  {'bayes vs x4only':>16}")
        for res, g in D.groupby("res"):
            f = g.J_frozen.sum()
            b, p, x = g.J_bayes.sum(), g.J_plugin.sum(), g.J_x4only.sum()
            print(f"{res:>4} {len(g):5d} {100*(b-f)/f:+8.2f}% "
                  f"{100*(p-f)/f:+8.2f}% {100*(x-f)/f:+8.2f}%  "
                  f"{100*(b-x)/x:+15.2f}%")
        f = D.J_frozen.sum()
        print(f"\n{'ALL':>4} {len(D):5d} {100*(D.J_bayes.sum()-f)/f:+8.2f}% "
              f"{100*(D.J_plugin.sum()-f)/f:+8.2f}% "
              f"{100*(D.J_x4only.sum()-f)/f:+8.2f}%  "
              f"{100*(D.J_bayes.sum()-D.J_x4only.sum())/D.J_x4only.sum():+15.2f}%")


if __name__ == "__main__":
    main()
