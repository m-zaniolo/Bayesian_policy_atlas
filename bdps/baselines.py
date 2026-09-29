#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/bench_baselines.py -- the three benchmarks the Bayesian atlas must beat.

Scored on the SAME trajectories, with the SAME year-by-year segmentation and
storage chaining as bayes_deploy.py, so the numbers join directly on
(res, clim, lulc) and are comparable without re-scaling.

    frozen2020   DE-optimise on the trailing 50-yr window ending at 2020, then
                 hold that policy for the whole century. The "no adaptation"
                 floor and the denominator everything else is quoted against.

    ff_15_50     THE PUBLISHED BASELINE (Sunkara & Herman): re-optimise every
                 15 years on the trailing 50-yr window. This is the scheme the
                 whole project is trying to improve on, so it -- not frozen --
                 is the comparison that carries the scientific claim.
                 Its first reopt (year 2020) IS the frozen2020 policy, so one
                 DE serves both baselines.

    oracle       HINDSIGHT-OPTIMAL POLICY SEQUENCE over the atlas candidate
                 set, found by BACKWARD coordinate descent on the realised
                 trajectory. NO UNCERTAINTY ANYWHERE -- it is pure hindsight
                 optimisation, not a belief.

                 Why backward: choosing year t's policy changes the storage
                 handed to t+1..end, so a forward/greedy choice is myopic (that
                 version measured WORSE than the Bayesian arm). Sweeping from
                 the last year backwards means the future is already optimised
                 when year t is chosen, so each decision is made against a good
                 continuation. Repeated sweeps let earlier choices react to
                 later ones.

                 It is initialised from the point lookup at the TRUE regime
                 exposure and only ever accepts improvements, so by
                 construction it is >= that initialisation, and in practice
                 >= every method compared. That makes it a genuine (if not
                 provably global) ceiling over the candidate set -- the same
                 greedy+local-search construction the project's earlier timing
                 oracle used.

                 It is NOT full DP: no value function, no storage
                 discretisation. Cost is n_sweeps x n_years x n_candidates
                 tail-simulations.

COST
    ~23 s per 50-yr DE at DE_TOL=0.1/trials=10 (measured). 6 reopts per
    trajectory x 9 res x 30 GCM x 8 demand = 12,960 DE ~ 83 core-hours ~ 3.8 h
    on 22 cores. The DE results are cached to disk by
    (res, gcm, demand, year), so an interrupted run resumes for free.

USAGE
    python scripts/bench_baselines.py -j 22 --n-climate 30 --n-demand 8
    python scripts/bench_baselines.py -j 22 --n-demand 3      # cheaper subset
"""
from __future__ import annotations
import os
import sys
import time
import pickle
import multiprocessing as _mp
try:                              # 'fork' spawns workers reliably in detached
    _mp.set_start_method("fork")  # background jobs; macOS default 'spawn' hangs
except RuntimeError:              # there (worker re-exec stalls on Box/stdin).
    pass
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))
sys.path.insert(0, str(REPO / "bdps"))

from data_io import (load_inflow, load_demand, reservoir_config,      # noqa: E402
                     scenario_names)
from model import simulate                                            # noqa: E402
from optimize import optimize_policy                                  # noqa: E402
from belief import annual_events                                  # noqa: E402
# NOTE: no belief/particle-filter import -- the oracle is pure hindsight.

from functools import lru_cache as _lru
# Box CloudStorage file-opens are ~1s each and occasionally hang for minutes, so
# read GCM/LULC/config from a LOCAL mirror at $BPA_DATA_ROOT (populated by a one-time
# copy) and cache each file per process. This is the ONLY reliable way to run at
# scale here; Box in the hot loop stalls the run indefinitely.
from paths import DATA_ROOT as _LOCAL   # configurable; see bdps/paths.py
@_lru(maxsize=None)
def _cinflow(cs, res):
    return load_inflow(cs, key=res, cmip5_dir=_LOCAL / "cmip5")
@_lru(maxsize=None)
def _cdemand(ls):
    return load_demand(ls, lulc_dir=_LOCAL / "lulc")
@_lru(maxsize=None)
def _ccfg(res):
    return reservoir_config(res, nodes_json=_LOCAL / "reference" / "nodes.json",
                            medians_csv=_LOCAL / "reference" / "historical_medians.csv")

START_YEAR = 2020
WINDOW_YEARS = 50           # paper's trailing window
FREQ_YEARS = 15             # paper's reoptimisation frequency
BURN = 8                    # match bayes_deploy's filter burn-in exactly
HORIZON = 20                # forward window the policy must suit (bayes_deploy)
ATLAS_YEARS = 50.0          # atlas scenarios are 50-yr records
DECACHE = REPO / "results" / "de_cache"
ACACHE = _LOCAL / "deploy_cache"

_TOL = float(os.environ.get("DE_TOL", 0.1))
_TRIALS = int(os.environ.get("DE_TRIALS", 10))


def oracle_sequence(years_q, years_dm, years_dowy, Xc, cfg, S0, init_idx, n_sweeps=2):
    """Hindsight-optimal policy sequence by BACKWARD coordinate descent.

    years_q/years_dm/years_dowy : per-year daily inflow, demand and LEAP-AWARE
                   day-of-water-year arrays (realised). dowy is passed in rather
                   than recomputed as (arange % 365) because the record carries
                   real dates -- see data_io.dowy_from_dates / HANDOFF §7.6.
    init_idx         : starting candidate index per year
    Returns (total_cost, seq).

    At each year t (sweeping backwards) every candidate is scored by simulating
    t -> end with the later years held at their current choices, starting from
    the storage the prefix actually produces. Only strict improvements are
    kept, so the result can never be worse than `init_idx`.
    """
    T = len(years_q)
    seq = list(init_idx)
    nC = Xc.shape[0]

    def tail_cost(t, S_start, seq_local):
        c, S = 0.0, S_start
        for u in range(t, T):
            qq = years_q[u]
            o = simulate(qq, Xc[seq_local[u]], cfg, years_dowy[u],
                         DM=years_dm[u], S0=S)
            c += float(o["J"]); S = float(o["storage"][-1])
        return c

    for _ in range(n_sweeps):
        # storage entering each year under the current sequence
        S_at, S = [], S0
        for u in range(T):
            S_at.append(S)
            qq = years_q[u]
            o = simulate(qq, Xc[seq[u]], cfg, years_dowy[u],
                         DM=years_dm[u], S0=S)
            S = float(o["storage"][-1])

        changed = False
        for t in range(T - 1, -1, -1):
            best_c, best_i = np.inf, seq[t]
            trial = list(seq)
            for i in range(nC):
                trial[t] = i
                c = tail_cost(t, S_at[t], trial)
                if c < best_c:
                    best_c, best_i = c, i
            if best_i != seq[t]:
                seq[t] = best_i
                changed = True
        if not changed:
            break

    return tail_cost(0, S0, seq), seq


def de_policy(res, cs, ls, year, q_all, dowy_all, dates_all, dm_all, cfg):
    """Trailing-50yr DE at `year`, cached on disk (resumable)."""
    DECACHE.mkdir(parents=True, exist_ok=True)
    f = DECACHE / f"{res}_{cs}_{ls}_{year}.pkl"
    if f.is_file():
        try:
            with open(f, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass                      # corrupt cache entry -> recompute
    yv = dates_all.year.values
    a1 = int(np.argmax(yv >= year))
    a0 = max(0, a1 - WINDOW_YEARS * 365)
    x, _ = optimize_policy(q_all[a0:a1], cfg, dowy_all[a0:a1],
                           DM=dm_all[a0:a1], S0=cfg.S_avg[-1],
                           maxiter=100000, trials=_TRIALS, tol=_TOL,
                           polish=True)
    with open(f, "wb") as fh:
        pickle.dump(x, fh)
    return x


def _run(task):
    res, cs, ls, oh, nsw, no_oracle = task
    try:
        cfg = _ccfg(res)
        z = np.load(ACACHE / f"{res}.npz")
        Xc = z["X"]                                   # atlas candidate policies
        Jt = z["J"]                                   # transfer matrix (same as bayes)
        coord_c = np.log(np.maximum(z["coord"], 1.0))
        dem_c = z["demand"]

        q_all, dowy_all, dates_all = _cinflow(cs, res)
        dm_all, ddates = _cdemand(ls)
        n = min(q_all.size, dm_all.size)
        q_all, dowy_all, dates_all, dm_all = (q_all[:n], dowy_all[:n],
                                              dates_all[:n], dm_all[:n])
        yv = dates_all.year.values
        wyd = yv + (dates_all.month.values >= 10)

        # --- the reopt schedule: 2020, 2035, ... (first one = frozen policy)
        yrs_future = np.unique(wyd[wyd >= START_YEAR])
        yrs_future = yrs_future[BURN:]                # align with bayes_deploy
        reopt_years = list(range(START_YEAR, int(yrs_future[-1]) + 1, FREQ_YEARS))

        # TRUE forward exposure per year, for the perfect-information oracle
        mfut = yv >= START_YEAR
        _, _, Vann = annual_events(q_all[mfut], dates_all[mfut], cfg.r_max_cfs)
        wy_f = wyd[mfut]
        dann = np.array([dm_all[mfut][wy_f == y].mean()
                         for y in np.unique(wy_f)])

        uy = np.unique(wy_f)
        tf_raw = np.full(len(uy), np.nan)
        td_raw = np.full(len(uy), np.nan)
        for k in range(len(uy)):
            wv = Vann[k:k + oh]; wd = dann[k:k + max(oh, 1)]
            if len(wv) >= 1:
                tf_raw[k] = np.log(max(wv.mean() * ATLAS_YEARS, 1.0))
                td_raw[k] = float(wd.mean())
        ok = ~np.isnan(tf_raw)
        last = np.where(ok)[0][-1] if ok.any() else 0
        tf_raw[~ok] = tf_raw[last]; td_raw[~ok] = td_raw[last]
        tf_s, td_s = tf_raw, td_raw          # no smoothing, no belief involved

        x_frozen = de_policy(res, cs, ls, START_YEAR, q_all, dowy_all,
                             dates_all, dm_all, cfg)
        ff_cache = {START_YEAR: x_frozen}

        # 'truelookup' = the atlas policy at the TRUE exposure coordinate each
        # year (perfect belief, same nearest-cell rule). Scored here so its
        # flood/demand split is available alongside the others.
        methods = ["frozen2020", "ff_15_50", "truelookup", "oracle"]
        S = {k: cfg.S_avg[-1] for k in methods}
        cost = {k: 0.0 for k in methods}
        cost_fl = {k: 0.0 for k in methods}     # flood component
        cost_sh = {k: 0.0 for k in methods}     # shortage/demand component
        yq, ydm, ydowy, init_seq = [], [], [], []

        for yr in yrs_future:
            sel = wyd == yr
            if not sel.any():
                continue
            qq = q_all[sel]
            dowy = dowy_all[sel]          # leap-aware; NOT (arange % 365)
            dmv = dm_all[sel]

            # f=15/w=50: newest reopt at or before this year
            ry = max([r for r in reopt_years if r <= yr])
            if ry not in ff_cache:
                ff_cache[ry] = de_policy(res, cs, ls, ry, q_all, dowy_all,
                                         dates_all, dm_all, cfg)
            pick = {"frozen2020": x_frozen, "ff_15_50": ff_cache[ry]}
            for k in ("frozen2020", "ff_15_50"):
                out = simulate(qq, pick[k], cfg, dowy, DM=dmv, S0=S[k])
                cost[k] += float(out["J"]); cost_fl[k] += float(out["J_flood"])
                cost_sh[k] += float(out["J_shortage"]); S[k] = float(out["storage"][-1])

            # stash the realised year for the oracle's backward pass, and a
            # HARD point-lookup at the true regime exposure as its starting
            # sequence. No posterior, no spread, no smoothing anywhere.
            yq.append(qq); ydm.append(dmv); ydowy.append(dowy)
            ti = min(int(np.searchsorted(uy, yr)), len(tf_s) - 1)
            i_true = int(np.argmin(np.abs(coord_c - tf_s[ti])
                                   + 10.0 * np.abs(dem_c - td_s[ti])))
            init_seq.append(i_true)
            out = simulate(qq, Xc[i_true], cfg, dowy, DM=dmv, S0=S["truelookup"])
            cost["truelookup"] += float(out["J"])
            cost_fl["truelookup"] += float(out["J_flood"])
            cost_sh["truelookup"] += float(out["J_shortage"])
            S["truelookup"] = float(out["storage"][-1])

        # ---- oracle: hindsight-optimal sequence, backward coordinate descent
        # (the SLOW step; --no-oracle skips it when only frozen/ff/truelookup
        # components are needed).
        if not no_oracle:
            cost["oracle"], oseq = oracle_sequence(yq, ydm, ydowy, Xc, cfg,
                                                   cfg.S_avg[-1], init_seq,
                                                   n_sweeps=nsw)
            So = cfg.S_avg[-1]
            for t in range(len(oseq)):
                out = simulate(yq[t], Xc[oseq[t]], cfg, ydowy[t], DM=ydm[t], S0=So)
                cost_fl["oracle"] += float(out["J_flood"])
                cost_sh["oracle"] += float(out["J_shortage"])
                So = float(out["storage"][-1])
            osw = int(np.sum(np.asarray(oseq[1:]) != np.asarray(oseq[:-1])))
        else:
            cost["oracle"] = cost_fl["oracle"] = cost_sh["oracle"] = float("nan")
            osw = 0
        row = dict(res=res, clim=cs, lulc=ls, n_reopt=len(ff_cache),
                   oracle_horizon=oh, oracle_switches=osw)
        for k in methods:
            row[f"J_{k}"] = cost[k]
            row[f"J_{k}_flood"] = cost_fl[k]
            row[f"J_{k}_short"] = cost_sh[k]
        return row
    except Exception as e:
        return dict(res=res, clim=cs, lulc=ls, error=repr(e)[:300])


def lulc_names(limit):
    d = REPO / "scenario_data" / "lulc"
    return sorted(p.name[:-8] for p in d.glob("*.csv.zip"))[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=None)
    ap.add_argument("--n-climate", type=int, default=30)
    ap.add_argument("--n-demand", type=int, default=8)
    ap.add_argument("--oracle-sweeps", type=int, default=2,
                    help="backward coordinate-descent sweeps for the oracle. "
                         "Each sweep can only improve on the last.")
    ap.add_argument("--oracle-horizon", type=int, default=20,
                    help="years of TRUE exposure the oracle sees. 20 (default) "
                         "= perfect knowledge of the REGIME. Do NOT set 1: "
                         "perfect single-year foresight measured +102%% vs "
                         "frozen against +1%% at 20 yr, because a policy chosen "
                         "for a known dry year empties the flood pool and hands "
                         "a full reservoir to the next year -- perfect "
                         "information whipsaws the storage state when the "
                         "decision cadence is shorter than the state's memory.")
    ap.add_argument("-j", "--workers", type=int, default=8)
    ap.add_argument("--no-oracle", action="store_true",
                    help="skip the (slow) oracle coordinate-descent; J_oracle=NaN. "
                         "Use when only frozen/ff/truelookup costs are needed.")
    ap.add_argument("--out", default=str(REPO / "results" / "bench_baselines.csv"))
    ap.add_argument("--join", default=str(REPO / "results" / "bayes_deploy.csv"),
                    help="bayes_deploy output to merge with for the final table")
    args = ap.parse_args()

    atlas = pd.read_csv(REPO / "results" / "find_atlas_fits.csv")
    reservoirs = ([r.strip() for r in args.res.split(",")] if args.res
                  else sorted(atlas.res.unique()))
    missing = [r for r in reservoirs if not (ACACHE / f"{r}.npz").is_file()]
    if missing:
        print(f"missing transfer caches for {missing}; run "
              f"bayes_deploy.py --phase A first")
        return

    clim = scenario_names()[:args.n_climate]
    lul = lulc_names(args.n_demand)

    tasks = [(r, c, l, args.oracle_horizon, args.oracle_sweeps, args.no_oracle)
             for r in reservoirs for c in clim for l in lul]
    print(f"DE settings: tol={_TOL} trials={_TRIALS}")
    print(f"{len(tasks)} trajectories on {args.workers} workers "
          f"(~{len(tasks)*6*23/3600/args.workers:.1f} h if no cache)\n", flush=True)

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, r in enumerate(pool.map(_run, tasks), 1):
            if "error" in r:
                print(f"  FAIL {r['res']}/{r['clim']}/{r['lulc']}: {r['error']}",
                      flush=True)
            else:
                rows.append(r)
            if i % 25 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(tasks)}  {el/60:.1f} min "
                      f"(eta {el/i*(len(tasks)-i)/60:.0f} min)", flush=True)

    B = pd.DataFrame(rows)
    B.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(B)} rows, {(time.time()-t0)/60:.1f} min)")

    # ---- joined comparison table ------------------------------------------
    jp = Path(args.join)
    if not jp.is_file():
        print(f"\n({jp.name} not found -- run bayes_deploy.py --phase B to "
              f"complete the comparison)")
        return
    D = pd.read_csv(jp)
    M = D.merge(B, on=["res", "clim", "lulc"], how="inner")
    if M.empty:
        print("\nno overlapping trajectories to compare")
        return

    print(f"\n{'='*74}\nRealised cost vs FROZEN 2020 (negative = better), "
          f"cost-weighted\n{'='*74}")
    print(f"{'res':>4} {'n':>5} {'ff15/50':>9} {'bayes':>9} {'plugin':>9} "
          f"{'x4only':>9} {'oracle':>9}")
    for res, g in M.groupby("res"):
        f = g.J_frozen2020.sum()
        def pc(c): return 100 * (g[c].sum() - f) / f
        print(f"{res:>4} {len(g):5d} {pc('J_ff_15_50'):+8.2f}% "
              f"{pc('J_bayes'):+8.2f}% {pc('J_plugin'):+8.2f}% "
              f"{pc('J_x4only'):+8.2f}% {pc('J_oracle'):+8.2f}%")
    f = M.J_frozen2020.sum()
    def pc(c): return 100 * (M[c].sum() - f) / f
    print(f"{'ALL':>4} {len(M):5d} {pc('J_ff_15_50'):+8.2f}% {pc('J_bayes'):+8.2f}% "
          f"{pc('J_plugin'):+8.2f}% {pc('J_x4only'):+8.2f}% {pc('J_oracle'):+8.2f}%")

    print(f"\n--- vs the PUBLISHED baseline (ff 15/50); this is the claim ---")
    ff = M.J_ff_15_50.sum()
    for c, lab in [("J_bayes", "bayes"), ("J_plugin", "plugin"),
                   ("J_oracle", "oracle")]:
        print(f"  {lab:>7}: {100*(M[c].sum()-ff)/ff:+.2f}%")
    gap = (M.J_bayes.sum() - M.J_oracle.sum())
    tot = (M.J_ff_15_50.sum() - M.J_oracle.sum())
    if tot > 0:
        print(f"\n  bayes captures {100*(1-gap/tot):.0f}% of the ff->oracle "
              f"headroom (belief error is the rest)")


if __name__ == "__main__":
    main()
