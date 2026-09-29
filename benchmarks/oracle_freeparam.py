#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FREE-PARAMETER hindsight oracle: the same 7-parameter policy SHAPE as the atlas,
but with parameters free in PARAM_BOUNDS instead of restricted to the 112 atlas
candidates. Annual switching, exactly like bench_baselines.oracle_sequence.

WHY. The existing J_oracle picks the best ATLAS policy sequence, so it cannot
separate two different error sources:

    J_oracle      - J_freeparam   ATLAS error: the cost of the atlas being a
                                  finite, pre-computed set over a designed
                                  exposure space (discretisation + coverage)
    J_freeparam   - J_trueopt     POLICY-CLASS error: the rigidity of the
                                  7-parameter shape itself (see true_optimum.py)

METHOD. Backward coordinate descent over years, initialised AT the atlas oracle's
own sequence. Year t influences the future only through its ending storage, so
for each position we build the tail value function V_{t+1}(S) on a storage grid
(one tail simulation per grid point) and let DE optimise the cheap surrogate

    f(x) = J_t(x ; S_at[t])  +  V_{t+1}( S_end(x) )

The DE proposal is then re-scored with an EXACT tail simulation and accepted only
on strict improvement -- the same rule oracle_sequence uses. Consequences:

  * J_freeparam <= J_oracle always (the atlas solution is in the search set), so
    the measured atlas error is a CONSERVATIVE LOWER BOUND;
  * V-interpolation error can only cost us proposals, never validity.

USAGE
    python scripts/freeparam_oracle.py --res ORO --one      # single trajectory
    python scripts/freeparam_oracle.py --jobs 24 --out outputs/freeparam_oracle.csv
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"                       # MUST precede numpy import

import sys, argparse, time, traceback                                   # noqa: E402
from pathlib import Path                                                # noqa: E402
import numpy as np, pandas as pd                                        # noqa: E402
import multiprocessing as mp                                            # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed        # noqa: E402
from scipy.optimize import differential_evolution                       # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps")); sys.path.insert(0, str(REPO / "bdps"))
from data_io import load_inflow, load_demand, reservoir_config           # noqa: E402
from model import simulate, PARAM_BOUNDS                                # noqa: E402
from belief import annual_events                                    # noqa: E402
from baselines import oracle_sequence                             # noqa: E402

OUT = REPO / "results"
CACHE = OUT / "deploy_cache"
XDIR = OUT / "freeparam_X"                 # per-trajectory optimal parameter paths
XDIR.mkdir(parents=True, exist_ok=True)
START_YEAR, BURN, ATLAS_YEARS = 2020, 8, 50.0
RES9 = ["NML", "EXC", "NHG", "BUL", "FOL", "SHA", "MIL", "ORO", "DNP"]
BND = np.asarray(PARAM_BOUNDS, dtype=float)

try:
    mp.set_start_method("fork")
except RuntimeError:
    pass


# ---------------------------------------------------------------- trajectory
def build_years(res, cs, ls, oh=20):
    """Per-year realised arrays for the evaluation window (WY2028-2099), plus
    the truelookup starting sequence -- mirrors bench_baselines._run."""
    cfg = reservoir_config(res)
    z = np.load(CACHE / f"{res}.npz")
    Xc = z["X"]
    coord_c = np.log(np.maximum(z["coord"], 1.0)); dem_c = z["demand"]

    q_all, dowy_all, dates_all = load_inflow(cs, key=res)
    dm_all, _ = load_demand(ls)
    n = min(q_all.size, dm_all.size)
    q_all, dowy_all, dates_all, dm_all = (q_all[:n], dowy_all[:n],
                                          dates_all[:n], dm_all[:n])
    yv = dates_all.year.values
    wyd = yv + (dates_all.month.values >= 10)
    yrs_future = np.unique(wyd[wyd >= START_YEAR])[BURN:]

    mfut = yv >= START_YEAR
    _, _, Vann = annual_events(q_all[mfut], dates_all[mfut], cfg.r_max_cfs)
    wy_f = wyd[mfut]
    dann = np.array([dm_all[mfut][wy_f == y].mean() for y in np.unique(wy_f)])
    uy = np.unique(wy_f)
    tf = np.full(len(uy), np.nan); td = np.full(len(uy), np.nan)
    for k in range(len(uy)):
        wv = Vann[k:k + oh]; wd = dann[k:k + max(oh, 1)]
        if len(wv) >= 1:
            tf[k] = np.log(max(wv.mean() * ATLAS_YEARS, 1.0)); td[k] = float(wd.mean())
    ok = ~np.isnan(tf); last = np.where(ok)[0][-1] if ok.any() else 0
    tf[~ok] = tf[last]; td[~ok] = td[last]

    yq, ydm, ydowy, init_idx = [], [], [], []
    for yr in yrs_future:
        sel = wyd == yr
        if not sel.any():
            continue
        yq.append(q_all[sel]); ydowy.append(dowy_all[sel]); ydm.append(dm_all[sel])
        ti = min(int(np.searchsorted(uy, yr)), len(tf) - 1)
        init_idx.append(int(np.argmin(np.abs(coord_c - tf[ti])
                                      + 10.0 * np.abs(dem_c - td[ti]))))
    return cfg, Xc, yq, ydm, ydowy, init_idx


# ------------------------------------------------------------- free-param DP
def _tail_cost_X(t, S_start, X, yq, ydm, ydowy, cfg):
    """Exact cost of years t..end under the per-year parameter matrix X."""
    c, S = 0.0, float(S_start)
    for u in range(t, len(yq)):
        o = simulate(yq[u], X[u], cfg, ydowy[u], DM=ydm[u], S0=S)
        c += float(o["J"]); S = float(o["storage"][-1])
    return c


def _score_X(X, S0, yq, ydm, ydowy, cfg):
    """Total cost of the whole horizon under X, split into flood / shortage."""
    J = Jf = Js = 0.0
    S = float(S0)
    for u in range(len(yq)):
        o = simulate(yq[u], X[u], cfg, ydowy[u], DM=ydm[u], S0=S)
        J += float(o["J"]); Jf += float(o["J_flood"]); Js += float(o["J_shortage"])
        S = float(o["storage"][-1])
    return J, Jf, Js


def _storage_path(X, S0, yq, ydm, ydowy, cfg):
    S_at, S = [], float(S0)
    for u in range(len(yq)):
        S_at.append(S)
        o = simulate(yq[u], X[u], cfg, ydowy[u], DM=ydm[u], S0=S)
        S = float(o["storage"][-1])
    return S_at, S


def freeparam_oracle(cfg, Xc, yq, ydm, ydowy, init_idx, S0, n_sweeps=2,
                     grid=25, popsize=12, maxiter=45, detol=0.01, seed=0,
                     atlas_sweeps=2):
    """Atlas oracle first, then continuous per-year refinement."""
    T = len(yq)
    J_atlas, seq = oracle_sequence(yq, ydm, ydowy, Xc, cfg, S0, init_idx,
                                   n_sweeps=atlas_sweeps)
    X = np.array([Xc[i] for i in seq], dtype=float)          # init at the atlas
    K = float(cfg.capacity_af)
    Sg = np.linspace(0.0, K, grid)
    rng = np.random.default_rng(seed)
    J_cur = J_atlas
    n_imp = 0

    for _ in range(n_sweeps):
        S_at, _ = _storage_path(X, S0, yq, ydm, ydowy, cfg)
        changed = False
        for t in range(T - 1, -1, -1):
            S_start = S_at[t]
            if t == T - 1:                                    # no tail
                def V(s):
                    return 0.0
            else:                                             # tail value grid
                Vg = np.array([_tail_cost_X(t + 1, s, X, yq, ydm, ydowy, cfg)
                               for s in Sg])
                def V(s, Sg=Sg, Vg=Vg):
                    return float(np.interp(s, Sg, Vg))

            qq, dw, dv = yq[t], ydowy[t], ydm[t]

            def f(x):
                o = simulate(qq, x, cfg, dw, DM=dv, S0=S_start)
                return float(o["J"]) + V(float(o["storage"][-1]))

            # seed the population AT the incumbent so DE can only help
            npop = max(popsize * BND.shape[0], 15)
            pop = rng.uniform(BND[:, 0], BND[:, 1], size=(npop, BND.shape[0]))
            pop[0] = X[t]
            try:
                r = differential_evolution(f, PARAM_BOUNDS, init=pop, tol=detol,
                                           maxiter=maxiter, polish=True,
                                           seed=seed + t, updating="immediate")
                x_new = np.asarray(r.x, dtype=float)
            except Exception:                                  # noqa: BLE001
                continue

            # EXACT re-score; accept only a strict improvement (oracle rule)
            trial = X.copy(); trial[t] = x_new
            c_new = _tail_cost_X(t, S_start, trial, yq, ydm, ydowy, cfg)
            c_old = _tail_cost_X(t, S_start, X, yq, ydm, ydowy, cfg)
            if c_new < c_old - 1e-9:
                X[t] = x_new; changed = True; n_imp += 1
        if not changed:
            break

    J_free, Jf_free, Js_free = _score_X(X, S0, yq, ydm, ydowy, cfg)
    Xa = np.array([Xc[i] for i in seq], dtype=float)
    _, Jf_atl, Js_atl = _score_X(Xa, S0, yq, ydm, ydowy, cfg)
    return (J_atlas, J_free, X, n_imp,
            dict(J_oracle_free_flood=Jf_free, J_oracle_free_short=Js_free,
                 J_oracle_atlas_flood=Jf_atl, J_oracle_atlas_short=Js_atl))


def _run(task):
    res, cs, ls, oh, nsw, grid, popsize, maxiter = task
    t0 = time.time()
    try:
        cfg, Xc, yq, ydm, ydowy, init_idx = build_years(res, cs, ls, oh)
        S0 = float(cfg.S_avg[-1])
        J_atlas, J_free, X, nimp, comp = freeparam_oracle(
            cfg, Xc, yq, ydm, ydowy, init_idx, S0, n_sweeps=nsw,
            grid=grid, popsize=popsize, maxiter=maxiter)
        np.save(XDIR / f"{res}__{cs}__{ls}.npy", X.astype(np.float32))
        return dict(res=res, clim=cs, lulc=ls, n_years=len(yq),
                    J_oracle_atlas=J_atlas, J_oracle_free=J_free,
                    atlas_error_pct=100.0 * (1.0 - J_free / J_atlas),
                    n_improved=nimp, secs=time.time() - t0, **comp)
    except Exception as e:                                     # noqa: BLE001
        return dict(res=res, clim=cs, lulc=ls, error=f"{type(e).__name__}: {e}",
                    tb=traceback.format_exc()[-400:], secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--res", default=",".join(RES9))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--one", action="store_true", help="single trajectory timing test")
    ap.add_argument("--oracle-horizon", type=int, default=20)
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--grid", type=int, default=25)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=45)
    ap.add_argument("--out", default=str(OUT / "freeparam_oracle.csv"))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    if a.one:
        res = a.res.split(",")[0]
        r = _run((res, "cnrm-cm5_rcp85_r1i1p1", "GCAM-26", a.oracle_horizon,
                  a.sweeps, a.grid, a.popsize, a.maxiter))
        for k, v in r.items():
            if k != "tb":
                print(f"  {k}: {v}")
        return

    ref = pd.read_csv(OUT / "bench_baselines.csv")[["res", "clim", "lulc"]]
    ref = ref[ref.res.isin([r.strip() for r in a.res.split(",")])]
    if a.limit:
        ref = ref.groupby("res", group_keys=False).head(a.limit)
    outp = Path(a.out)
    done = set()
    if a.resume and outp.exists():
        done = set(map(tuple, pd.read_csv(outp)[["res", "clim", "lulc"]].values))
        print(f"resuming: {len(done)} done")
    tasks = [(r.res, r.clim, r.lulc, a.oracle_horizon, a.sweeps, a.grid,
              a.popsize, a.maxiter)
             for r in ref.itertuples() if (r.res, r.clim, r.lulc) not in done]
    print(f"{len(tasks)} trajectories | {a.jobs} workers | BLAS pinned to 1")
    if not tasks:
        print("nothing to do"); return

    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futs = [pool.submit(_run, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result(); rows.append(r)
            if "error" in r:
                print(f"  [{i}/{len(tasks)}] ERR {r['res']}: {r['error'][:70]}",
                      flush=True)
            else:
                print(f"  [{i}/{len(tasks)}] {r['res']} {r['clim'][:14]} "
                      f"atlas_err={r['atlas_error_pct']:+.2f}% "
                      f"({r['n_improved']} yrs improved) {r['secs']:.0f}s",
                      flush=True)
            if i % 10 == 0 or i == len(tasks):
                pd.DataFrame(rows).to_csv(outp, index=False)

    df = pd.DataFrame(rows)
    if a.resume and outp.exists() and done:
        df = pd.concat([pd.read_csv(outp), df], ignore_index=True).drop_duplicates(
            ["res", "clim", "lulc"], keep="last")
    df.to_csv(outp, index=False)
    print(f"\nwrote {outp} ({len(df)} rows) in {(time.time()-t0)/60:.1f} min")
    if "atlas_error_pct" in df.columns:
        g = df.dropna(subset=["atlas_error_pct"])
        print("\nATLAS ERROR (% of atlas-oracle cost), cost-weighted by reservoir:")
        print(g.groupby("res").apply(
            lambda x: 100 * (1 - x.J_oracle_free.sum() / x.J_oracle_atlas.sum())
        ).round(2).to_string())


if __name__ == "__main__":
    main()
