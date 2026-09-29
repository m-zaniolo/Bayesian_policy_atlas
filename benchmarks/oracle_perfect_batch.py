#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch the TRUE perfect-foresight optimum (scripts/true_optimum.py) over all
(reservoir x GCM x demand) trajectories, in parallel.

Each trajectory is an independent convex QP -- embarrassingly parallel. The
terminal storage is MATCHED to what `truelookup` ends at on the same trajectory,
so J_trueopt <= J_truelookup holds by construction and the difference

    J_truelookup - J_trueopt

is the cost of being confined to the atlas AND the 7-parameter policy class,
given perfect information. (Matched-terminal values are only a valid ceiling for
truelookup -- report them as a regret, not as a standalone cost column.)

The evaluation window replicates bench_baselines exactly: water years
START_YEAR+BURN .. end (2028-2099), S0 = cfg.S_avg[-1], leap-aware dowy.

BLAS threads are pinned to 1 BEFORE numpy is imported so N worker processes do
not oversubscribe the machine (critical: without this, 24 workers each spawning
BLAS threads run slower than serial).

USAGE
    python scripts/true_optimum_batch.py --jobs 24 --out outputs/true_optimum.csv
    python scripts/true_optimum_batch.py --jobs 4 --res ORO --limit 3   # smoke test
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"                       # MUST precede numpy/osqp import

import sys, argparse, time, traceback                                  # noqa: E402
from pathlib import Path                                               # noqa: E402
import numpy as np, pandas as pd                                       # noqa: E402
import multiprocessing as mp                                           # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed       # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps")); sys.path.insert(0, str(REPO / "bdps"))
from data_io import load_inflow, load_demand, reservoir_config          # noqa: E402
from model import simulate                                             # noqa: E402
from belief import annual_events                                   # noqa: E402
from true_optimum import solve_true_optimum                            # noqa: E402

OUT = REPO / "results"
CACHE = OUT / "deploy_cache"
START_YEAR, BURN, ATLAS_YEARS = 2020, 8, 50.0
RES9 = ["NML", "EXC", "NHG", "BUL", "FOL", "SHA", "MIL", "ORO", "DNP"]

try:
    mp.set_start_method("fork")
except RuntimeError:
    pass


def truelookup_path(res, cs, ls, oh=20):
    """Replicate bench_baselines' `truelookup` run. Returns the concatenated
    evaluation-window arrays plus truelookup's J and TERMINAL STORAGE."""
    cfg = reservoir_config(res)
    z = np.load(CACHE / f"{res}.npz")
    Xc = z["X"]
    coord_c = np.log(np.maximum(z["coord"], 1.0))
    dem_c = z["demand"]

    q_all, dowy_all, dates_all = load_inflow(cs, key=res)
    dm_all, _ = load_demand(ls)
    n = min(q_all.size, dm_all.size)
    q_all, dowy_all, dates_all, dm_all = (q_all[:n], dowy_all[:n],
                                          dates_all[:n], dm_all[:n])
    yv = dates_all.year.values
    wyd = yv + (dates_all.month.values >= 10)

    yrs_future = np.unique(wyd[wyd >= START_YEAR])[BURN:]

    # TRUE forward exposure per year (perfect information), as in bench_baselines
    mfut = yv >= START_YEAR
    _, _, Vann = annual_events(q_all[mfut], dates_all[mfut], cfg.r_max_cfs)
    wy_f = wyd[mfut]
    dann = np.array([dm_all[mfut][wy_f == y].mean() for y in np.unique(wy_f)])
    uy = np.unique(wy_f)
    tf_raw = np.full(len(uy), np.nan); td_raw = np.full(len(uy), np.nan)
    for k in range(len(uy)):
        wv = Vann[k:k + oh]; wd = dann[k:k + max(oh, 1)]
        if len(wv) >= 1:
            tf_raw[k] = np.log(max(wv.mean() * ATLAS_YEARS, 1.0))
            td_raw[k] = float(wd.mean())
    ok = ~np.isnan(tf_raw)
    last = np.where(ok)[0][-1] if ok.any() else 0
    tf_raw[~ok] = tf_raw[last]; td_raw[~ok] = td_raw[last]

    S = float(cfg.S_avg[-1]); S0 = S
    Jtl = Jfl = Jsh = 0.0
    QQ, DW, DM = [], [], []
    for yr in yrs_future:
        sel = wyd == yr
        if not sel.any():
            continue
        qq, dowy, dmv = q_all[sel], dowy_all[sel], dm_all[sel]
        ti = min(int(np.searchsorted(uy, yr)), len(tf_raw) - 1)
        i_true = int(np.argmin(np.abs(coord_c - tf_raw[ti])
                               + 10.0 * np.abs(dem_c - td_raw[ti])))
        o = simulate(qq, Xc[i_true], cfg, dowy, DM=dmv, S0=S)
        Jtl += float(o["J"]); Jfl += float(o["J_flood"]); Jsh += float(o["J_shortage"])
        S = float(o["storage"][-1])
        QQ.append(qq); DW.append(dowy); DM.append(dmv)

    return (cfg, np.concatenate(QQ), np.concatenate(DW), np.concatenate(DM),
            S0, S, Jtl, Jfl, Jsh)


def _run(task):
    res, cs, ls, oh, eps = task
    t0 = time.time()
    try:
        cfg, q, dowy, dm, S0, S_end, Jtl, Jfl, Jsh = truelookup_path(res, cs, ls, oh)
        o = solve_true_optimum(q, dowy, dm, cfg, S0, eps=eps,
                               pin_terminal=True, S_end_af=S_end)
        return dict(res=res, clim=cs, lulc=ls, T=int(q.size),
                    J_trueopt=o["J"], J_trueopt_flood=o["J_flood"],
                    J_trueopt_short=o["J_shortage"],
                    J_truelookup=Jtl, J_truelookup_flood=Jfl,
                    J_truelookup_short=Jsh,
                    S0_af=S0, S_end_af=S_end,
                    S_end_solved=float(o["storage"][-1]),
                    qp_status=o["status"], qp_iters=o["iters"],
                    qp_relgap=abs(o["obj_qp"] - o["J"]) / max(o["J"], 1.0),
                    secs=time.time() - t0)
    except Exception as e:                                             # noqa: BLE001
        return dict(res=res, clim=cs, lulc=ls, error=f"{type(e).__name__}: {e}",
                    tb=traceback.format_exc()[-400:], secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--res", default=",".join(RES9))
    ap.add_argument("--limit", type=int, default=0, help="cap trajectories per reservoir")
    ap.add_argument("--oracle-horizon", type=int, default=20)
    ap.add_argument("--eps", type=float, default=1e-7)
    ap.add_argument("--out", default=str(OUT / "true_optimum.csv"))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    ref = pd.read_csv(OUT / "bench_baselines.csv")[["res", "clim", "lulc"]]
    keep = [r.strip() for r in a.res.split(",")]
    ref = ref[ref.res.isin(keep)]
    if a.limit:
        ref = ref.groupby("res", group_keys=False).head(a.limit)

    outp = Path(a.out)
    done = set()
    if a.resume and outp.exists():
        d0 = pd.read_csv(outp)
        done = set(map(tuple, d0[["res", "clim", "lulc"]].values))
        print(f"resuming: {len(done)} already done")

    tasks = [(r.res, r.clim, r.lulc, a.oracle_horizon, a.eps)
             for r in ref.itertuples() if (r.res, r.clim, r.lulc) not in done]
    print(f"{len(tasks)} trajectories | {a.jobs} workers | BLAS threads pinned to 1")
    if not tasks:
        print("nothing to do"); return

    t0 = time.time(); rows, nerr = [], 0
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futs = [pool.submit(_run, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result(); rows.append(r)
            if "error" in r:
                nerr += 1
                print(f"  [{i}/{len(tasks)}] ERR {r['res']}/{r['clim'][:14]}: "
                      f"{r['error'][:70]}", flush=True)
            else:
                print(f"  [{i}/{len(tasks)}] {r['res']} {r['clim'][:14]} "
                      f"J={r['J_trueopt']:.4g} vs tl {r['J_truelookup']:.4g} "
                      f"({100*(1-r['J_trueopt']/r['J_truelookup']):+.1f}%) "
                      f"{r['qp_status']} {r['secs']:.0f}s", flush=True)
            if i % 20 == 0 or i == len(tasks):        # checkpoint
                df = pd.DataFrame(rows)
                if a.resume and outp.exists():
                    df = pd.concat([pd.read_csv(outp), df], ignore_index=True)
                df.to_csv(outp, index=False)

    df = pd.DataFrame(rows)
    if a.resume and outp.exists():
        prev = pd.read_csv(outp)
        prev = prev[~prev.set_index(["res", "clim", "lulc"]).index.isin(
            df.set_index(["res", "clim", "lulc"]).index)]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(outp, index=False)
    print(f"\nwrote {outp}  ({len(df)} rows, {nerr} errors) in "
          f"{(time.time()-t0)/60:.1f} min")

    g = df[~df.get("error", pd.Series(index=df.index, dtype=object)).notna()] \
        if "error" in df.columns else df
    if len(g):
        bad = (g.J_trueopt > g.J_truelookup * 1.0001).sum()
        print(f"ceiling violations (J_trueopt > J_truelookup): {bad}")
        print("\nregret vs truelookup (% of truelookup), by reservoir:")
        gg = g.groupby("res").apply(
            lambda x: 100 * (1 - x.J_trueopt.sum() / x.J_truelookup.sum()))
        print(gg.round(1).to_string())


if __name__ == "__main__":
    main()
