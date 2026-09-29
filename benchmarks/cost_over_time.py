#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-year flood & demand cost of every deployed method, all 9 reservoirs — the
time-resolved version of the cost-decomposition barplot.

Runs the exact canonical deployment (bayes_deploy._run, horizon=1, historical
leak-free hyperparameters, cwidth on) with EMIT_YEARLY so each trajectory returns
its per-water-year (J_flood, J_shortage) for methods frozen / plugin / bayes /
blend / x4only. Flattened to a tidy long table for the cost-over-time analysis:
  - does adaptation only pay off AFTER a flood is observed (event-aligned)?
  - do costs / spread grow as the climate gets more extreme (late-century)?
  - WHERE do plugin/bayes/blend actually differ (tails, not means)?

Output: outputs/cost_over_time.csv.gz  (res,clim,lulc,method,year,J_flood,J_shortage)
"""
from __future__ import annotations
import sys, functools, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd

try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))
sys.path.insert(0, str(REPO / "bdps"))
import deploy as bd                                              # noqa: E402
from belief import fit_gamma_shape, fit_hyper                      # noqa: E402
from concurrent.futures import ProcessPoolExecutor                     # noqa: E402

HORIZON = 1                     # canonical decision horizon (HANDOFF §7.12)
NC, ND = 30, 8
RES = ["NML", "EXC", "NHG", "BUL", "FOL", "SHA", "MIL", "ORO", "DNP"]


def main():
    bd.EMIT_YEARLY = True                     # inherited by fork workers
    clim = bd.scenario_names()[:NC]
    lul = bd.lulc_names(ND)
    print(f"{len(RES)} res x {len(clim)} GCM x {len(lul)} demand = "
          f"{len(RES)*len(clim)*len(lul)} trajectories, horizon={HORIZON}", flush=True)

    hyper = {}
    for res in RES:
        n, V = bd.historical_obs(res)
        a = fit_gamma_shape([(n, V)])
        ql, qm, phi = fit_hyper([(n, V)], a, n_part=1200)
        cw = bd.calibrate_belief_width(res, ql, qm, a, phi, HORIZON)
        hyper[res] = (ql, qm, a, phi, cw)
        print(f"  hyper {res}: a={a:.2f} Q=({ql:.4f},{qm:.4f}) cwidth={cw:.2f}", flush=True)

    tasks = [(res, cs, ls, (hyper[res][0], hyper[res][1]), hyper[res][2],
              hyper[res][3], HORIZON, hyper[res][4])
             for res in RES for cs in clim for ls in lul]
    rn = functools.partial(bd._run, switch_margin=0.0)

    recs, done, t0 = [], 0, __import__("time").time()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for r in pool.map(rn, tasks):
            done += 1
            if "error" in r:
                print("  ERR", r.get("res"), str(r.get("error"))[:80], flush=True)
                continue
            for (yr, meth, jf, js) in r["yearly"]:
                recs.append((r["res"], r["clim"], r["lulc"], meth, yr, jf, js))
            if done % 300 == 0:
                print(f"  {done}/{len(tasks)} ({__import__('time').time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(recs, columns=["res", "clim", "lulc", "method", "year",
                                     "J_flood", "J_shortage"])
    out = REPO / "results" / "cost_over_time.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    print(f"\nwrote {out}  ({len(df)} rows, years "
          f"{int(df.year.min())}-{int(df.year.max())})", flush=True)


if __name__ == "__main__":
    main()
