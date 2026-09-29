#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LEARNING-RATE (process-noise Q) sensitivity sweep — HANDOFF §7.15.

Q is the belief's forgetting knob: large Q -> high Kalman gain -> short memory
(trusts recent obs); small Q -> long memory (averages the record). We scale the
historically-FITTED Q of BOTH the flood filter (Q_lam,Q_mag) and the demand LLT
(q_level,q_slope) by a common multiplier c_Q, redeploy 3 reservoirs at each level
(canonical horizon=1), and report the tracking/stability/cost trade-off.

For each level: refit the baseline hyperparameters on the leak-free pre-2020
record, scale Q by c_Q, RECOMPUTE the width recalibration cwidth on the scaled Q
(so width stays honestly calibrated and the sweep isolates learning RATE, not
width), then run the exact deployment loop (bayes_deploy._run).

The effective MEMORY (years) of each c_Q is estimated empirically from the
filtered series (EWMA-equivalent gain K; memory ~ 1/K) so the x-axis is
interpretable, baseline (c_Q=1) marked.

Outputs: outputs/qsweep/qscale_<c>.csv (one row per trajectory, with n_switch)
and outputs/qsweep/memory.csv (c_Q -> effective memory).
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
from belief import (annual_events, fit_gamma_shape, fit_hyper,     # noqa: E402
                        particle_filter)
from concurrent.futures import ProcessPoolExecutor                     # noqa: E402

GRID = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RES = ["NML", "EXC", "ORO"]
HORIZON = 1
NC, ND = 30, 8
OUT = REPO / "results" / "qsweep"


def effective_memory(res, ql, qm, a, phi):
    """EWMA-equivalent memory (yrs) of the flood filter at this Q: fit the gain K
    from fm[t] ~ (1-K) fm[t-1] + K z[t] on the pre-2020 record; memory = 1/K."""
    n, V = bd.historical_obs(res)
    fm, _, _ = particle_filter(n, V, ql, qm, a, 1500, np.random.default_rng(1), phi=phi)
    z = np.log(np.maximum(V.astype(float), 1.0))
    innov = z[1:] - fm[:-1]
    dfm = fm[1:] - fm[:-1]
    denom = float(innov @ innov)
    K = float(dfm @ innov) / denom if denom > 1e-12 else np.nan
    return 1.0 / max(K, 1e-3) if np.isfinite(K) else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clim = bd.scenario_names()[:NC]
    lul = bd.lulc_names(ND)
    print(f"{len(RES)} res x {len(clim)} GCM x {len(lul)} demand, horizon={HORIZON}, "
          f"grid c_Q={GRID}", flush=True)

    # baseline hyperparameters per reservoir (unscaled), fit once
    base = {}
    for res in RES:
        n, V = bd.historical_obs(res)
        a = fit_gamma_shape([(n, V)])
        ql, qm, phi = fit_hyper([(n, V)], a, n_part=1200)
        base[res] = (ql, qm, a, phi)
        print(f"  baseline {res}: a={a:.2f} Q=({ql:.4f},{qm:.4f}) phi={phi:.1f}", flush=True)

    mem_rows = []
    for c in GRID:
        bd.DEMAND_Q_SCALE = c                      # demand Q scaled (read in _run at fork)
        hyper = {}
        mems = []
        for res in RES:
            ql0, qm0, a, phi = base[res]
            ql, qm = ql0 * c, qm0 * c              # flood Q scaled
            cw = bd.calibrate_belief_width(res, ql, qm, a, phi, HORIZON)   # recompute width
            hyper[res] = (ql, qm, a, phi, cw)
            mems.append(effective_memory(res, ql, qm, a, phi))
        mem = float(np.nanmean(mems))
        mem_rows.append(dict(c_Q=c, memory_yr=mem))
        tasks = [(res, cs, ls, (hyper[res][0], hyper[res][1]), hyper[res][2],
                  hyper[res][3], HORIZON, hyper[res][4])
                 for res in RES for cs in clim for ls in lul]
        rn = functools.partial(bd._run, switch_margin=0.0)
        rows = []
        with ProcessPoolExecutor(max_workers=8) as pool:
            for r in pool.map(rn, tasks):
                if "error" in r:
                    print("  ERR", r.get("res"), r.get("error", "")[:80], flush=True)
                else:
                    rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"qscale_{c}.csv", index=False)
        print(f"c_Q={c:<6} memory~{mem:4.1f}yr  n={len(df)}  "
              f"switch_rate med={df.switch_rate.median():.2f}  "
              f"J_bayes tot={df.J_bayes.sum():.3g}", flush=True)

    pd.DataFrame(mem_rows).to_csv(OUT / "memory.csv", index=False)
    print("wrote", OUT / "memory.csv")


if __name__ == "__main__":
    main()
