#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/optimize.py -- differential-evolution policy search for a single reservoir.

Mirrors the upstream call in Reopt_main.py:
    DE(reservoir_fit, tol=1, maxiter=100000,
       bounds=[(1,3),(0,100),(100,250),(250,366),(0,1),(0,1),(0,0.2)])
run with 10 independent restarts, keeping the minimum-objective result.

The full-scale settings (maxiter=100000, trials=10) are HPC-scale. For local
validation pass small maxiter/trials via the driver's --dry-run.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import differential_evolution

from model import objective, PARAM_BOUNDS


def _de_obj(x, inflow, cfg, dowy, DM, S0):
    return objective(inflow, x, cfg, dowy, DM=DM, S0=S0)


def optimize_policy(inflow, cfg, dowy, DM=None, S0=None,
                    maxiter=100000, tol=1.0, trials=10, seed=0,
                    polish=False, workers=1):
    """Return (x_star, J_star): best policy over `trials` DE restarts.

    tol=1 (upstream) makes DE stop early once the population collapses, so the
    maxiter cap is rarely binding. `polish=False` avoids an extra local
    minimization (the objective is noisy/flat in places).
    """
    inflow = np.ascontiguousarray(inflow, dtype=np.float64)
    dowy = np.ascontiguousarray(dowy, dtype=np.int64)
    best_x, best_J = None, np.inf
    for k in range(trials):
        res = differential_evolution(
            _de_obj, PARAM_BOUNDS,
            args=(inflow, cfg, dowy, DM, S0),
            maxiter=maxiter, tol=tol, seed=seed + k,
            polish=polish, workers=workers, updating="deferred" if workers != 1 else "immediate",
        )
        if res.fun < best_J:
            best_J, best_x = float(res.fun), np.asarray(res.x, dtype=np.float64)
    return best_x, best_J


def optimize_constant_policy(inflows, dowys, cfg, DMs=None, maxiter=3000, tol=1.0,
                             trials=6, seed=0, polish=False):
    """Single policy minimizing TOTAL cost across a set of scenarios.

    The honest floor the atlas must beat: one well-tuned fixed policy applied to
    every scenario (each scored over its full record from median initial storage,
    no cross-scenario storage chaining). Pass DMs (list of demand-multiplier
    series, aligned to each inflow) to score against time-varying demand.
    Returns (x_const, J_total_best).
    """
    inflows = [np.ascontiguousarray(q, dtype=np.float64) for q in inflows]
    dowys = [np.ascontiguousarray(d, dtype=np.int64) for d in dowys]
    if DMs is None:
        DMs = [None] * len(inflows)
    S0 = float(cfg.S_avg[-1])

    def total_obj(x):
        return sum(objective(q, x, cfg, d, DM=dm, S0=S0)
                   for q, d, dm in zip(inflows, dowys, DMs))

    best_x, best_J = None, np.inf
    for k in range(trials):
        res = differential_evolution(total_obj, PARAM_BOUNDS, maxiter=maxiter,
                                     tol=tol, seed=seed + k, polish=polish)
        if res.fun < best_J:
            best_J, best_x = float(res.fun), np.asarray(res.x, dtype=np.float64)
    return best_x, best_J


def optimize_policy_spread(inflow, cfg, dowy, DM=None, S0=None,
                           n_seeds=8, maxiter=1000, tol=1.0):
    """Diagnostic for the theta->x_star identifiability risk.

    Run independent single-restart DE with different seeds on the SAME window and
    return the array of solutions (n_seeds, 7) plus their objective values. A
    large spread in x for a fixed window means the atlas target is ill-posed.
    """
    inflow = np.ascontiguousarray(inflow, dtype=np.float64)
    dowy = np.ascontiguousarray(dowy, dtype=np.int64)
    xs, Js = [], []
    for s in range(n_seeds):
        res = differential_evolution(
            _de_obj, PARAM_BOUNDS, args=(inflow, cfg, dowy, DM, S0),
            maxiter=maxiter, tol=tol, seed=s, polish=False,
        )
        xs.append(res.x)
        Js.append(res.fun)
    return np.array(xs), np.array(Js)
