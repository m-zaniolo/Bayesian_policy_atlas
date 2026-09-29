#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/model.py -- Oroville single-reservoir simulation, hedging policy, and objective.

Adapted (NOT rewritten) from the Sunkara-Herman upstream repo
(ssaiveena/Continuous-Reoptimization), files reopt/model.py and
reopt/setup_Future.py. The reservoir dynamics and cost here are kept
numerically identical to `reservoir_step` / `reservoir_fit` upstream, so the
policy parameters and objective are directly comparable to the paper.

Key facts established during reconnaissance (see docs/upstream_inventory.md):
  * Oroville is decoupled from the statewide network: its release policy and
    its shortage+flood cost depend ONLY on Oroville inflow, capacity, safe
    release, historical median release/storage, and a demand multiplier.
    The Delta gains / pumping code upstream is downstream diagnostics only and
    never feeds back into reservoir storage.
  * Units: inflow / release in cfs; storage / capacity in acre-feet (af);
    daily timestep; day-of-water-year (dowy) runs 0..365 (leap-aware).
  * Policy vector x = [x0..x6]; objective J = sum(shortage^2) + c*sum(flood).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# Optional numba: fall back to a no-op decorator if numba is unavailable so the
# module still imports and runs (just slower). The offline atlas build should
# have numba installed for speed.
try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def _wrap(f):
            return f
        return _wrap

# --- unit conversions (identical to upstream) --------------------------------
CFS_TO_AFD = 2.29568411e-5 * 86400.0   # cubic feet/s -> acre-feet/day
AFD_TO_CFS = 1.0 / CFS_TO_AFD

# --- policy parameter convention (fixed project-wide) ------------------------
# x0 : hedging exponent
# x1 : TOCS breakpoint 1 (dowy)   -- end of drawdown
# x2 : TOCS breakpoint 2 (dowy)   -- begin refill
# x3 : TOCS breakpoint 3 (dowy)   -- end refill
# x4 : top-of-conservation-storage level (fraction of capacity)
# x5 : flood-pool release rate (fraction of excess above TOCS)
# x6 : dead-pool level (fraction of capacity)
PARAM_NAMES = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]
PARAM_BOUNDS = [(1, 3), (0, 100), (100, 250), (250, 366), (0, 1), (0, 1), (0, 0.2)]

# Oroville physical constants (from data/reference/nodes.json, key "ORO")
ORO_CAPACITY_AF = 3524.0 * 1000.0     # capacity_taf * 1000
ORO_SAFE_RELEASE_CFS = 150000.0       # safe_release_cfs
FLOOD_PENALTY = 1e5                   # c in objective (upstream: 10**5)


@dataclass
class ReservoirConfig:
    """Everything a single-reservoir simulation needs, in consistent units."""
    capacity_af: float                 # storage capacity, acre-feet
    r_max_cfs: float                   # safe max downstream release, cfs
    R_avg: np.ndarray                  # (366,) median release by dowy, cfs (demand proxy)
    S_avg: np.ndarray                  # (366,) median storage by dowy, acre-feet
    flood_penalty: float = FLOOD_PENALTY


# -----------------------------------------------------------------------------
# Core numerics (numba-compiled). Operate on plain numpy arrays only.
# -----------------------------------------------------------------------------

def tocs_series(x, dowy):
    """Top-of-conservation-storage fraction for each day (vectorized np.interp).

    Kept in plain numpy (not njit) to avoid numba np.interp edge cases; it is a
    one-shot O(T) call, negligible next to the main loop.
    """
    tp = np.array([0.0, x[1], x[2], x[3], 366.0])
    sp = np.array([1.0, x[4], x[4], 1.0, 1.0])
    return np.interp(dowy.astype(np.float64), tp, sp)


@njit(cache=True)
def _reservoir_step(x0, x5, x6, Q_cfs, S, K, demand_cfs, S_avg_prev, tocs):
    """One daily step. Mirror of upstream reservoir_step (reopt/model.py).

    Returns (release_cfs, storage_af_next). Storage is bounded to [0, K]:
    spill enforces S <= K, and the mass balance floors it at 0.
    """
    R_avg = demand_cfs * CFS_TO_AFD   # demand target, converted to af/day
    Q = Q_cfs * CFS_TO_AFD

    R_target = R_avg
    if S < S_avg_prev:
        R_target = R_avg * (S / S_avg_prev) ** x0   # exponential hedging

    S_target = max(0.0, S + Q - R_target)           # assumes 1-day forecast

    if S_target > K * tocs:                          # flood pool
        R_target += (S_target - K * tocs) * x5
        S_target = max(0.0, S + Q - R_target)

    if S_target > K:                                 # spill
        R_target += S_target - K
    elif S_target < K * x6:                          # below dead pool
        R_target = max(0.0, R_target - (K * x6 - S_target))

    S_next = S + Q - R_target
    if S_next < 0.0:
        S_next = 0.0
    return R_target * AFD_TO_CFS, S_next


@njit(cache=True)
def _simulate_core(x, dowy, Q, K, R_avg, S_avg, S0, R_max, DM, tocs, flood_penalty):
    T = dowy.size
    R = np.zeros(T)
    S = np.zeros(T)
    shortage = np.zeros(T)
    flood = np.zeros(T)

    for t in range(T):
        d = dowy[t]
        demand = R_avg[d] * DM[t]
        S_prev = S0 if t == 0 else S[t - 1]
        # NOTE: upstream indexes median storage with the *previous* day's dowy
        # (S_avg[dowy[t-1]]); at t==0 this is dowy[-1] (last day). Replicated
        # faithfully to stay numerically identical to the paper.
        S_avg_prev = S_avg[dowy[t - 1]]

        Rt, St = _reservoir_step(x[0], x[5], x[6], Q[t], S_prev, K,
                                 demand, S_avg_prev, tocs[t])
        R[t] = Rt
        S[t] = St

        sc = demand - Rt
        if sc > 0.0:
            shortage[t] = sc * sc
        if Rt > R_max:
            flood[t] = flood_penalty * (Rt - R_max)

    return R, S, shortage, flood


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def simulate(inflow, x, cfg: ReservoirConfig, dowy, DM=None, S0=None,
             return_series=False):
    """Run the daily water balance over an inflow array.

    Parameters
    ----------
    inflow : (T,) daily inflow, cfs
    x      : policy parameter vector [x0..x6]
    cfg    : ReservoirConfig
    dowy   : (T,) day-of-water-year (0..365) for each day
    DM     : (T,) demand multiplier; default 1.0 everywhere (historical demand)
    S0     : initial storage, af; default = last median storage (upstream default)

    Returns
    -------
    dict with keys: storage, release, shortage, flood (each (T,)) plus
    J_shortage, J_flood, J (scalars). If return_series is False the arrays are
    still returned (cheap); callers wanting only J use `objective`.
    """
    x = np.asarray(x, dtype=np.float64)
    inflow = np.ascontiguousarray(inflow, dtype=np.float64)
    dowy = np.ascontiguousarray(dowy, dtype=np.int64)
    K = float(cfg.capacity_af)

    if DM is None:
        DM = np.ones(inflow.size, dtype=np.float64)
    else:
        DM = np.ascontiguousarray(DM, dtype=np.float64)
    if S0 is None:
        S0 = float(cfg.S_avg[-1])

    tocs = tocs_series(x, dowy)
    R, S, shortage, flood = _simulate_core(
        x, dowy, inflow, K,
        np.ascontiguousarray(cfg.R_avg, dtype=np.float64),
        np.ascontiguousarray(cfg.S_avg, dtype=np.float64),
        float(S0), float(cfg.r_max_cfs), DM, tocs, float(cfg.flood_penalty),
    )

    J_shortage = float(shortage.sum())
    J_flood = float(flood.sum())
    return {
        "storage": S,
        "release": R,
        "shortage": shortage,
        "flood": flood,
        "J_shortage": J_shortage,
        "J_flood": J_flood,
        "J": J_shortage + J_flood,
    }


@njit(cache=True)
def _release_af(x0, x5, x6, Q_af, S, K, demand_af, S_avg_prev, tocs):
    """Release (acre-feet/day) a policy prescribes from a GIVEN state -- the
    release part of _reservoir_step, without advancing storage. Used to average
    ACTIONS across policies from a shared state."""
    R = demand_af
    if S < S_avg_prev:
        R = demand_af * (S / S_avg_prev) ** x0
    St = max(0.0, S + Q_af - R)
    if St > K * tocs:
        R += (St - K * tocs) * x5
        St = max(0.0, S + Q_af - R)
    if St > K:
        R += St - K
    elif St < K * x6:
        R = max(0.0, R - (K * x6 - St))
    return R


@njit(cache=True)
def _simulate_blended_core(X0, X5, X6, TOCS, w, dowy, Q, K, R_avg, S_avg, S0,
                           R_max, DM, flood_penalty):
    """Belief-weighted ACTION averaging. One shared storage trajectory; each day
    the applied release is sum_j w_j * release_j(shared state). Smooth in w."""
    T = dowy.size
    nc = X0.size
    R = np.zeros(T); S = np.zeros(T)
    shortage = np.zeros(T); flood = np.zeros(T)
    for t in range(T):
        d = dowy[t]
        demand_cfs = R_avg[d] * DM[t]
        demand_af = demand_cfs * CFS_TO_AFD
        S_prev = S0 if t == 0 else S[t - 1]
        S_avg_prev = S_avg[dowy[t - 1]]
        Q_af = Q[t] * CFS_TO_AFD
        Rb = 0.0
        for j in range(nc):
            Rb += w[j] * _release_af(X0[j], X5[j], X6[j], Q_af, S_prev, K,
                                     demand_af, S_avg_prev, TOCS[j, t])
        S_next = S_prev + Q_af - Rb
        if S_next > K:                       # spill: excess must be released
            Rb += S_next - K; S_next = K
        if S_next < 0.0:                      # cannot release more than available
            Rb += S_next; S_next = 0.0
            if Rb < 0.0:
                Rb = 0.0
        Rc = Rb * AFD_TO_CFS
        R[t] = Rc; S[t] = S_next
        sc = demand_cfs - Rc
        if sc > 0.0:
            shortage[t] = sc * sc
        if Rc > R_max:
            flood[t] = flood_penalty * (Rc - R_max)
    return R, S, shortage, flood


def simulate_blended(inflow, X, w, cfg: ReservoirConfig, dowy, DM=None, S0=None):
    """Deploy the belief-weighted AVERAGE ACTION of a set of policies.

    X : (ncell, 7) candidate policy vectors; w : (ncell,) belief weights summing
    to 1. Each day every policy prescribes a release from the SHARED storage
    state and the applied release is sum_j w_j release_j -- the posterior-expected
    DECISION, not a blend of parameters (which §7.1 warns against). Changes
    smoothly with w, so no discrete policy hops -> lower cost variance. Same cost
    accounting and return dict as simulate().
    """
    X = np.asarray(X, dtype=np.float64)
    w = np.ascontiguousarray(np.asarray(w, dtype=np.float64))
    inflow = np.ascontiguousarray(inflow, dtype=np.float64)
    dowy = np.ascontiguousarray(dowy, dtype=np.int64)
    K = float(cfg.capacity_af)
    if DM is None:
        DM = np.ones(inflow.size, dtype=np.float64)
    else:
        DM = np.ascontiguousarray(DM, dtype=np.float64)
    if S0 is None:
        S0 = float(cfg.S_avg[-1])
    TOCS = np.ascontiguousarray(
        np.array([tocs_series(X[j], dowy) for j in range(X.shape[0])]))
    R, S, shortage, flood = _simulate_blended_core(
        np.ascontiguousarray(X[:, 0]), np.ascontiguousarray(X[:, 5]),
        np.ascontiguousarray(X[:, 6]), TOCS, w, dowy, inflow, K,
        np.ascontiguousarray(cfg.R_avg, dtype=np.float64),
        np.ascontiguousarray(cfg.S_avg, dtype=np.float64),
        float(S0), float(cfg.r_max_cfs), DM, float(cfg.flood_penalty))
    return {"storage": S, "release": R, "shortage": shortage, "flood": flood,
            "J_shortage": float(shortage.sum()), "J_flood": float(flood.sum()),
            "J": float(shortage.sum() + flood.sum())}


def objective(inflow, x, cfg: ReservoirConfig, dowy, DM=None, S0=None) -> float:
    """Total cost J = sum(shortage^2) + c * sum(flood exceedance). Lower is better.

    This is exactly what differential evolution minimizes (upstream reservoir_fit).
    """
    out = simulate(inflow, x, cfg, dowy, DM=DM, S0=S0)
    return out["J"]


def release_policy(storage, dowy, x, cfg: ReservoirConfig, demand_cfs=None):
    """Single-day release for a given storage/dowy (convenience for the online loop).

    Uses the same numerics as the simulation. `demand_cfs` defaults to the
    historical median release for that dowy.
    """
    x = np.asarray(x, dtype=np.float64)
    d = int(dowy)
    if demand_cfs is None:
        demand_cfs = float(cfg.R_avg[d])
    tocs = float(tocs_series(x, np.array([d]))[0])
    S_avg_prev = float(cfg.S_avg[d])
    R, _ = _reservoir_step(x[0], x[5], x[6], 0.0, float(storage),
                           float(cfg.capacity_af), demand_cfs, S_avg_prev, tocs)
    # Note: release with zero inflow context; for a full step use simulate().
    return R
