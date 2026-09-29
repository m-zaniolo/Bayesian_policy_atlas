#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_model.py -- protect the reservoir simulation against silent breakage
while adapting upstream code. Runnable with `pytest` or directly as a script.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import ReservoirConfig, simulate, objective, PARAM_BOUNDS  # noqa: E402


def _toy_config(cap_af=1000.0, r_max=50.0):
    # Flat medians so the physics is easy to reason about.
    R_avg = np.full(366, 10.0)     # cfs demand proxy
    S_avg = np.full(366, cap_af * 0.5)
    return ReservoirConfig(capacity_af=cap_af, r_max_cfs=r_max,
                           R_avg=R_avg, S_avg=S_avg)


def _mid_policy():
    return np.array([lo + 0.5 * (hi - lo) for lo, hi in PARAM_BOUNDS])


def test_storage_bounds():
    """Storage must stay within [0, capacity] under any inflow."""
    cfg = _toy_config()
    dowy = np.tile(np.arange(365), 3)[: 365 * 3].astype(int)
    rng = np.random.default_rng(0)
    inflow = rng.uniform(0, 5000, size=dowy.size)  # includes flood-scale inflow
    out = simulate(inflow, _mid_policy(), cfg, dowy)
    S = out["storage"]
    assert np.all(S >= -1e-6), f"storage went negative: min={S.min()}"
    assert np.all(S <= cfg.capacity_af + 1e-3), f"storage exceeded capacity: max={S.max()}"


def test_harder_hedging_lowers_releases_when_dry():
    """Larger hedging exponent x0 => lower releases when storage is below median."""
    cfg = _toy_config()
    dowy = np.arange(365).astype(int)
    inflow = np.full(365, 2.0)          # low inflow -> storage stays below S_avg
    x_soft = _mid_policy().copy(); x_soft[0] = 1.0
    x_hard = _mid_policy().copy(); x_hard[0] = 3.0
    S0 = cfg.S_avg[0] * 0.4             # start dry
    r_soft = simulate(inflow, x_soft, cfg, dowy, S0=S0)["release"].sum()
    r_hard = simulate(inflow, x_hard, cfg, dowy, S0=S0)["release"].sum()
    assert r_hard <= r_soft + 1e-9, f"harder hedging released more: {r_hard} > {r_soft}"


def test_objective_penalizes_flood():
    """Objective increases when releases exceed r_max (flood penalty active)."""
    cfg = _toy_config(cap_af=1000.0, r_max=50.0)
    dowy = np.arange(365).astype(int)
    calm = np.full(365, 10.0)
    flood = calm.copy()
    flood[100:110] = 500000.0          # huge pulse forces spill above r_max
    J_calm = objective(calm, _mid_policy(), cfg, dowy)
    J_flood = objective(flood, _mid_policy(), cfg, dowy)
    assert J_flood > J_calm, f"flood did not raise objective: {J_flood} !> {J_calm}"
    out = simulate(flood, _mid_policy(), cfg, dowy)
    assert out["J_flood"] > 0, "expected nonzero flood cost"


def test_deterministic():
    cfg = _toy_config()
    dowy = np.arange(365).astype(int)
    inflow = np.linspace(1, 100, 365)
    a = objective(inflow, _mid_policy(), cfg, dowy)
    b = objective(inflow, _mid_policy(), cfg, dowy)
    assert a == b


if __name__ == "__main__":
    test_storage_bounds()
    test_harder_hedging_lowers_releases_when_dry()
    test_objective_penalizes_flood()
    test_deterministic()
    print("test_model.py: all passed")
