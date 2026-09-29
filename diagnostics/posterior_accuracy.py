#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Posterior accuracy of the deployed belief, decomposed so the flood-vs-demand
comparison is not confounded by calibration.

For every scenario and decision year we form the belief's forward-H-year
predictive of the exposure the atlas keys on, and compare to the realised
forward-H-year exposure. We record, per (driver, reservoir/scenario, year, H):
    mean, true, sd   ->   z = (mean-true)/sd            (CALIBRATION: honest sd?)
                          rawnorm = |mean-true|/sd_clim  (ACCURACY: beats climatology?)
where sd_clim is the spread of the realised exposure across all scenarios/years
for that driver at that horizon (the variation the belief must resolve).

Belief exactly as deployed (bayes_deploy): flood = particle_filter on
(events, volume) with historical-calibrated hyperparameters + random-walk
horizon widening; demand = local-linear-trend projection (demand_llt). The
flood particle filter is horizon-independent, so it is run ONCE per (res,clim)
and the horizon sweep is analytic.

Writes outputs/posterior_accuracy.parquet (tidy) and prints a summary.
"""
from __future__ import annotations
import sys, functools
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))
sys.path.insert(0, str(REPO / "bdps"))
from data_io import load_demand, reservoir_config, CMIP5_DIR            # noqa: E402
from belief import annual_events, fit_gamma_shape, fit_hyper, particle_filter  # noqa: E402
from deploy import demand_llt                                      # noqa: E402  (pure)


@functools.lru_cache(maxsize=2)
def _clim_df(cs):
    """Whole climate file (all reservoirs) read once -- avoids 9x Box re-reads."""
    return pd.read_csv(Path(CMIP5_DIR) / f"{cs}.csv.zip", index_col=0, parse_dates=True)


def historical_obs(res, cfg, before=2020):
    """Pre-2020 flood (events, volume) from the leak-free reference record --
    same as bayes_deploy.historical_obs but reading the real config, not /tmp."""
    r = pd.read_csv(REPO / "data" / f"find_input_{res}.csv")
    dates = pd.DatetimeIndex(pd.to_datetime(r["date"]))
    q = r["inflow_cfs"].to_numpy(float)
    m = dates.year.values < before
    _, n, V = annual_events(q[m], dates[m], cfg.r_max_cfs)
    return n, V

START = 2020
ATLAS_YEARS = 50.0
BURN = 8
HORIZONS = [1, 2, 3, 5, 7, 10, 15, 20, 30]
RES = ["NML", "EXC", "NHG", "BUL", "FOL", "SHA", "MIL", "ORO", "DNP"]


def true_flood(V, k, H):
    w = V[k:k + H]
    # FULL forward-H window only (match plot_belief_multi); no partial windows.
    return np.log(max(w.mean() * ATLAS_YEARS, 1.0)) if len(w) == H else np.nan


def main():
    b = pd.read_csv(REPO / "results" / "bench_baselines.csv")
    clim = sorted(b.clim.unique()); lulc = sorted(b.lulc.unique())

    # ---- hyperparameters per reservoir (historical, leak-free) ----------------
    hyper = {}
    for r in RES:
        n, V = historical_obs(r, reservoir_config(r))
        a = fit_gamma_shape([(n, V)])
        ql, qm, phi = fit_hyper([(n, V)], a, n_part=1000)
        hyper[r] = (ql, qm, a, phi)
        print(f"hyper {r}: a={a:.2f} Q=({ql:.4f},{qm:.4f}) phi={phi:.1f}", flush=True)

    cfgs = {r: reservoir_config(r) for r in RES}
    rows = []
    # ---- FLOOD: clim OUTER (read each file once), particle filter per (res,clim)
    for ci, cs in enumerate(clim):
        df = _clim_df(cs)
        dates = df.index; m = dates.year.values >= START
        for r in RES:
            cfg = cfgs[r]; ql, qm, a, phi = hyper[r]
            q = df[f"{r}_inflow_cfs"].to_numpy(float)
            yrs, nev, V = annual_events(q[m], dates[m], cfg.r_max_cfs)
            fm, fsd0, _ = particle_filter(nev, V, ql, qm, a, 1500,
                                          np.random.default_rng(1), phi=phi)
            fm = fm + np.log(ATLAS_YEARS)
            T = len(fm)
            for H in HORIZONS:
                fsd = np.sqrt(fsd0 ** 2 + max(H - 1, 0) * (ql + qm))
                for k in range(BURN, T - 1):
                    tf = true_flood(V, k, H)
                    if not np.isfinite(tf):
                        continue
                    rows.append(("flood", r, cs, int(yrs[k]), H,
                                 float(fm[k]), tf, float(fsd[k])))
        print(f"flood clim {ci+1}/{len(clim)}", flush=True)

    # ---- DEMAND: per LULC (reservoir-independent), demand_llt per H -----------
    for ls in lulc:
        dm, dd = load_demand(ls)
        md = dd.year.values >= START
        wy = dd.year.values[md] + (dd.month.values[md] >= 10)
        yrs = np.unique(wy)
        ann = np.array([dm[md][wy == y].mean() for y in yrs])
        for H in HORIZONS:
            mu, sd = demand_llt(ann, H=H)
            for k in range(BURN, len(ann) - 1):
                w = ann[k:k + H]
                if len(w) < H:                      # FULL forward-H window only
                    continue
                rows.append(("demand", ls, ls, int(yrs[k]), H,
                             float(mu[k]), float(w.mean()), float(sd[k])))
        print(f"demand done {ls}", flush=True)

    df = pd.DataFrame(rows, columns=["driver", "unit", "scen", "year", "H",
                                     "mean", "true", "sd"])
    df["z"] = (df["mean"] - df["true"]) / df["sd"].clip(lower=1e-9)
    # accuracy normaliser: spread of realised exposure per (driver,H)
    sd_clim = df.groupby(["driver", "H"])["true"].transform("std")
    df["rawnorm"] = (df["mean"] - df["true"]).abs() / sd_clim.clip(lower=1e-9)

    out = REPO / "results" / "posterior_accuracy.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    print("\nwrote", out, len(df), "rows")

    piv = df.groupby(["driver", "H"]).apply(
        lambda g: pd.Series({"rms_z": np.sqrt(np.mean(g.z ** 2)),
                             "med_rawnorm": g.rawnorm.median()}))
    print("\nby horizon (rms_z = calibration, med_rawnorm = raw accuracy):")
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
