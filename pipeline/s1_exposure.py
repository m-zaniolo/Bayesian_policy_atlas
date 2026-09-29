#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/export_find_input.py -- hand a reservoir's daily inflow to FIND.

FIND is MATLAB and reads a plain CSV (date, inflow_cfs). This writes that file
and prints the physical event statistics FIND/main_flood_sweep.m will target,
so the two sides can be checked against each other before a long SA run.

Event definition is INFLOW ABOVE SAFE RELEASE, not a percentile: a percentile
gives every reservoir the same event rate by construction and so says nothing
about where the flood side actually binds. Measured over 479 scenario-years,
the physical definition separates NML (1.0/yr) from ORO (1 per 53 yr) by 50x.

USAGE
    python scripts/export_find_input.py --res NML
    python scripts/export_find_input.py --res NML --scen access1-0_rcp85_r1i1p1
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))
sys.path.insert(0, str(REPO / "bdps"))

from data_io import load_inflow, reservoir_config, scenario_names   # noqa: E402

MIN_DUR, GAP = 3, 2


def physical_events(q, thr, min_dur=MIN_DUR, gap=GAP):
    """Runs of inflow above an absolute threshold, tolerating `gap` dips."""
    hi = q > thr
    durs, i, n = [], 0, q.size
    while i < n:
        if hi[i]:
            j, g = i, 0
            while j + 1 < n and (hi[j + 1] or g < gap):
                g = 0 if hi[j + 1] else g + 1
                j += 1
            j -= g
            if j - i + 1 >= min_dur:
                durs.append(j - i + 1)
            i = j + 1
        else:
            i += 1
    return np.asarray(durs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="NML")
    ap.add_argument("--scen", default=None,
                    help="climate scenario to export (default: first available)")
    ap.add_argument("--start-year", type=int, default=None,
                    help="clip to this year onward (default: full record)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scen = args.scen or scenario_names()[0]
    cfg = reservoir_config(args.res)
    inflow, dowy, dates = load_inflow(scen, key=args.res)

    if args.start_year is not None:
        m = dates.year.values >= args.start_year
        inflow, dates = inflow[m], dates[m]

    out = Path(args.out or (REPO / "data" / f"find_input_{args.res}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
                  "inflow_cfs": inflow}).to_csv(out, index=False)

    nyr = len(np.unique(dates.year.values))
    durs = physical_events(inflow, cfg.r_max_cfs)
    freq = len(durs) / nyr

    print(f"wrote {out}  ({inflow.size} days, {nyr} years, scenario {scen})")
    print(f"\n--- targets for FIND/main_flood_sweep.m ({args.res}) ---")
    print(f"  safe release            : {cfg.r_max_cfs:,.0f} cfs")
    print(f"  events (>safe, >={MIN_DUR}d)  : {len(durs)} in {nyr} yr "
          f"= {freq:.3f}/yr (1 per {1/max(freq,1e-9):.1f} yr)")
    if len(durs):
        print(f"  mean event duration     : {durs.mean():.1f} d "
              f"(median {np.median(durs):.0f}, max {durs.max()})")
    print(f"  target_nevents per 50 yr: {max(1, round(freq * 50))}")

    if freq < 0.05:
        print(f"\n  !! WARNING: the flood constraint binds less than once per 20 "
              f"years at {args.res}.\n     This is the ORO failure mode -- the "
              f"optimal policy is near-stationary and\n     no flood atlas can "
              f"help. Pick a reservoir where the constraint binds.")


if __name__ == "__main__":
    main()
