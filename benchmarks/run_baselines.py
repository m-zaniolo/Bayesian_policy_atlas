#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-Bayesian baselines plus the ATLAS oracle, all on the same trajectories and
with the same year-to-year storage chaining as the deployment, so the numbers
join directly on (res, clim, lulc):

    frozen2020    policy fit once on the pre-2020 record, never updated
    ff_15_50      periodic re-optimisation, 50-yr trailing window every 15 yr
    truelookup    atlas policy at the TRUE exposure each year (perfect belief)
    oracle        hindsight-optimal ATLAS policy SEQUENCE (backward coord. descent)

`oracle` is restricted to the atlas; see benchmarks/oracle_freeparam.py and
benchmarks/oracle_perfect.py for the two benchmarks that relax that.

Thin entry point; the implementation lives in bdps/baselines.py.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))

from paths import require_data          # noqa: E402
from baselines import main              # noqa: E402

if __name__ == "__main__":
    require_data()
    main()
