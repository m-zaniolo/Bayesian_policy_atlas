#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4 -- BAYESIAN ACTION SELECTION.

Turns the posterior over the climate-demand exposure space into an actual
release decision, by three rules:

    mean-exposure (plug-in)  atlas policy at the posterior MEAN (discards spread)
    posterior-optimal        argmin_i E_w[J_i] over the posterior
    posterior-averaged       posterior-expected DECISION: release = sum_j w_j r_j
                             (a weighted average of RELEASES, not of parameters)

Thin entry point; the implementation lives in bdps/deploy.py.
Run `python pipeline/s4_action.py --help` for options.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))

from paths import require_data          # noqa: E402
from deploy import main                 # noqa: E402

if __name__ == "__main__":
    require_data()
    main()
