#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single source of truth for every path in the repo. Nothing else should hardcode
a directory.

Layout expected under DATA_ROOT (see data/README.md and data/fetch_data.sh --
the inputs are NOT redistributed here, they come from the upstream repo):

    <DATA_ROOT>/cmip5/<scenario>.csv.zip      daily inflow, 97 CMIP5 scenarios
    <DATA_ROOT>/lulc/<scenario>.csv           demand multiplier scenarios
    <DATA_ROOT>/reference/nodes.json          capacity, safe release per node
    <DATA_ROOT>/reference/historical_medians.csv   median release/storage by dowy
    <DATA_ROOT>/reference/params.json         upstream baseline policy parameters

Override any of these with environment variables:

    BPA_DATA_ROOT   inputs               (default: <repo>/data)
    BPA_RESULTS     generated outputs    (default: <repo>/results)

On a cluster or when the repo lives on a synced drive (Box/Dropbox), point
BPA_DATA_ROOT at fast local disk -- the daily loops open these files constantly.
"""
from __future__ import annotations
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("BPA_DATA_ROOT", REPO / "data"))
RESULTS = Path(os.environ.get("BPA_RESULTS", REPO / "results"))

CMIP5_DIR = DATA_ROOT / "cmip5"
LULC_DIR = DATA_ROOT / "lulc"
REF_DIR = DATA_ROOT / "reference"
NODES_JSON = REF_DIR / "nodes.json"
MEDIANS_CSV = REF_DIR / "historical_medians.csv"

ATLAS_CACHE = RESULTS / "deploy_cache"      # per-reservoir atlas + transfer matrix
DE_CACHE = RESULTS / "de_cache"             # cached DE policy fits (regenerable)

RESULTS.mkdir(parents=True, exist_ok=True)


def require_data() -> None:
    """Fail early with an actionable message if the inputs are not in place."""
    missing = [p for p in (CMIP5_DIR, LULC_DIR, NODES_JSON, MEDIANS_CSV)
               if not p.exists()]
    if missing:
        raise SystemExit(
            "Input data not found:\n  "
            + "\n  ".join(str(p) for p in missing)
            + f"\n\nDATA_ROOT is currently: {DATA_ROOT}\n"
              "These inputs are not redistributed with this repository.\n"
              "Run  bash data/fetch_data.sh  (see data/README.md), or set\n"
              "BPA_DATA_ROOT to a directory holding cmip5/, lulc/ and reference/."
        )
