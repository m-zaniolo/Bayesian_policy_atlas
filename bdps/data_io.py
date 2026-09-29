#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/data_io.py -- load Oroville inflow scenarios and historical medians.

All paths default to the layout in this repo:
  scenario_data/cmip5/<scenario>.csv.zip   (97 CMIP5 daily inflow scenarios)
  data/reference/historical_medians.csv     (median release/storage by dowy)
  data/reference/nodes.json                 (Oroville capacity, safe release)
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from model import ReservoirConfig

# Repo root = parent of this file's parent (src/ -> repo)
REPO = Path(__file__).resolve().parent.parent
from paths import (CMIP5_DIR, LULC_DIR, REF_DIR,            # noqa: E402
                   MEDIANS_CSV as DEFAULT_MEDIANS,
                   NODES_JSON as DEFAULT_NODES)

RES_KEY = "ORO"


def water_day(doy: int, year: int) -> int:
    """Day of water year (0-based, Oct 1 = 0), leap-aware. Mirror of upstream."""
    leap = 1 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 0
    if doy >= (274 + leap):
        return doy - (274 + leap)
    return doy + 91


def dowy_from_dates(dates) -> np.ndarray:
    """Leap-aware dowy (0-based, Oct 1 = 0) for a real-calendar daily index.

    USE THIS for any series carrying real dates -- the historical record and the
    CMIP5 projections both do. The tempting shortcut `(arange(n) % 365)` is
    WRONG for them: data/find_input_*.csv spans 1951-2099 and contains 37 leap
    days, so the modular index slips one day earlier per leap year and reshaped
    water-day 146 drifts from 23 Feb (1951) to 17 Jan (2099).

    That is not cosmetic. x1/x2/x3 are TOCS breakpoints expressed IN DOWY, so a
    drifting label smears the seasonal signal the rule curve is fitted against.
    Measured: it pushed the apparent seasonal peak ~18 d late and made every
    policy trained on a clean 365-day calendar (i.e. every FIND- and atlas-
    trained policy) look 15-22 d misaligned when scored on the real record.
    """
    d = pd.DatetimeIndex(dates)
    return np.array([water_day(doy, yr)
                     for doy, yr in zip(d.dayofyear, d.year)], dtype=np.int64)


def dowy_synthetic(n_days: int) -> np.ndarray:
    """Dowy (0-based, Oct 1 = 0) for a synthetic series with exactly 365-day years.

    Correct ONLY for generated series that have no leap days -- FIND output is
    exactly NYEARS*365 and starts 1 Oct, so the modular index is exact there.
    For dated series use dowy_from_dates().
    """
    return np.arange(n_days, dtype=np.int64) % 365


def scenario_names(cmip5_dir: Path = CMIP5_DIR) -> list[str]:
    """All available CMIP5 scenario names (from the .csv.zip filenames)."""
    return sorted(p.name[:-8] for p in Path(cmip5_dir).glob("*.csv.zip"))


def load_inflow(scenario: str, key: str = RES_KEY, cmip5_dir: Path = CMIP5_DIR):
    """Load one scenario's daily inflow for a reservoir.

    Returns (inflow_cfs, dowy, dates) with matching length.
    """
    path = Path(cmip5_dir) / f"{scenario}.csv.zip"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    col = f"{key}_inflow_cfs"
    if col not in df.columns:
        raise KeyError(f"{col} not in {path.name}; columns={list(df.columns)}")
    dowy = np.array([water_day(d, y) for d, y in zip(df.index.dayofyear, df.index.year)],
                    dtype=np.int64)
    inflow = df[col].to_numpy(dtype=np.float64)
    return inflow, dowy, df.index


def load_medians(key: str = RES_KEY, medians_csv: Path = DEFAULT_MEDIANS) -> dict:
    """Median release/storage/inflow by dowy (length 366) for a reservoir."""
    m = pd.read_csv(medians_csv, index_col=0)
    return {
        "R_avg": m[f"{key}_outflow_cfs"].to_numpy(dtype=np.float64),
        "S_avg": m[f"{key}_storage_af"].to_numpy(dtype=np.float64),
        "Q_avg": m[f"{key}_inflow_cfs"].to_numpy(dtype=np.float64),
    }


def reservoir_config(key: str = RES_KEY, nodes_json: Path = DEFAULT_NODES,
                     medians_csv: Path = DEFAULT_MEDIANS) -> ReservoirConfig:
    """Build a ReservoirConfig for any reservoir node key (ORO, BUL, FOL, ...)."""
    nodes = json.loads(Path(nodes_json).read_text())
    r = nodes[key]
    med = load_medians(key, medians_csv)
    return ReservoirConfig(
        capacity_af=float(r["capacity_taf"]) * 1000.0,
        r_max_cfs=float(r["safe_release_cfs"]),
        R_avg=med["R_avg"],
        S_avg=med["S_avg"],
    )


def oroville_config(nodes_json: Path = DEFAULT_NODES,
                    medians_csv: Path = DEFAULT_MEDIANS) -> ReservoirConfig:
    """Backward-compatible alias for reservoir_config('ORO')."""
    return reservoir_config(RES_KEY, nodes_json, medians_csv)


def demand_scenarios(lulc_dir: Path = LULC_DIR) -> list[str]:
    """All available LULC demand scenario names."""
    return sorted(p.name[:-8] for p in Path(lulc_dir).glob("*.csv.zip"))


def load_demand(lulc_scenario: str, lulc_dir: Path = LULC_DIR):
    """System-wide demand multiplier series (combined_demand), aligned to the
    inflow record dates (same 1951-2099 daily index). Returns (dm, dates).

    dm multiplies the reservoir's historical median release to form the demand
    target: D_t = R_median[dowy] * dm[t]. It is ~1.0 pre-2020 and grows after.
    """
    path = Path(lulc_dir) / f"{lulc_scenario}.csv.zip"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["combined_demand"].to_numpy(dtype=np.float64), df.index
