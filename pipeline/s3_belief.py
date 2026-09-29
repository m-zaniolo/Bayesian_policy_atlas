#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE 1 GUARDRAIL (HANDOFF §5): leave-GCM-out PIT / coverage calibration of the belief.

The make-or-break test. The belief's forward predictive (src/belief.py) is the method; if it
cannot COVER held-out realized exposure with calibrated frequency, the forward-VOI layer will
overfit the GCMs and must be abandoned (the user's original instinct, made testable).

Protocol
--------
For every (flood-GO reservoir x climate scenario), fit the belief over the FULL record ONCE per
width knob q_scale (the KF is causal, so ms[t]/Ps[t] are the posteriors given obs<=t). At a grid
of decision years t and horizons H, form the MC predictive of the next-H-year window-mean
exposure and compute PIT = P(predictive <= realized). Under calibration PIT ~ Uniform(0,1), so
central-p coverage = mean( PIT in [(1-p)/2, (1+p)/2] ).

The width knob q_scale is THE anti-overfit dial. LEAVE-GCM-OUT: for each held-out GCM g, pick
q* on the OTHER 12 GCMs (best 80% coverage) and score coverage on g -- so the width is never
tuned to the GCM it is judged on. If held-out coverage tracks nominal across GCMs -> calibrated,
proceed to Phase 2. If systematically over-confident (coverage << nominal) at a reasonable width
-> STOP.

    python scripts/calibrate_belief.py --reservoirs NML,EXC,DNP,NHG --demand GCAM-26 \
        --horizons 10,20,30 --t-step 4 --n-mc 2000

No differential evolution -- runs in minutes locally.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))
sys.path.insert(0, str(REPO / "bdps"))

from data_io import load_inflow, reservoir_config, load_demand   # noqa: E402
from benchmarks import water_year_bounds                          # noqa: E402
from feature_importance import FUTURE_START_YEAR                  # noqa: E402
from run_stage_c_parallel import stratified_climate               # noqa: E402
from belief import ExposureBelief                                 # noqa: E402

Q_GRID = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]  # process-noise inflation (0.05 = project default)
NOMINAL = [0.5, 0.8, 0.9]


def coverage_from_pit(pit, p):
    pit = np.asarray(pit)
    return float(np.mean((pit >= (1 - p) / 2) & (pit <= (1 + p) / 2)))


def collect(reservoirs, demand, horizons, t_step, n_mc, seed=0):
    """Return records: list of dict(gcm, res, driver, H, q, pit)."""
    dm_full, _ = load_demand(demand)
    climate = stratified_climate(30)
    recs = []
    for res in reservoirs:
        cfg = reservoir_config(res)
        for sc in climate:
            gcm = sc.split("_")[0]
            inflow, dowy, dates = load_inflow(sc, key=res)
            yb = water_year_bounds(dowy)
            yr = np.array([dates.year.values[min(a + 200, len(dates) - 1)] for a, _ in yb])
            ev = int(np.argmax(yr >= FUTURE_START_YEAR))
            dm = dm_full[: len(inflow)]
            T = len(yb)
            for q in Q_GRID:
                bel = ExposureBelief(inflow, dm, yb, ev, q_scale=q)
                rng = np.random.default_rng(seed)
                for H in horizons:
                    for t in range(ev, T - H, t_step):
                        real = bel.realized_window(t, H)
                        for drv, belief_obj, key in (("flood", bel.flood, "flood_win_mean"),
                                                     ("demand", bel.demand, "demand_win_mean")):
                            pred = belief_obj.predict_window(t, H, n_mc=n_mc, rng=rng)["win_mean"]
                            pit = float(np.mean(pred <= real[key]))
                            recs.append({"gcm": gcm, "res": res, "driver": drv,
                                         "H": H, "q": q, "pit": pit})
    return recs


def summarize(recs, horizons):
    R = np.array([(r["gcm"], r["driver"], r["H"], r["q"], r["pit"]) for r in recs], dtype=object)
    gcms = sorted(set(r["gcm"] for r in recs))
    out = {"drivers": {}, "gcms": gcms, "q_grid": Q_GRID}
    for driver in ("flood", "demand"):
        out["drivers"][driver] = {}
        for H in horizons:
            sel = [r for r in recs if r["driver"] == driver and r["H"] == H]
            byq = {q: np.array([r["pit"] for r in sel if r["q"] == q]) for q in Q_GRID}
            # global width: best 80% coverage over all data
            q_star = min(Q_GRID, key=lambda q: abs(coverage_from_pit(byq[q], 0.8) - 0.8))
            glob = {"q_star": q_star,
                    "coverage": {str(p): coverage_from_pit(byq[q_star], p) for p in NOMINAL},
                    "pit_mean": float(byq[q_star].mean()), "n": int(byq[q_star].size)}
            # LEAVE-GCM-OUT: choose q on the other GCMs, score on the held-out one
            logo = []
            for g in gcms:
                other = {q: np.array([r["pit"] for r in sel if r["q"] == q and r["gcm"] != g])
                         for q in Q_GRID}
                held = {q: np.array([r["pit"] for r in sel if r["q"] == q and r["gcm"] == g])
                        for q in Q_GRID}
                if held[Q_GRID[0]].size == 0:
                    continue
                q_g = min(Q_GRID, key=lambda q: abs(coverage_from_pit(other[q], 0.8) - 0.8))
                logo.append({"gcm": g, "q": q_g, "n": int(held[q_g].size),
                             "cov80": coverage_from_pit(held[q_g], 0.8),
                             "cov90": coverage_from_pit(held[q_g], 0.9)})
            cov80 = np.array([x["cov80"] for x in logo])
            out["drivers"][driver][str(H)] = {
                "global": glob, "logo": logo,
                "logo_cov80_median": float(np.median(cov80)),
                "logo_cov80_min": float(cov80.min()), "logo_cov80_max": float(cov80.max()),
                "pit_q_star_values": byq[q_star].tolist()}
    return out


def verdict(summary, horizons):
    print("\n" + "=" * 78)
    print("PHASE-1 CALIBRATION VERDICT (leave-GCM-out; nominal 80% central interval)")
    print("=" * 78)
    ok = True
    for driver in ("flood", "demand"):
        print(f"\n-- {driver.upper()} --")
        for H in horizons:
            d = summary["drivers"][driver][str(H)]
            g = d["global"]
            med = d["logo_cov80_median"]
            flag = "OK  " if 0.70 <= med <= 0.90 else ("WIDE" if med > 0.90 else "OVERCONF")
            if driver == "flood" and not (0.68 <= med <= 0.92):
                ok = False
            print(f"  H={H:2d} | global q*={g['q_star']:<4} cov(50/80/90)="
                  f"{g['coverage']['0.5']:.2f}/{g['coverage']['0.8']:.2f}/{g['coverage']['0.9']:.2f}"
                  f" PITmean={g['pit_mean']:.2f} | LOGO cov80 med={med:.2f}"
                  f" [{d['logo_cov80_min']:.2f},{d['logo_cov80_max']:.2f}]  {flag}")
    print("\nGATE:", "PASS -> proceed to Phase 2 (predictive generator)." if ok else
          "FAIL -> belief over-confident OOD; widen knob or STOP (would overfit GCMs).")
    return ok


def make_figure(summary, recs, horizons, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nH = len(horizons)
    fig, ax = plt.subplots(2, nH, figsize=(4 * nH, 7), squeeze=False)
    for j, H in enumerate(horizons):
        d = summary["drivers"]["flood"][str(H)]
        pit = np.array(d["pit_q_star_values"])
        ax[0, j].hist(pit, bins=10, range=(0, 1), color="#4C72B0", edgecolor="w")
        ax[0, j].axhline(len(pit) / 10, ls="--", c="k", lw=1)
        ax[0, j].set_title(f"FLOOD PIT  H={H}  (q*={d['global']['q_star']})")
        ax[0, j].set_xlabel("PIT"); ax[0, j].set_ylabel("count")
        cov = [d["global"]["coverage"][str(p)] for p in NOMINAL]
        ax[1, j].plot(NOMINAL, NOMINAL, "k--", lw=1, label="ideal")
        ax[1, j].plot(NOMINAL, cov, "o-", c="#4C72B0", label="global")
        c80 = np.array([x["cov80"] for x in d["logo"]])
        ax[1, j].scatter([0.8] * len(c80), c80, c="#C44E52", s=18, alpha=.7, label="LOGO held-out")
        ax[1, j].set_xlim(0.4, 1); ax[1, j].set_ylim(0.3, 1)
        ax[1, j].set_xlabel("nominal coverage"); ax[1, j].set_ylabel("empirical")
        ax[1, j].legend(fontsize=7)
    fig.suptitle("Phase-1 belief calibration (flood exposure) — PIT uniformity + leave-GCM-out coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reservoirs", default="NML,EXC,DNP,NHG")
    ap.add_argument("--demand", default="GCAM-26")
    ap.add_argument("--horizons", default="10,20,30")
    ap.add_argument("--t-step", type=int, default=4)
    ap.add_argument("--n-mc", type=int, default=2000)
    args = ap.parse_args()
    reservoirs = args.reservoirs.split(",")
    horizons = [int(h) for h in args.horizons.split(",")]
    print(f"calibrate belief | res={reservoirs} demand={args.demand} H={horizons} "
          f"t_step={args.t_step} n_mc={args.n_mc} q_grid={Q_GRID}", flush=True)
    recs = collect(reservoirs, args.demand, horizons, args.t_step, args.n_mc)
    print(f"collected {len(recs)} PIT records", flush=True)
    summary = summarize(recs, horizons)
    ok = verdict(summary, horizons)
    outdir = REPO / "results"
    (outdir / "belief_calibration.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {outdir/'belief_calibration.json'}")
    make_figure(summary, recs, horizons, outdir / "belief_calibration.png")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
