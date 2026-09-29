#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SI table: effective memory of the flood filter per reservoir, at the DEPLOYED Q.

The deployed random-walk process noise Q = (Q_lambda, Q_mu) sets how fast the
particle filter forgets. We summarise it as an EWMA-equivalent gain K, estimated
empirically from the pre-2020 record by regressing the filtered-mean increment on
the innovation:

    fm[t] - fm[t-1] = K (z[t] - fm[t-1]) + noise      ->  K = <dfm, innov>/<innov, innov>

The filter's impulse response then decays as (1-K)^lag, giving

    memory (EWMA mean lag) = 1/K
    half-life              = ln(0.5) / ln(1-K)         # years to forget half a shock

This is robust for all 9 reservoirs (unlike the earlier Gaussian-kernel fit, which
returned 0 at FOL/ORO where the belief is near-static, var~0). Deployed Q is read
from outputs/learning_rates.csv so the numbers match the actual deployment.

Writes outputs/memory_table.csv and outputs/memory_table.tex.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))
sys.path.insert(0, str(REPO / "bdps"))
import deploy as bd                                                # noqa: E402
from belief import particle_filter                                   # noqa: E402

OUT = REPO / "results"
RES = ["NML", "EXC", "NHG", "BUL", "FOL", "SHA", "MIL", "ORO", "DNP"]
FULL = {"NML": "New Melones", "EXC": "Exchequer", "NHG": "New Hogan",
        "BUL": "New Bullards Bar", "FOL": "Folsom", "SHA": "Shasta",
        "MIL": "Millerton", "ORO": "Oroville", "DNP": "Don Pedro"}


def impulse_response(res, ql, qm, a, phi, burn=60, after=160, n_part=4000, seeds=5):
    """Impulse-response memory of the flood filter at this deployed Q.

    Effective memory is a property of the FILTER, not of whether the observed
    record happens to contain a flood. We therefore drive the deployed filter
    with a synthetic sequence -- a long no-flood baseline (n=0, V=0), a single
    flood-event shock, then no-flood years again -- and measure how fast the
    posterior mean of log(lambda*mu) decays back to baseline.

    The decay is geometric, dev[lag] ~ peak * exp(slope*lag); we fit slope by OLS
    on log-deviation over the 90%..10%-of-peak window (averaged over seeds) and
    return (gain K, memory 1/K, half-life), with
        K = 1 - exp(slope),  half-life = ln(0.5)/slope.
    Robust for all reservoirs, including FOL/ORO where real floods never bind.
    """
    n_hist, V_hist = bd.historical_obs(res)
    Vpos = V_hist[V_hist > 0]
    Vshock = float(Vpos.max()) if Vpos.size else float(max(V_hist.max(), 1.0) * 10)
    T = burn + 1 + after
    n = np.zeros(T); V = np.zeros(T)
    n[burn] = 1.0; V[burn] = Vshock                       # single flood-event shock
    slopes = []
    for s in range(seeds):
        fm, _, _ = particle_filter(n, V, ql, qm, a, n_part,
                                   np.random.default_rng(100 + s), phi=phi)
        base = float(fm[burn - 1])                        # pre-shock steady level
        dev = fm[burn:] - base                            # deviation from shock onward
        peak = float(dev[0])
        if peak <= 1e-6:
            continue
        lag = np.arange(len(dev))
        keep = (dev > 0.10 * peak) & (dev < 0.90 * peak) & (lag > 0)
        if keep.sum() < 3:
            keep = (dev > 0.05 * peak) & (lag > 0)
        if keep.sum() < 3:
            continue
        slopes.append(np.polyfit(lag[keep], np.log(dev[keep]), 1)[0])
    if not slopes:
        return np.nan, np.nan, np.nan
    slope = float(np.mean(slopes))                        # < 0
    K = 1.0 - np.exp(slope)
    return K, 1.0 / K, np.log(0.5) / slope


def main():
    lr = pd.read_csv(OUT / "learning_rates.csv").set_index("res")
    rows = []
    for res in RES:
        r = lr.loc[res]
        ql, qm, phi, a = float(r.Q_lambda), float(r.Q_m), float(r.phi), float(r.gamma_a)
        K, mem, half = impulse_response(res, ql, qm, a, phi)
        rows.append(dict(res=res, name=FULL[res], Q_lambda=ql, Q_mu=qm,
                         gain_K=K, memory_yr=mem, halflife_yr=half))
        print(f"  {res:>4}  Q=({ql:.4f},{qm:.4f})  K={K:6.4f}  "
              f"memory={mem:6.1f}yr  half-life={half:5.1f}yr", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "memory_table.csv", index=False)
    print("wrote", OUT / "memory_table.csv")

    # LaTeX (ordered longest-memory first)
    df = df.sort_values("halflife_yr", ascending=False)
    lines = [r"\begin{table}[h]", r"\centering",
             r"\caption{Effective memory of the flood belief at the deployed process "
             r"noise $Q$, measured by the impulse response of the filter to a single "
             r"flood-event shock. $K$ is the equivalent forgetting gain; the half-life "
             r"is the number of years for the shock's effect on the posterior mean to "
             r"decay by half.}",
             r"\label{tab:memory}",
             r"\begin{tabular}{llccc}", r"\toprule",
             r"Reservoir & & $K$ & Memory (yr) & Half-life (yr) \\", r"\midrule"]
    def fmt(v):
        return "--" if not np.isfinite(v) else f"{v:.0f}"
    for _, r in df.iterrows():
        lines.append(f"{r['name']} & {r.res} & {r.gain_K:.3f} & "
                     f"{fmt(r.memory_yr)} & {fmt(r.halflife_yr)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "memory_table.tex").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "memory_table.tex")


if __name__ == "__main__":
    main()
