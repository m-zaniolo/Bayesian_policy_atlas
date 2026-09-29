#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/bayes_atlas.py -- the Bayesian belief machine over the policy atlas.

WHAT MAKES THIS ACTUALLY BAYESIAN (see memory/precise-terminology.md: "Bayesian
= maintains AND USES a posterior in the decision"). All four must hold:

  1. AN EXPLICIT GENERATIVE MODEL, stated and testable
         z_y  = theta_y + eps,      eps ~ N(0, R)          observation
         theta_y = theta_{y-1} + b_{y-1} + w,  w ~ N(0, Q_l)   latent level
         b_y     = b_{y-1} + u,                u ~ N(0, Q_s)   latent drift
     z_y = log(1 + annual volume above safe release). Linear + Gaussian, so the
     Kalman recursion returns the EXACT posterior p(theta_y | z_{1:y}) -- this
     is exact Bayesian inference, not an approximation to it.

  2. HYPERPARAMETERS FROM THE MARGINAL LIKELIHOOD, not hand-tuned.
     (Q_l, Q_s, R) maximise p(z_{1:T}) via the prediction-error decomposition.
     A KF with knobs picked by eye is a smoother; estimating them from the
     evidence is what makes the output a posterior. This is empirical Bayes,
     and it is labelled as such -- the hyperparameters are point estimates, so
     hyperparameter uncertainty is NOT propagated.

  3. THE POSTERIOR IS CALIBRATED, and the check can fail.
     PIT of the one-step-ahead predictive on HELD-OUT trajectories must be
     uniform; coverage of the 50/90% intervals must hit nominal. An
     uncalibrated posterior is not a posterior in any usable sense, so this
     script reports the check rather than assuming it.

  4. THE POSTERIOR ENTERS THE DECISION -- the part that is usually skipped.
         BAYES ACTION   x* = argmin_x  E_{theta ~ p(theta|z)} [ J(x, theta) ]
         PLUG-IN        x  = atlas( E[theta] )
     These differ precisely because flood loss is ASYMMETRIC: under-sizing the
     flood pool is catastrophic, over-sizing is mildly costly, so integrating
     over the posterior pulls the action conservative. If they did not differ,
     the machinery would not be earning its keep -- so the script reports the
     gap as its headline, and reports it honestly when it is ~0.

  Candidate actions are WHOLE atlas policies, never per-parameter predictions:
  leave-one-out showed assembling 7 independently-predicted parameters costs
  10.2% excess vs 3.1% for keeping the vector intact (HANDOFF §7.1 is the same
  failure mode -- a coordinate-wise median of optima is optimal for nothing).

USAGE
    python scripts/bayes_atlas.py --res NHG
    python scripts/bayes_atlas.py --res NHG,DNP,BUL --n-climate 8
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

from data_io import load_inflow, load_demand, reservoir_config, scenario_names  # noqa: E402
from model import objective, PARAM_NAMES                                        # noqa: E402

CFSD_TO_AF = 1.98347
COORD = "vol_above_safe_af"


# ---------------------------------------------------------------------------
# 1. observations
# ---------------------------------------------------------------------------
def annual_excess(q, dates, safe):
    """log(1 + volume above safe release) per water year -- the observation.

    Log because annual excess volume is zero-inflated and heavy-tailed; the
    +1 keeps dry years finite. The Gaussian observation model on this scale is
    an APPROXIMATION to a zero-inflated process, which is exactly why step 3
    (PIT calibration) exists rather than being assumed away.
    """
    wy = dates.year.values + (dates.month.values >= 10)
    exc = np.maximum(q - safe, 0.0) * CFSD_TO_AF
    yrs = np.unique(wy)
    return yrs, np.array([np.log1p(exc[wy == y].sum()) for y in yrs])


# ---------------------------------------------------------------------------
# 2. exact posterior + marginal likelihood
# ---------------------------------------------------------------------------
def kalman(z, q_l, q_s, R, m0=None, P0=None):
    """Local-linear-trend KF. Returns filtered posterior, 1-step predictive, logL.

    For a linear-Gaussian model this recursion IS the posterior -- no
    approximation. logL is the exact marginal likelihood via the
    prediction-error decomposition, which is what step 2 maximises.
    """
    T = len(z)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([q_l, q_s])
    m = np.array([z[0] if m0 is None else m0, 0.0])
    P = np.diag([1.0, 0.1]) if P0 is None else P0
    mf, Pf, pm, pv, logL = np.empty((T, 2)), np.empty((T, 2, 2)), np.empty(T), np.empty(T), 0.0
    for t in range(T):
        m = F @ m
        P = F @ P @ F.T + Q                      # predict
        s = float(H @ P @ H.T) + R               # predictive variance
        yhat = float(H @ m)
        pm[t], pv[t] = yhat, s
        r = z[t] - yhat
        logL += -0.5 * (np.log(2 * np.pi * s) + r * r / s)
        K = (P @ H.T) / s                        # update
        m = m + (K * r).ravel()
        P = P - K @ H @ P
        mf[t], Pf[t] = m, P
    return mf, Pf, pm, pv, logL


def fit_hyperparams(z_list, seed=0):
    """(q_l, q_s, R) by maximum marginal likelihood over training trajectories.

    Empirical Bayes: point estimates, so hyperparameter uncertainty is not
    propagated. Stated plainly rather than glossed.
    """
    from scipy.optimize import minimize

    def nll(p):
        q_l, q_s, R = np.exp(p)
        return -sum(kalman(z, q_l, q_s, R)[4] for z in z_list)

    best, bl = None, np.inf
    rng = np.random.default_rng(seed)
    for _ in range(6):
        p0 = np.log([10 ** rng.uniform(-4, -1), 10 ** rng.uniform(-6, -3),
                     10 ** rng.uniform(-2, 1)])
        r = minimize(nll, p0, method="Nelder-Mead",
                     options=dict(maxiter=2000, xatol=1e-4, fatol=1e-4))
        if r.fun < bl:
            bl, best = r.fun, r.x
    return tuple(np.exp(best))


def calibration(z_list, hp):
    """PIT + interval coverage of the one-step-ahead predictive (held-out)."""
    pits = []
    for z in z_list:
        _, _, pm, pv, _ = kalman(z, *hp)
        from math import erf
        pits += [0.5 * (1 + erf((z[t] - pm[t]) / np.sqrt(2 * pv[t])))
                 for t in range(3, len(z))]      # skip burn-in
    p = np.array(pits)
    # KS distance against U(0,1)
    ks = float(np.max(np.abs(np.sort(p) - np.linspace(0, 1, len(p)))))
    return dict(n=len(p), ks=ks, mean=float(p.mean()),
                cov50=float(((p > .25) & (p < .75)).mean()),
                cov90=float(((p > .05) & (p < .95)).mean()))


# ---------------------------------------------------------------------------
# 3. decision
# ---------------------------------------------------------------------------
def build_transfer(res, atlas, demand_tol=0.08):
    """J[i, j] = cost of atlas policy i deployed on atlas scenario j.

    The exact loss surface over the candidate set, so the expectation in the
    Bayes action is computed rather than approximated by a surrogate.
    """
    cfg = reservoir_config(res)
    n = len(atlas)
    X = atlas[PARAM_NAMES].to_numpy()
    J = np.empty((n, n))
    cache = {}
    for j in range(n):
        f = str(REPO / "scenario_data" / "find_flood" / res / atlas.iloc[j]["file"])
        if f not in cache:
            cache[f] = pd.read_csv(f)["inflow_cfs"].to_numpy(dtype=float)
        q = cache[f]
        dowy = np.arange(q.size, dtype=np.int64) % 365   # FIND: exact, 0-based
        dm = np.full(q.size, atlas.iloc[j]["demand"])
        for i in range(n):
            J[i, j] = objective(q, X[i], cfg, dowy, DM=dm, S0=cfg.S_avg[-1])
    return J


def posterior_weights(atlas, mu, var, demand_now, demand_sd=0.05):
    """p(atlas point | data): Gaussian belief on the flood coordinate.

    Demand is treated as observed with small noise -- it is a scenario input,
    not something inferred from streamflow. Only the flood coordinate carries a
    genuine posterior, and that is stated rather than implied.
    """
    c = np.log1p(atlas[COORD].to_numpy())
    w = np.exp(-0.5 * (c - mu) ** 2 / var) / np.sqrt(var)
    w *= np.exp(-0.5 * ((atlas["demand"].to_numpy() - demand_now) / demand_sd) ** 2)
    s = w.sum()
    return w / s if s > 0 else np.full(len(w), 1.0 / len(w))


def decide(atlas, J, w):
    """Bayes action vs plug-in. THE step that earns the word 'Bayesian'.

    bayes  : argmin_i  sum_j w_j J[i, j]      -- expected loss under the posterior
    plugin : policy at the atlas point nearest the posterior MEAN (discards spread)
    """
    exp_loss = J @ w
    i_bayes = int(np.argmin(exp_loss))
    c = np.log1p(atlas[COORD].to_numpy())
    i_plug = int(np.argmin(np.abs(c - (w * c).sum())))
    return i_bayes, i_plug, exp_loss


# ---------------------------------------------------------------------------
def run(res, n_climate, n_train, verbose=True):
    cfg = reservoir_config(res)
    atlas = pd.read_csv(REPO / "results" / "find_atlas_fits.csv")
    atlas = atlas[atlas.res == res].reset_index(drop=True)
    if atlas.empty:
        print(f"{res}: no atlas fits"); return None

    clim = scenario_names()[:n_climate]
    Z = []
    for cs in clim:
        q, _, dates = load_inflow(cs, key=res)
        m = dates.year.values >= 2020
        _, z = annual_excess(q[m], dates[m], cfg.r_max_cfs)
        Z.append(z)

    hp = fit_hyperparams(Z[:n_train])
    cal = calibration(Z[n_train:], hp)          # HELD OUT
    if verbose:
        print(f"\n{'='*76}\n{res}\n{'='*76}")
        print(f"marginal-likelihood hyperparameters: q_level={hp[0]:.2e} "
              f"q_slope={hp[1]:.2e} R={hp[2]:.3f}")
        print(f"held-out calibration (n={cal['n']}): KS={cal['ks']:.3f}  "
              f"PIT mean={cal['mean']:.3f}  50%={cal['cov50']:.2f}  "
              f"90%={cal['cov90']:.2f}")
        bad = cal['ks'] > 0.15 or not (0.40 < cal['cov50'] < 0.60)
        print("  -> " + ("MISCALIBRATED: posterior width is wrong; the decision "
                         "numbers below inherit that error"
                         if bad else "calibrated"))

    if verbose:
        print("\nbuilding transfer matrix (exact loss over candidate policies)...",
              flush=True)
    J = build_transfer(res, atlas)

    rows = []
    for k, cs in enumerate(clim[n_train:]):
        q, _, dates = load_inflow(cs, key=res)
        m = dates.year.values >= 2020
        yrs, z = annual_excess(q[m], dates[m], cfg.r_max_cfs)
        mf, Pf, _, _, _ = kalman(z, *hp)
        dm_series, ddates = load_demand("LUCAS-BAU_Low-0010")
        for t in range(10, len(z)):
            mu, var = mf[t, 0], Pf[t, 0, 0] + hp[2]
            w = posterior_weights(atlas, mu, var, 1.25)
            ib, ip, el = decide(atlas, J, w)
            rows.append(dict(res=res, scen=cs, year=int(yrs[t]),
                             mu=mu, sd=np.sqrt(var),
                             x4_bayes=atlas.iloc[ib]["x4"],
                             x4_plug=atlas.iloc[ip]["x4"],
                             exp_bayes=el[ib], exp_plug=el[ip],
                             same=int(ib == ip)))
    R = pd.DataFrame(rows)
    gap = 100 * (R.exp_plug - R.exp_bayes) / R.exp_bayes
    if verbose:
        print(f"\ndecisions along {len(clim)-n_train} held-out trajectories "
              f"(n={len(R)} timesteps)")
        print(f"  posterior sd on log-coordinate : {R.sd.mean():.3f}")
        print(f"  Bayes action == plug-in        : {100*R.same.mean():.0f}% of steps")
        print(f"  expected-loss gap (plug-in worse): median {np.median(gap):+.2f}%  "
              f"p90 {np.percentile(gap,90):+.2f}%")
        print(f"  x4 deployed: Bayes {R.x4_bayes.mean():.3f} vs "
              f"plug-in {R.x4_plug.mean():.3f}  "
              f"(Bayes {'more' if R.x4_bayes.mean()>R.x4_plug.mean() else 'less'} "
              f"conservative)")
    return R, cal, hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="NHG")
    ap.add_argument("--n-climate", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "results" / "bayes_atlas.csv"))
    args = ap.parse_args()
    out = []
    for r in [s.strip() for s in args.res.split(",")]:
        got = run(r, args.n_climate, args.n_train)
        if got:
            out.append(got[0])
    if out:
        pd.concat(out).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
