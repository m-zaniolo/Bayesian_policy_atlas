#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/bayes_rate.py -- Bayesian belief over the flood REGIME, as a rate.

WHY A RATE MODEL (this replaces the Gaussian observation layer in bayes_atlas.py)
    The regime is a decadal-scale property: "how often, and how big, are the
    floods I am exposed to". Modelling log(annual excess volume) as Gaussian
    fails because annual excess is EXACTLY ZERO in 34-65% of years -- the
    Gaussian likelihood cannot represent a point mass, so the fitted
    observation variance inflates to absorb it (R=23.9 at NHG) and the posterior
    becomes uselessly wide (sd 5.2 on a log coordinate that spans the atlas).
    Its held-out PIT failed: KS=0.186, 50%-coverage 0.39 against nominal 0.50.

    Here a zero year is DATA, not a defect: it is a Poisson zero, and it is
    informative about the rate.

MODEL  (compound Poisson-Gamma; both parts of the observation are used)
    state    s_y = [log lambda_y, log mu_y]        rate, mean event magnitude
             s_y = s_{y-1} + w,   w ~ N(0, diag(Q_lam, Q_mag))    slow drift
    observe  n_y ~ Poisson(lambda_y)                              how many
             V_y | n_y>0 ~ Gamma(shape = n_y*a, scale = mu_y/a)   how big
             n_y = 0 contributes exp(-lambda_y) -- a real likelihood term.

    Expected annual excess volume is lambda_y * mu_y, which is exactly the
    atlas coordinate (per year), so the belief lands in atlas space directly.

    Both rate AND magnitude drift. Modelling only the rate would misspecify a
    warming climate, where event size grows too.

INFERENCE
    Bootstrap particle filter. Non-conjugate (Gaussian random walk + Poisson /
    Gamma observations), so there is no closed form; the PF is asymptotically
    exact as particles -> infinity, which is a documented approximation rather
    than a hidden one. Hyperparameters (Q_lam, Q_mag) by maximising the PF
    estimate of the marginal likelihood under COMMON RANDOM NUMBERS, so the
    objective is smooth enough to optimise. The Gamma shape `a` is fitted
    offline by MLE on training-set event magnitudes.

WHAT WOULD FALSIFY IT
    Held-out PIT of the one-step-ahead predictive for n_y (randomised PIT, since
    counts are discrete) must be uniform, and interval coverage must hit
    nominal. Reported, not assumed. At NHG (~2 events per DECADE) the posterior
    should stay genuinely wide -- that is Poisson counting noise, not a bug, and
    it is the honest version of what the Gaussian model was crudely signalling.

USAGE
    python scripts/bayes_rate.py --res DNP
    python scripts/bayes_rate.py --res DNP,NML,NHG --n-particles 4000
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats, optimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps"))
sys.path.insert(0, str(REPO / "bdps"))

from data_io import load_inflow, reservoir_config, scenario_names   # noqa: E402

CFSD_TO_AF = 1.98347
MIN_DUR, GAP = 3, 2


# ---------------------------------------------------------------------------
# observations: per water year, how many flood events and how much volume
# ---------------------------------------------------------------------------
def annual_events(q, dates, safe):
    """(n_y, V_y) per water year: event count and total excess volume (af).

    Event = inflow above safe release for >= MIN_DUR days, tolerating GAP dips
    -- the same physical definition used to build the atlas.
    """
    wy = dates.year.values + (dates.month.values >= 10)
    hi = q > safe
    exc = np.maximum(q - safe, 0.0) * CFSD_TO_AF
    yrs = np.unique(wy)
    n = np.zeros(len(yrs), dtype=int)
    V = np.zeros(len(yrs))
    for k, y in enumerate(yrs):
        m = wy == y
        h, e = hi[m], exc[m]
        V[k] = e.sum()
        i, N = 0, h.size
        while i < N:
            if h[i]:
                j, g = i, 0
                while j + 1 < N and (h[j + 1] or g < GAP):
                    g = 0 if h[j + 1] else g + 1
                    j += 1
                j -= g
                if j - i + 1 >= MIN_DUR:
                    n[k] += 1
                i = j + 1
            else:
                i += 1
    # a year with volume but no qualifying event still had exceedance days;
    # count it as one event so V is never orphaned from n
    n[(n == 0) & (V > 0)] = 1
    return yrs, n, V


# ---------------------------------------------------------------------------
# particle filter over [log lambda, log mu]
# ---------------------------------------------------------------------------
def loglik_obs(n, V, lam, mu, a, phi=None):
    """log p(n, V | lambda, mu). Zeros are a real likelihood term, not a defect.

    Counts are NEGATIVE BINOMIAL with mean lambda and dispersion phi
    (var = lambda + lambda^2/phi), which NESTS Poisson as phi -> inf. Flood
    counts are overdispersed -- measured var/mean is 1.58 at DNP and 1.46 at
    MIL, because wet years deliver several events. Under a Poisson likelihood
    the one-step predictive is then too narrow and held-out PIT coverage falls
    below nominal (observed: 50% -> 0.32, 90% -> 0.78). phi is estimated from
    the marginal likelihood, so if counts really are Poisson the data say so.
    """
    if phi is None or not np.isfinite(phi) or phi > 1e6:
        ll = n * np.log(np.maximum(lam, 1e-12)) - lam - special.gammaln(n + 1)
    else:
        lam = np.maximum(lam, 1e-12)
        ll = (special.gammaln(n + phi) - special.gammaln(phi)
              - special.gammaln(n + 1)
              + phi * np.log(phi / (phi + lam))
              + n * np.log(lam / (phi + lam)))
    if n > 0 and V > 0:
        shape = n * a
        scale = np.maximum(mu, 1e-9) / a
        ll = ll + ((shape - 1) * np.log(V) - V / scale
                   - special.gammaln(shape) - shape * np.log(scale))
    return ll


def particle_filter(n_obs, V_obs, q_lam, q_mag, a, n_part=3000, rng=None,
                    s0=None, noise=None, phi=None):
    """Bootstrap PF. Returns per-year posterior mean/sd of log(lambda*mu) and logL.

    `noise` lets the caller supply pre-drawn standard normals (common random
    numbers) so the marginal-likelihood surface is smooth in the hyperparameters.
    """
    rng = rng or np.random.default_rng(0)
    T = len(n_obs)
    if s0 is None:
        lam0 = max(n_obs.mean(), 0.05)
        mu0 = max(V_obs[V_obs > 0].mean() if (V_obs > 0).any() else 1.0, 1.0)
        s0 = np.array([np.log(lam0), np.log(mu0)])
    S = s0 + np.column_stack([rng.normal(0, 0.5, n_part),
                              rng.normal(0, 0.5, n_part)])
    m_out, sd_out, logL = np.empty(T), np.empty(T), 0.0
    for t in range(T):
        eps = (noise[t] if noise is not None
               else np.column_stack([rng.normal(0, 1, n_part),
                                     rng.normal(0, 1, n_part)]))
        S = S + eps * np.array([np.sqrt(q_lam), np.sqrt(q_mag)])   # evolve
        lam, mu = np.exp(S[:, 0]), np.exp(S[:, 1])
        lw = np.array([loglik_obs(n_obs[t], V_obs[t], lam[i], mu[i], a, phi)
                       for i in range(n_part)])
        mx = lw.max()
        w = np.exp(lw - mx)
        sw = w.sum()
        if not np.isfinite(sw) or sw <= 0:            # filter collapse
            w = np.ones(n_part); sw = n_part
        logL += mx + np.log(sw / n_part)
        w /= sw
        ann = np.log(lam * mu)                        # log expected annual volume
        m_out[t] = float(w @ ann)
        sd_out[t] = float(np.sqrt(w @ (ann - m_out[t]) ** 2))
        idx = rng.choice(n_part, n_part, p=w)         # resample
        S = S[idx]
    return m_out, sd_out, logL


def fit_hyper(train, a, n_part=1500, seed=0):
    """(Q_lam, Q_mag) by maximum marginal likelihood, common random numbers."""
    rng = np.random.default_rng(seed)
    Tmax = max(len(n) for n, V in train)
    noise = [np.column_stack([rng.normal(0, 1, n_part), rng.normal(0, 1, n_part)])
             for _ in range(Tmax)]

    def nll(p):
        q_lam, q_mag, phi = np.exp(p)
        tot = 0.0
        for n, V in train:
            tot += particle_filter(n, V, q_lam, q_mag, a, n_part,
                                   np.random.default_rng(seed), noise=noise,
                                   phi=phi)[2]
        return -tot

    r = optimize.minimize(nll, np.log([0.02, 0.02, 4.0]), method="Nelder-Mead",
                          options=dict(maxiter=90, xatol=0.05, fatol=1.0))
    return tuple(np.exp(r.x))


def fit_gamma_shape(train):
    """MLE Gamma shape on per-event magnitudes (training set only)."""
    mags = []
    for n, V in train:
        m = (n > 0) & (V > 0)
        mags += list(V[m] / n[m])
    mags = np.array(mags)
    if len(mags) < 5:
        return 1.0
    return float(stats.gamma.fit(mags, floc=0)[0])


def randomised_pit(n_obs, V_obs, q_lam, q_mag, a, n_part=2000, seed=0, phi=None):
    """Held-out calibration. Counts are discrete, so PIT must be randomised."""
    rng = np.random.default_rng(seed)
    T = len(n_obs)
    lam0 = max(n_obs.mean(), 0.05)
    mu0 = max(V_obs[V_obs > 0].mean() if (V_obs > 0).any() else 1.0, 1.0)
    S = np.array([np.log(lam0), np.log(mu0)]) + np.column_stack(
        [rng.normal(0, 0.5, n_part), rng.normal(0, 0.5, n_part)])
    pits = []
    for t in range(T):
        S = S + np.column_stack([rng.normal(0, np.sqrt(q_lam), n_part),
                                 rng.normal(0, np.sqrt(q_mag), n_part)])
        lam = np.exp(S[:, 0])
        if t >= 3:                                    # skip burn-in
            if phi is None or phi > 1e6:
                F = stats.poisson.cdf(n_obs[t], lam).mean()
                Fm = stats.poisson.cdf(n_obs[t] - 1, lam).mean()
            else:
                pr = phi / (phi + lam)
                F = stats.nbinom.cdf(n_obs[t], phi, pr).mean()
                Fm = stats.nbinom.cdf(n_obs[t] - 1, phi, pr).mean()
            pits.append(Fm + rng.uniform() * (F - Fm))
        lw = np.array([loglik_obs(n_obs[t], V_obs[t], lam[i],
                                  np.exp(S[i, 1]), a) for i in range(n_part)])
        w = np.exp(lw - lw.max())
        if w.sum() <= 0 or not np.isfinite(w.sum()):
            w = np.ones(n_part)
        w /= w.sum()
        S = S[rng.choice(n_part, n_part, p=w)]
    p = np.array(pits)
    return dict(n=len(p),
                ks=float(np.max(np.abs(np.sort(p) - np.linspace(0, 1, len(p))))),
                mean=float(p.mean()),
                cov50=float(((p > .25) & (p < .75)).mean()),
                cov90=float(((p > .05) & (p < .95)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="DNP")
    ap.add_argument("--n-climate", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-particles", type=int, default=3000)
    ap.add_argument("--out", default=str(REPO / "results" / "bayes_rate.csv"))
    args = ap.parse_args()

    rows = []
    for res in [s.strip() for s in args.res.split(",")]:
        cfg = reservoir_config(res)
        obs = []
        for cs in scenario_names()[:args.n_climate]:
            q, _, dates = load_inflow(cs, key=res)
            m = dates.year.values >= 2020
            yrs, n, V = annual_events(q[m], dates[m], cfg.r_max_cfs)
            obs.append((n, V, yrs, cs))
        train = [(n, V) for n, V, _, _ in obs[:args.n_train]]
        test = obs[args.n_train:]

        a = fit_gamma_shape(train)
        q_lam, q_mag, phi = fit_hyper(train, a, n_part=1500)
        nz = np.mean([np.mean(n == 0) for n, V in train])
        rate = np.mean([n.mean() for n, V in train])

        print(f"\n{'='*76}\n{res}\n{'='*76}")
        print(f"events/yr {rate:.2f}  ({100*nz:.0f}% zero years, "
              f"~{10*rate:.1f} per decade)")
        vm = np.var(np.concatenate([n for n, V in train])) / max(
            np.mean(np.concatenate([n for n, V in train])), 1e-9)
        print(f"Gamma shape a={a:.2f} | drift Q_lambda={q_lam:.4f} "
              f"Q_mu={q_mag:.4f} | NB dispersion phi={phi:.2f} "
              f"(count var/mean={vm:.2f}; phi->inf would mean Poisson)")

        cal = randomised_pit(*test[0][:2], q_lam, q_mag, a, n_part=2000, phi=phi)
        for n, V, _, _ in test[1:]:
            c = randomised_pit(n, V, q_lam, q_mag, a, n_part=2000, phi=phi)
            for k in ("ks", "mean", "cov50", "cov90"):
                cal[k] = (cal[k] + c[k]) / 2
        ok = cal["ks"] < 0.15 and 0.38 < cal["cov50"] < 0.62
        print(f"held-out randomised PIT: KS={cal['ks']:.3f} mean={cal['mean']:.3f} "
              f"50%={cal['cov50']:.2f} 90%={cal['cov90']:.2f}  -> "
              f"{'CALIBRATED' if ok else 'MISCALIBRATED'}")

        for n, V, yrs, cs in test:
            m_, sd_, _ = particle_filter(n, V, q_lam, q_mag, a,
                                         args.n_particles,
                                         np.random.default_rng(1), phi=phi)
            for t in range(len(n)):
                rows.append(dict(res=res, scen=cs, year=int(yrs[t]),
                                 n=int(n[t]), V=float(V[t]),
                                 log_ann_vol_mean=m_[t], log_ann_vol_sd=sd_[t]))
        sub = pd.DataFrame([r for r in rows if r["res"] == res])
        print(f"posterior sd on log expected annual volume: "
              f"mean {sub.log_ann_vol_sd.mean():.3f} "
              f"(Gaussian model gave 5.2 at NHG)")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
