#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRUE perfect-foresight optimum over RELEASES -- not restricted to the policy
atlas. This is the unrestricted ceiling: the best any controller could do with
perfect knowledge of the inflow and demand trajectory.

The existing `J_oracle` picks the best ATLAS POLICY SEQUENCE in hindsight, so it
is still confined to the 7-parameter hedging class. Subtracting this benchmark
splits the remaining gap in two:

    J_truelookup - J_oracle    representation regret (exposure space)
    J_oracle     - J_trueopt   POLICY-CLASS regret (the 7-param rule itself)

FORMULATION. The dynamics are linear and both stage costs are convex in release,
so the problem is a convex QP with a unique global optimum -- no DP grid, no
discretization error:

    min  sum_t  (d_t - R_t)_+^2  +  c (R_t - Rmax)_+
    s.t. S_t = S_{t-1} + Q_t - R_t          (mass balance, af)
         0 <= S_t <= K,  R_t >= 0           (physical only)
         S_{-1} = S0,  S_{T-1} = S0         (pinned terminal storage)

Costs mirror model._simulate_core exactly (shortage squared, flood linear at
`flood_penalty`). Only PHYSICAL constraints are imposed: dead pool and TOCS are
parameters of the heuristic policy, not physics, so they are dropped. Mass
balance is enforced honestly (R_t <= S_t + Q_t), which is consistent with the
policy simulations -- their zero-floor was verified never to bind.

Units are scaled internally (kcfs / taf) for solver conditioning; reported J is
in the same units as `simulate`.

USAGE
    python scripts/true_optimum.py --res ORO --years 10        # quick validate
    python scripts/true_optimum.py --res ORO                   # full horizon
"""
from __future__ import annotations
import sys, argparse, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bdps")); sys.path.insert(0, str(REPO / "bdps"))
from data_io import load_inflow, load_demand, reservoir_config          # noqa: E402
from model import simulate, CFS_TO_AFD                                  # noqa: E402

START_YEAR = 2020
OUT = REPO / "results"


def solve_true_optimum(q_cfs, dowy, dm, cfg, S0_af, eps=1e-7, max_iter=400_000,
                       verbose=False, pin_terminal=True, S_end_af=None):
    """Global optimum of the perfect-foresight release problem (convex QP).

    Returns dict with J, J_shortage, J_flood, release (cfs), storage (af).
    """
    import osqp
    T = int(q_cfs.size)
    c = float(CFS_TO_AFD)                      # taf per (kcfs*day) == af per (cfs*day)

    d = (cfg.R_avg[dowy] * dm) / 1000.0        # demand, kcfs
    Q = q_cfs / 1000.0                         # inflow, kcfs
    Rmax = float(cfg.r_max_cfs) / 1000.0       # kcfs
    K = float(cfg.capacity_af) / 1000.0        # taf
    S0 = float(S0_af) / 1000.0                 # taf
    pen = float(cfg.flood_penalty)

    # objective in ORIGINAL units = 1e6*sum(s_kcfs^2) + pen*1e3*sum(f_kcfs)
    # scale by 1e6 -> sum(s^2) + (pen*1e-3)*sum(f);  J_orig = 1e6 * obj_scaled
    wf = pen * 1e-3
    SCALE = 1e6

    # x = [R (T) | S (T) | s (T) | f (T)]
    n = 4 * T
    iR, iS, iSh, iF = 0, T, 2 * T, 3 * T

    # --- P: quadratic only on shortage block; (1/2)x'Px = sum s^2  -> P_ss = 2I
    Pdiag = np.concatenate([np.zeros(2 * T), 2.0 * np.ones(T), np.zeros(T)])
    P = sp.diags(Pdiag).tocsc()
    qv = np.zeros(n); qv[iF:iF + T] = wf

    rows = []
    # --- mass balance: S_t - S_{t-1} + c*R_t = c*Q_t   (t=0 uses S0)
    r_idx = np.arange(T)
    A_dyn = sp.coo_matrix(
        (np.concatenate([c * np.ones(T), np.ones(T), -np.ones(T - 1)]),
         (np.concatenate([r_idx, r_idx, r_idx[1:]]),
          np.concatenate([iR + r_idx, iS + r_idx, iS + r_idx[:-1]]))),
        shape=(T, n))
    b_dyn = c * Q.copy(); b_dyn[0] += S0
    rows.append((A_dyn, b_dyn, b_dyn))

    # --- shortage hinge: R_t + s_t >= d_t
    A_sh = sp.hstack([sp.eye(T), sp.csr_matrix((T, T)), sp.eye(T),
                      sp.csr_matrix((T, T))])
    rows.append((A_sh, d, np.full(T, np.inf)))

    # --- flood hinge: -R_t + f_t >= -Rmax
    A_fl = sp.hstack([-sp.eye(T), sp.csr_matrix((T, T)), sp.csr_matrix((T, T)),
                      sp.eye(T)])
    rows.append((A_fl, np.full(T, -Rmax), np.full(T, np.inf)))

    # --- box bounds (identity), with S_{T-1} pinned to S0
    lb = np.concatenate([np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)])
    ub = np.concatenate([np.full(T, np.inf), np.full(T, K),
                         np.full(T, np.inf), np.full(T, np.inf)])
    if pin_terminal:                                   # pinned terminal storage
        Sfin = S0 if S_end_af is None else float(S_end_af) / 1000.0
        lb[iS + T - 1] = Sfin; ub[iS + T - 1] = Sfin
    rows.append((sp.eye(n), lb, ub))

    A = sp.vstack([r[0] for r in rows]).tocsc()
    l = np.concatenate([r[1] for r in rows])
    u = np.concatenate([r[2] for r in rows])

    prob = osqp.OSQP()
    prob.setup(P=P, q=qv, A=A, l=l, u=u, eps_abs=eps, eps_rel=eps,
               max_iter=max_iter, polish=True, verbose=verbose)
    r = prob.solve()
    status = str(r.info.status)
    x = r.x
    if x is None or not np.all(np.isfinite(x)):
        raise RuntimeError(f"QP failed: {status}")

    R = np.maximum(x[iR:iR + T], 0.0) * 1000.0            # cfs
    S = np.clip(x[iS:iS + T], 0.0, K) * 1000.0            # af
    # score the recovered release with the SIMULATOR's own cost accounting
    d_cfs = cfg.R_avg[dowy] * dm
    sh = np.maximum(d_cfs - R, 0.0) ** 2
    fl = pen * np.maximum(R - cfg.r_max_cfs, 0.0)
    return dict(J=float(sh.sum() + fl.sum()), J_shortage=float(sh.sum()),
                J_flood=float(fl.sum()), release=R, storage=S,
                status=status, obj_qp=float(r.info.obj_val) * SCALE,
                iters=int(r.info.iter))


def load_traj(res, cs, ls, years=None):
    cfg = reservoir_config(res)
    q, dowy, dts = load_inflow(cs, key=res)
    dm, _ = load_demand(ls)
    n = min(q.size, dm.size)
    q, dowy, dts, dm = q[:n], dowy[:n], dts[:n], dm[:n]
    m = dts.year.values >= START_YEAR
    q, dowy, dts, dm = q[m], dowy[m], dts[m], dm[m]
    if years:
        keep = dts.year.values < START_YEAR + years
        q, dowy, dts, dm = q[keep], dowy[keep], dts[keep], dm[keep]
    return cfg, q, dowy, dm, dts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="ORO")
    ap.add_argument("--clim", default="cnrm-cm5_rcp85_r1i1p1")
    ap.add_argument("--lulc", default="GCAM-26")
    ap.add_argument("--years", type=int, default=0, help="0 = full horizon")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    cfg, q, dowy, dm, dts = load_traj(a.res, a.clim, a.lulc, a.years or None)
    S0 = float(cfg.S_avg[-1])
    print(f"{a.res}  T={q.size} days  ({dts.year.values[0]}-{dts.year.values[-1]})  "
          f"S0={S0:,.0f} af  K={cfg.capacity_af:,.0f} af")

    t0 = time.time()
    o = solve_true_optimum(q, dowy, dm, cfg, S0, verbose=a.verbose)
    dt = time.time() - t0
    print(f"  status={o['status']}  iters={o['iters']}  {dt:.1f}s")
    print(f"  J_trueopt = {o['J']:.6g}   (flood {o['J_flood']:.6g} + "
          f"shortage {o['J_shortage']:.6g})")
    print(f"  QP objective (rescaled) = {o['obj_qp']:.6g}   "
          f"[gap vs re-scored J: {abs(o['obj_qp']-o['J'])/max(o['J'],1):.2e}]")
    print(f"  terminal storage = {o['storage'][-1]:,.0f} af  (S0 = {S0:,.0f})")

    # reference: best single atlas policy on this trajectory (hindsight)
    z = np.load(OUT / "deploy_cache" / f"{a.res}.npz")
    sims = [simulate(q, x, cfg, dowy, DM=dm, S0=S0) for x in z["X"]]
    Js = np.array([s["J"] for s in sims])
    ib = int(np.argmin(Js)); best = float(Js[ib])
    S_end_pol = float(sims[ib]["storage"][-1])
    print(f"  best FIXED atlas policy (hindsight) = {best:.6g}"
          f"   (its terminal storage = {S_end_pol:,.0f} af)")
    print(f"  -> pinned-terminal optimum is {100*(1-o['J']/best):+.1f}% vs it"
          f"   {'OK' if o['J'] <= best * 1.0001 else '<-- end-effect, not comparable'}")

    # free terminal = unconstrained lower bound; and matched terminal = fair test
    of = solve_true_optimum(q, dowy, dm, cfg, S0, pin_terminal=False)
    om = solve_true_optimum(q, dowy, dm, cfg, S0, S_end_af=S_end_pol)
    print(f"  FREE-terminal optimum    = {of['J']:.6g}  "
          f"(ends {of['storage'][-1]:,.0f} af)  -> {100*(1-of['J']/best):+.1f}% vs policy")
    print(f"  MATCHED-terminal optimum = {om['J']:.6g}  "
          f"-> {100*(1-om['J']/best):+.1f}% vs policy"
          f"   {'OK' if om['J'] <= best * 1.0001 else '*** VIOLATION ***'}")
    print(f"  cost of pinning S_end=S0 = {100*(o['J']/of['J']-1):.1f}% of J")


if __name__ == "__main__":
    main()
