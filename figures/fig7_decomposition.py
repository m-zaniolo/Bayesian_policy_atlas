#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why do the action-selection methods tie on the average? Decompose each method's
realised cost difference vs the MEAN-EXPOSURE plug-in into two offsetting parts,
by YEAR TYPE:
    premium = extra cost paid in NORMAL (non-flood) years   (over-hedging)   [+]
    payoff  = cost SAVED in FLOOD years                       (hedge cashes in) [-]
    net     = premium + payoff  (the small number the barplot shows)

A "flood year" = a trajectory-year in which the plug-in incurs flood cost (>0).
Costs normalised to each reservoir's total plug-in cost (%). Per-reservoir 3x3
and a pooled aggregate panel. Reads outputs/cost_over_time.csv.gz.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent; OUT = REPO / "results"
RES = ["NML", "EXC", "NHG", "BUL", "FOL", "SHA", "MIL", "ORO", "DNP"]
DISP = {"bayes": "Posterior-\noptimal", "blend": "Posterior-\naveraged"}
C_PREM, C_PAY, C_NET = "#c0392b", "#2980b9", "#111"

d = pd.read_csv(OUT / "cost_over_time.csv.gz"); d["J"] = d.J_flood + d.J_shortage
key = ["res", "clim", "lulc", "year"]
piv = d.pivot_table(index=key, columns="method", values="J")
flood = d[d.method == "plugin"].set_index(key).J_flood > 0     # flood-year indicator
piv = piv.join(flood.rename("flood")).reset_index()


def decomp(g):
    base = g.plugin.sum(); out = {}
    for m in ("bayes", "blend"):
        diff = g[m] - g.plugin
        out[m] = (100 * diff[~g.flood].sum() / base,      # premium (normal yrs, +)
                  100 * diff[g.flood].sum() / base,        # payoff (flood yrs, -)
                  100 * diff.sum() / base)                 # net
    out["fl_frac"] = 100 * g.flood.mean()
    return out


def panel(ax, dec, title, ymax=None):
    x = np.arange(2); meth = ["bayes", "blend"]
    prem = [dec[m][0] for m in meth]; pay = [dec[m][1] for m in meth]
    net = [dec[m][2] for m in meth]
    ax.bar(x, prem, 0.55, color=C_PREM, edgecolor="#333", lw=0.6, zorder=3,
           label="normal-year premium")
    ax.bar(x, pay, 0.55, color=C_PAY, edgecolor="#333", lw=0.6, zorder=3,
           label="flood-year payoff")
    ax.scatter(x, net, marker="D", s=70, color=C_NET, zorder=6, label="net")
    for xi, nv in zip(x, net):
        ax.plot([xi - 0.3, xi + 0.3], [nv, nv], color=C_NET, lw=1.6, zorder=5)
    ax.axhline(0, color="#333", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([DISP[m] for m in meth], fontsize=8.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    if ymax:
        ax.set_ylim(-ymax, ymax)
    ax.text(0.03, 0.96, f"flood yrs: {dec['fl_frac']:.0f}%", transform=ax.transAxes,
            va="top", fontsize=7.5, color="#555")


# ---- 3x3 per reservoir ----
decs = {r: decomp(g) for r, g in piv.groupby("res")}
ymax = 1.15 * max(abs(v) for r in RES for m in ("bayes", "blend") for v in decs[r][m][:2])
fig, axes = plt.subplots(3, 3, figsize=(12, 9.5), sharey=True)
for ax, r in zip(axes.flat, RES):
    panel(ax, decs[r], r, ymax=ymax)
for a in axes[:, 0]:
    a.set_ylabel("cost vs mean-exposure\n(% of plug-in cost)", fontsize=9)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, 1.005))
fig.suptitle("Why the methods tie on average: a normal-year PREMIUM (red, +) offset "
             "by a flood-year PAYOFF (blue, −) → small NET (◆)", y=1.035,
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(OUT / "yeartype_decomp.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "yeartype_decomp.pdf", bbox_inches="tight")
print("wrote yeartype_decomp.png/.pdf")

# ---- aggregate (pooled over all reservoirs) ----
figA, axA = plt.subplots(figsize=(6.4, 5.2))
panel(axA, decomp(piv), "All reservoirs (pooled)")
axA.set_ylabel("cost vs mean-exposure  (% of plug-in cost)", fontsize=10)
axA.legend(frameon=False, fontsize=9, loc="upper right")
figA.tight_layout()
figA.savefig(OUT / "yeartype_decomp_agg.png", dpi=160, bbox_inches="tight")
print("wrote yeartype_decomp_agg.png")

# mean posterior SD (belief width) per reservoir -- mean of post_sd over all
# deployed trajectory-years (outputs/bayes_deploy.csv); BUL & MIL added here.
POST_SD = {"ORO": 1.05, "FOL": 1.02, "SHA": 0.62, "BUL": 0.60, "EXC": 0.60,
           "MIL": 0.51, "NML": 0.51, "DNP": 0.41, "NHG": 0.37}
from matplotlib.patches import Patch                                    # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402


# per-method plot style: (bar offset factor, hatch, marker, fill, alpha, label)
MSTY = {"bayes": (None, "D", C_NET, 1.0, "posterior-optimal"),
        "blend": (None, "o", "white", 1.0, "posterior-averaged")}


def draw_bars(axS, methods=("bayes", "blend"), legend=True):
    key = methods[0] if len(methods) == 1 else "bayes"
    order = sorted(RES, key=lambda r: decs[r][key][2])
    xs = np.arange(len(order))
    n = len(methods); BW = 0.5 / n if n > 1 else 0.55
    offs = ([-0.20, 0.20] if n == 2 else [0.0])
    for m, off in zip(methods, offs):
        hatch = "///" if (n == 2 and m == "blend") else None
        alpha = 0.55 if (n == 2 and m == "blend") else 1.0
        mk, fill = MSTY[m][1], MSTY[m][2]
        prem = np.array([decs[r][m][0] for r in order])
        pay = np.array([decs[r][m][1] for r in order])
        net = np.array([decs[r][m][2] for r in order])
        axS.bar(xs + off, prem, BW, color=C_PREM, alpha=alpha, hatch=hatch,
                edgecolor="#333", lw=0.6, zorder=3)
        axS.bar(xs + off, pay, BW, color=C_PAY, alpha=alpha, hatch=hatch,
                edgecolor="#333", lw=0.6, zorder=3)
        for xi, nv in zip(xs + off, net):
            axS.plot([xi - BW / 2, xi + BW / 2], [nv, nv], color=C_NET, lw=1.8, zorder=5)
        axS.scatter(xs + off, net, marker=mk, s=75, color=C_NET,
                    facecolor=fill, edgecolor=C_NET, lw=1.4, zorder=6)
    axS.axhline(0, color="#333", lw=1.0)
    axS.set_xticks(xs); axS.set_xticklabels(order, fontweight="bold", fontsize=14)
    axS.set_ylabel("relative cost difference\nfrom mean-exposure (%)", fontsize=15)
    axS.tick_params(axis="y", labelsize=13)
    axS.grid(axis="y", alpha=0.3); axS.set_axisbelow(True)
    if legend:
        handles = [
            Patch(fc=C_PREM, ec="#333", label="normal-year premium (over-hedging cost)"),
            Patch(fc=C_PAY, ec="#333", label="flood-year payoff (cost saved in floods)"),
        ]
        if n == 2:
            handles += [
                Patch(fc="0.55", ec="#333", label="left bar = posterior-optimal"),
                Patch(fc="0.8", ec="#333", hatch="///", label="right bar = posterior-averaged"),
                Line2D([0], [0], marker="D", color=C_NET, lw=0, label="net (posterior-optimal)"),
                Line2D([0], [0], marker="o", color=C_NET, mfc="white", lw=0,
                       label="net (posterior-averaged)"),
            ]
        else:
            m = methods[0]
            handles.append(Line2D([0], [0], marker=MSTY[m][1], color=C_NET, lw=0,
                                  mfc=MSTY[m][2], label=f"net ({MSTY[m][4]})"))
        axS.legend(handles=handles, loc="lower right", frameon=True, fontsize=12, ncol=1)


def draw_width_scatter(axSc, methods=("bayes", "blend"), legend=True):
    """Net cost advantage vs mean-exposure as a function of belief width. Points
    above 0 = mean-exposure plug-in performs BETTER; wider posterior -> the
    Bayesian version over-hedges and the plug-in wins. Trend fit on methods[0]."""
    rr = [r for r in RES if r in POST_SD]
    sd = np.array([POST_SD[r] for r in rr])
    axSc.axhspan(0, 40, color=C_PREM, alpha=0.06, zorder=0)
    axSc.axhspan(-40, 0, color=C_PAY, alpha=0.06, zorder=0)
    axSc.axhline(0, color="#333", lw=1.0, zorder=1)
    key = methods[0]
    nk = np.array([decs[r][key][2] for r in rr])
    b1, b0 = np.polyfit(sd, nk, 1)                       # OLS trend on lead method
    xr = np.array([sd.min() - 0.05, sd.max() + 0.05])
    axSc.plot(xr, b0 + b1 * xr, color=C_NET, lw=1.4, ls="--", zorder=2)
    for m in methods:                                    # secondary methods behind
        if m == key:
            continue
        nm = np.array([decs[r][m][2] for r in rr])
        axSc.scatter(sd, nm, marker=MSTY[m][1], s=70, facecolor=MSTY[m][2],
                     edgecolor=C_NET, lw=1.4, zorder=4)
    nlead = nk
    axSc.scatter(sd, nlead, marker=MSTY[key][1], s=85, facecolor=MSTY[key][2],
                 edgecolor=C_NET, lw=1.4, zorder=5)
    for r, x, y in zip(rr, sd, nlead):
        axSc.annotate(r, (x, y), textcoords="offset points", xytext=(-8, 6),
                      ha="right", va="bottom", fontsize=11, fontweight="bold")
    axSc.set_xlabel("mean posterior SD  (belief width)", fontsize=15)
    axSc.set_ylabel("relative cost difference\nfrom mean-exposure (%)", fontsize=15)
    axSc.set_ylim(-18, 9)
    axSc.tick_params(labelsize=13)
    axSc.grid(alpha=0.25, zorder=0); axSc.set_axisbelow(True)
    axSc.text(0.03, 0.97, "mean-exposure better", transform=axSc.transAxes,
              va="top", ha="left", fontsize=13, color=C_PREM, fontweight="bold")
    axSc.text(0.03, 0.03, "Bayesian better", transform=axSc.transAxes,
              va="bottom", ha="left", fontsize=13, color=C_PAY, fontweight="bold")
    if legend:
        r_val = np.corrcoef(sd, nk)[0, 1]
        hh = [Line2D([0], [0], marker=MSTY[m][1], color=C_NET, lw=0, mfc=MSTY[m][2],
                     label=MSTY[m][4]) for m in methods]
        hh.append(Line2D([0], [0], color=C_NET, lw=1.4, ls="--", label=f"trend (r={r_val:.2f})"))
        axSc.legend(handles=hh, loc="lower right", frameon=True, fontsize=12)


# ---- atlas representation-regret data (for the bottom panel) ----
try:
    _K = ["res", "clim", "lulc"]
    _tl = pd.read_csv(OUT / "truelookup_only.csv")[_K + ["J_truelookup"]]
    _bb = pd.read_csv(OUT / "bench_baselines.csv")[_K + ["J_oracle", "J_frozen2020"]]
    _ma = _tl.merge(_bb, on=_K)
    _ma["reg"] = 100.0 * (_ma.J_truelookup - _ma.J_oracle) / _ma.J_frozen2020
    ATLAS_DATA = {r: _ma.loc[_ma.res == r, "reg"].values for r in RES}
    _acw = {r: 100.0 * (g.J_truelookup.sum() - g.J_oracle.sum()) / g.J_frozen2020.sum()
            for r, g in _ma.groupby("res")}
    ATLAS_ORDER = sorted(RES, key=lambda r: _acw[r], reverse=True)
    HAVE_ATLAS = True
except Exception as e:                                                   # noqa: BLE001
    print("atlas data unavailable, skipping bottom panel:", e); HAVE_ATLAS = False

# Same regret, but measured against the FREE-PARAMETER oracle (same 7-param
# shape, parameters continuous instead of restricted to the 112 atlas cells).
# The gap is then the full cost of the atlas, not just its exposure-space axes.
try:
    _fp = pd.read_csv(OUT / "freeparam_oracle.csv")[_K + ["J_oracle_free"]]
    _mf = _tl.merge(_bb, on=_K).merge(_fp, on=_K)
    _mf["reg"] = 100.0 * (_mf.J_truelookup - _mf.J_oracle_free) / _mf.J_frozen2020
    ATLAS_DATA_FREE = {r: _mf.loc[_mf.res == r, "reg"].values for r in RES}
    _acwf = {r: 100.0 * (g.J_truelookup.sum() - g.J_oracle_free.sum()) / g.J_frozen2020.sum()
             for r, g in _mf.groupby("res")}
    # order by MEDIAN per-trajectory regret: the boxes show distributions, and the
    # median separates DNP from the rest more sharply than the cost-weighted mean
    # (DNP/next = 2.38 vs 2.02).
    _amed = {r: float(np.median(ATLAS_DATA_FREE[r])) for r in RES}
    ATLAS_ORDER_FREE = sorted(RES, key=lambda r: _amed[r], reverse=True)
    HAVE_FREE = True
except Exception as e:                                                   # noqa: BLE001
    print("free-param oracle unavailable:", e); HAVE_FREE = False


def draw_atlas_box(ax, data=None, order=None):
    """Representation regret per reservoir (atlas-at-true-exposure vs oracle),
    one box over the 240 trajectories. No cost-weighted-mean diamond."""
    data = ATLAS_DATA if data is None else data
    order = ATLAS_ORDER if order is None else order
    pos = np.arange(len(order))
    ax.boxplot([data[r] for r in order], positions=pos, widths=0.62,
               showfliers=False, patch_artist=True,
               medianprops=dict(color="#0d2b45", lw=1.8),
               whiskerprops=dict(color="#555", lw=1.1),
               capprops=dict(color="#555", lw=1.1),
               boxprops=dict(facecolor=C_PAY, edgecolor="#1b5a86", alpha=0.75, lw=1.0))
    ax.set_xticks(pos); ax.set_xticklabels(order, fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("representation regret\n(% of frozen-2020)", fontsize=14)
    ax.set_ylim(bottom=-3)
    ax.grid(axis="y", color="#ddd", lw=0.6, zorder=0); ax.set_axisbelow(True)


# ---- (a) SINGLE panel decomposition, all reservoirs, both methods ----
figS, axS = plt.subplots(figsize=(13.5, 6.4))
draw_bars(axS)
axS.set_title("Why the action-selection methods tie on the average: a steady premium "
              "offset by rare flood-year payoffs", fontsize=12.5, fontweight="bold")
figS.tight_layout()
figS.savefig(OUT / "yeartype_decomp_single.png", dpi=160, bbox_inches="tight")
figS.savefig(OUT / "yeartype_decomp_single.pdf", bbox_inches="tight")
print("wrote yeartype_decomp_single.png/.pdf")

# ---- (b) standalone belief-width scatter ----
figW, axW = plt.subplots(figsize=(6.0, 5.4))
draw_width_scatter(axW)
figW.tight_layout()
figW.savefig(OUT / "width_vs_advantage.png", dpi=160, bbox_inches="tight")
figW.savefig(OUT / "width_vs_advantage.pdf", bbox_inches="tight")
print("wrote width_vs_advantage.png/.pdf")

# ---- (c) combined two-panel figure (decomp | width scatter) ----
figC, (axC0, axC1) = plt.subplots(1, 2, figsize=(18.5, 6.2),
                                  gridspec_kw={"width_ratios": [2.3, 1.0]})
draw_bars(axC0); draw_width_scatter(axC1)
axC0.text(-0.06, 1.02, "a", transform=axC0.transAxes, fontsize=15, fontweight="bold")
axC1.text(-0.14, 1.02, "b", transform=axC1.transAxes, fontsize=15, fontweight="bold")
figC.tight_layout()
figC.savefig(OUT / "yeartype_decomp_with_width.png", dpi=160, bbox_inches="tight")
figC.savefig(OUT / "yeartype_decomp_with_width.pdf", bbox_inches="tight")
print("wrote yeartype_decomp_with_width.png/.pdf")

# ---- (d) posterior-averaged ONLY, scatter first then decomposition ----
figB, (axB0, axB1) = plt.subplots(1, 2, figsize=(18.5, 6.2),
                                  gridspec_kw={"width_ratios": [1.0, 2.3]})
draw_width_scatter(axB0, methods=("blend",))
draw_bars(axB1, methods=("blend",))
axB0.text(-0.14, 1.02, "a", transform=axB0.transAxes, fontsize=17, fontweight="bold")
axB1.text(-0.05, 1.02, "b", transform=axB1.transAxes, fontsize=17, fontweight="bold")
figB.tight_layout()
figB.savefig(OUT / "blend_width_decomp.png", dpi=160, bbox_inches="tight")
figB.savefig(OUT / "blend_width_decomp.pdf", bbox_inches="tight")
print("wrote blend_width_decomp.png/.pdf")

# ---- (e) single 3-panel figure: scatter (a) + shrunk decomp (b) over atlas (c) ----
if HAVE_ATLAS:
    figV = plt.figure(figsize=(16, 10.5))
    gsV = figV.add_gridspec(2, 2, width_ratios=[1.0, 1.4], height_ratios=[1.0, 0.92],
                            hspace=0.30, wspace=0.20)
    axVa = figV.add_subplot(gsV[0, 0])
    axVb = figV.add_subplot(gsV[0, 1])
    axVc = figV.add_subplot(gsV[1, :])
    draw_width_scatter(axVa, methods=("blend",))
    draw_bars(axVb, methods=("blend",))
    draw_atlas_box(axVc)
    for ax_, lab, dx in ((axVa, "a", -0.16), (axVb, "b", -0.06), (axVc, "c", -0.065)):
        ax_.text(dx, 1.03, lab, transform=ax_.transAxes, fontsize=18, fontweight="bold")
    figV.savefig(OUT / "action_value_3panel.png", dpi=160, bbox_inches="tight")
    figV.savefig(OUT / "action_value_3panel.pdf", bbox_inches="tight")
    print("wrote action_value_3panel.png/.pdf")

# ---- (f) same 3-panel, panel c measured vs the FREE-PARAMETER oracle ----
if HAVE_ATLAS and HAVE_FREE:
    figF = plt.figure(figsize=(16, 10.5))
    gsF = figF.add_gridspec(2, 2, width_ratios=[1.0, 1.4], height_ratios=[1.0, 0.92],
                            hspace=0.30, wspace=0.20)
    aFa = figF.add_subplot(gsF[0, 0])
    aFb = figF.add_subplot(gsF[0, 1])
    aFc = figF.add_subplot(gsF[1, :])
    draw_width_scatter(aFa, methods=("blend",))
    draw_bars(aFb, methods=("blend",))
    draw_atlas_box(aFc, data=ATLAS_DATA_FREE, order=ATLAS_ORDER_FREE)
    for ax_, lab, dx in ((aFa, "a", -0.16), (aFb, "b", -0.06), (aFc, "c", -0.065)):
        ax_.text(dx, 1.03, lab, transform=ax_.transAxes, fontsize=18, fontweight="bold")
    figF.savefig(OUT / "action_value_3panel_freeopt.png", dpi=160, bbox_inches="tight")
    figF.savefig(OUT / "action_value_3panel_freeopt.pdf", bbox_inches="tight")
    print("wrote action_value_3panel_freeopt.png/.pdf")
    print("\npanel c, regret vs free-param oracle (% of frozen-2020), median order:")
    for r in ATLAS_ORDER_FREE:
        print(f"  {r}: median {_amed[r]:6.1f}  cost-wtd {_acwf[r]:6.1f}   "
              f"(cost-wtd was {_acw[r]:6.1f} vs atlas oracle)")

print("\nper reservoir (% of plug-in cost):   premium / payoff / net  [bayes]")
for r in RES:
    b = decs[r]["bayes"]
    print(f"  {r}: {b[0]:+6.1f} / {b[1]:+6.1f} / {b[2]:+5.1f}   (flood {decs[r]['fl_frac']:.0f}%)")
