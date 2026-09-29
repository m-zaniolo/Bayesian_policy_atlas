#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effect of the climate-exposure atlas: how much performance is lost purely
because the chosen exposure space does not fully capture the aspects of climate
that drive optimal operation.  This is the True-lookup -> Oracle gap alone
(no belief / no estimation term):

    representation regret  =  (J_truelookup - J_oracle) / J_frozen2020  x 100%

  * J_truelookup = the atlas policy at the TRUE exposure coordinate each year
    (perfect belief); its only handicap vs the oracle is the exposure space.
  * J_oracle     = perfect-foresight reoptimisation (best attainable).

One box per reservoir over the 240 trajectories (30 GCM x 8 demand). Reservoirs
ordered by the cost-weighted mean (diamond). Reducible by enriching the exposure
space / policy atlas: drought persistence, drought frequency, snowmelt-timing
shifts, etc.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results"

K = ["res", "clim", "lulc"]
tl = pd.read_csv(OUT / "truelookup_only.csv")[K + ["J_truelookup"]]
b = pd.read_csv(OUT / "bench_baselines.csv")[K + ["J_oracle", "J_frozen2020"]]
m = tl.merge(b, on=K)
m["reg"] = 100.0 * (m["J_truelookup"] - m["J_oracle"]) / m["J_frozen2020"]

RES = ["NML", "EXC", "NHG", "BUL", "FOL", "SHA", "MIL", "ORO", "DNP"]
data = {res: m.loc[m.res == res, "reg"].values for res in RES}
# cost-weighted mean per reservoir (the headline number)
cw = {res: 100.0 * (g.J_truelookup.sum() - g.J_oracle.sum()) / g.J_frozen2020.sum()
      for res, g in m.groupby("res")}
order = sorted(RES, key=lambda r: cw[r], reverse=True)

BOX, DIA = "#2980b9", "#c0392b"          # blue box, red cost-weighted-mean diamond
plt.rcParams.update({"font.size": 14, "axes.edgecolor": "#444", "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(10.8, 6.2))
pos = np.arange(len(order))

bp = ax.boxplot([data[r] for r in order], positions=pos, widths=0.62,
                showfliers=False, patch_artist=True, medianprops=dict(color="#0d2b45", lw=1.8),
                whiskerprops=dict(color="#555", lw=1.1), capprops=dict(color="#555", lw=1.1),
                boxprops=dict(facecolor=BOX, edgecolor="#1b5a86", alpha=0.75, lw=1.0))

# cost-weighted mean marker (value shown in legend, not as clutter by the diamond)
cwm = [cw[r] for r in order]
ax.scatter(pos, cwm, marker="D", s=60, color=DIA, edgecolor="#7d1a12", lw=0.8,
           zorder=6, label="cost-weighted mean")

ax.set_xticks(pos); ax.set_xticklabels(order, fontsize=16, fontweight="bold")
ax.tick_params(axis="y", labelsize=14)
ax.set_ylabel("representation regret\n(% of frozen-2020)", fontsize=15)
ax.set_ylim(bottom=-3)
ax.grid(axis="y", color="#ddd", lw=0.6, zorder=0); ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=14, frameon=False)
fig.tight_layout()
for d in (OUT, OUT / "all_reservoir_figures"):
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "atlas_representation.png", dpi=170, bbox_inches="tight")
print("wrote", OUT / "atlas_representation.png")

print("\nrepresentation regret (% of frozen)   median   cost-wt mean")
for r in order:
    print(f"  {r}: {np.median(data[r]):7.1f}   {cw[r]:7.1f}")
