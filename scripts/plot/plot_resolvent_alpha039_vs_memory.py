# plot_resolvent_alpha039_vs_memory.py
#
# Discounted resolvent of the trained weight matrix at a FIXED discount factor
# alpha = 0.39 (the largest value at which ALL agents' series converge,
# since max spectral radius ~2.52 -> 1/2.52 ~ 0.40).
#   x = ‖(I - alpha A)^-1‖_F ,  A = W_trained o M ,  y = memory.
#
# Usage: python3 scripts/plot/plot_resolvent_alpha039_vs_memory.py

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

VER = "v7_20260622_alpha"
COND = "cs0_n010"
ALPHA = 0.39   # discount factor (safe: all agents converge)
GROUPS = ["a0", "a05", "a10", "a15"]
COLORS = {"a0": "steelblue", "a05": "mediumseagreen", "a10": "goldenrod", "a15": "crimson"}
ALPHA_LABEL = {"a0": "0", "a05": "0.5", "a10": "1.0", "a15": "1.5"}

rows = []
for s in GROUPS:
    for f in sorted(glob.glob(f"Results/results_degseq_{VER}_{s}_{COND}/*.pkl")):
        r = pickle.load(open(f, "rb"))
        A = np.asarray(r["W_trained"], float) * np.asarray(r["M_hh"], float)
        n = A.shape[0]
        res = np.linalg.norm(np.linalg.inv(np.eye(n) - ALPHA * A), "fro")
        rows.append({"struct": s, "mem": float(r["memory_sum"]), "res": res})
print(f"loaded {len(rows)} agents ({COND}), alpha={ALPHA}")
mem = np.array([r["mem"] for r in rows]); res = np.array([r["res"] for r in rows])
print(f"  rho(memory, ‖(I-αA)⁻¹‖) = {spearmanr(mem, res).correlation:+.3f}")


def fitline(ax, x, y, color, lw):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size >= 2 and np.ptp(x) > 0:
        m, b = np.polyfit(x, y, 1)
        xf = np.linspace(x.min(), x.max(), 50)
        ax.plot(xf, m * xf + b, color=color, lw=lw, alpha=0.9)


fig, ax = plt.subplots(figsize=(6.5, 5))
fig.suptitle(f"Memory vs discounted resolvent norm (α = {ALPHA})", fontsize=13)
for s in GROUPS:
    xs = [r["res"] for r in rows if r["struct"] == s]
    ys = [r["mem"] for r in rows if r["struct"] == s]
    ax.scatter(xs, ys, s=10, color=COLORS[s], alpha=0.18, edgecolors="none", label=ALPHA_LABEL[s])
    fitline(ax, xs, ys, COLORS[s], lw=2.0)
ax.set_xlabel(f"‖(I − αA)⁻¹‖_F,  trained weight matrix  (α = {ALPHA})", fontsize=12)
ax.set_ylabel("Memory", fontsize=13)
ax.tick_params(labelsize=11)
ax.legend(fontsize=10, title="Mask heterogeneity", title_fontsize=10, loc="upper left", framealpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs("Results/plots/misc", exist_ok=True)
out = "Results/plots/misc/plot_resolvent_alpha039_vs_memory_n100_cs0_n010.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.show()
