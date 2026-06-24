# plot_eig_distribution_v6.py
#
# Advisor's eigenvalue-DISTRIBUTION plots, from the v6 per-state |lambda| data
# ("eff_lambdas_per_state", shape (n_states, n) per agent).
#   (A) 2D density image: x = memory_sum (or fitness) VALUE, y = |lambda|,
#       colour = P(|lambda| | x)  (each x-column normalised to a probability).
#   (B) mean / std / max of pooled |lambda| vs memory_sum and fitness (VALUES).
# Hypothesis (advisor): SPREAD of |lambda| (std) predicts memory better than max.
#
# Usage: python3 scripts/plot/plot_eig_distribution_v6.py --cond cs0_n010

import os
import glob
import pickle
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--cond", choices=["cs0_n010", "cs01_n005"], default="cs0_n010")
parser.add_argument("--xbins", type=int, default=35)
parser.add_argument("--min_lam", type=float, default=0.05,
                    help="lower |λ| bin edge; 0.0 INCLUDES the ~0 instant-decay mass (will dominate)")
parser.add_argument("--logcolor", action="store_true",
                    help="log colour scale (use with --min_lam 0 so the ~0 bin doesn't wash out the rest)")
parser.add_argument("--global_norm", action="store_true",
                    help="normalise the WHOLE image to sum 1 (joint P(|λ|, perf)); default is per-column P(|λ|·)")
parser.add_argument("--dataset", choices=["n10", "n100"], default="n10",
                    help="n10 = v6 named structures; n100 = v7 alpha sweep")
args = parser.parse_args()

if args.dataset == "n100":
    VER = "v7_20260622_alpha"
    GROUPS = ["a0", "a05", "a10", "a15"]
    COLORS = {"a0": "steelblue", "a05": "mediumseagreen", "a10": "goldenrod", "a15": "crimson"}
else:
    VER = "v6_20260622_degseq"
    GROUPS = ["regular", "d4", "medium", "high", "d8"]
    COLORS = {"regular": "steelblue", "d4": "mediumseagreen", "medium": "darkorange",
              "high": "tomato", "d8": "purple"}
# y-axis (|λ|) blocks: 80 edges from min_lam to 1.6 -> 79 blocks, each ~ (1.6-min_lam)/79 ≈ 0.02 tall.
# (min_lam is the LOWER EDGE, not the block height.) --min_lam 0.05 (default) drops the ~0
# instant-decay mass; --min_lam 0 includes it (then it dominates).
YBINS = np.linspace(args.min_lam, 1.6, 80)
YCENT = 0.5 * (YBINS[:-1] + YBINS[1:])      # block centres (unused; handy if plotting curves)

rows = []
for s in GROUPS:
    for f in sorted(glob.glob(f"Results/results_degseq_{VER}_{s}_{args.cond}/*.pkl")):
        r = pickle.load(open(f, "rb"))
        a = np.asarray(r["eff_lambdas_per_state"], float).flatten()  # ALL of this agent's |λ| (n_states*n)
        if a.size == 0:
            continue
        rows.append({"struct": s, "mem": float(r["memory_sum"]), "fit": float(r["fitness"]),
                     "mean": a.mean(), "std": a.std(), "max": a.max(),   # scalars -> summary scatter (Fig B)
                     # count of this agent's |λ| in each y-block ([0]=counts) -> density image (Fig A)
                     "hist": np.histogram(a, bins=YBINS)[0].astype(float)})
print(f"loaded {len(rows)} agents ({args.cond})")
present = sorted({r['struct'] for r in rows})
print("structures present:", present)


def density(xkey):
    # Build the 2D density image. Idea: cut the x-axis (memory_sum or fitness) into xbins
    # vertical blocks and the y-axis (|λ|) into the YBINS blocks; pixel (y,x) = probability
    # of finding an eigenvalue in that (|λ|-block, perf-block) cell.
    xs = np.array([r[xkey] for r in rows])            # each agent's x value (memory or fitness)
    xe = np.linspace(xs.min(), xs.max(), args.xbins + 1)   # x-axis block EDGES (xbins blocks)
    img = np.zeros((len(YBINS) - 1, args.xbins))      # pixel grid: rows = |λ| blocks, cols = x blocks
    # which x-column each agent falls in: digitize -> 1-based bin, -1 -> 0-based, clip the max
    # value (lands at xbins) back into [0, xbins-1].
    col = np.clip(np.digitize(xs, xe) - 1, 0, args.xbins - 1)
    for r, c in zip(rows, col):                       # for each agent (record r) and its column c
        img[:, c] += r["hist"]                        # add its full |λ| histogram into that column
    # img[:, x] is now the pooled |λ| histogram of all agents whose perf falls in x-block x.
    if args.global_norm:                              # whole image sums to 1: joint P(|λ|, perf)
        tot = img.sum()
        return (img / tot if tot > 0 else img), xe
    # per-column (default): divide each column by its column total -> each column sums to 1
    # -> P(|λ| | perf). axis=0 sums over the |λ| rows, keepdims keeps shape (1, xbins) for broadcast.
    s = img.sum(axis=0, keepdims=True)                # column totals, shape (1, xbins)
    # safe divide: only where s>0 (skip empty columns); out=zeros fills skipped cells with 0
    # (not nan), avoiding 0/0 on perf-blocks that contain no agents.
    return np.divide(img, s, out=np.zeros_like(img), where=s > 0), xe


# ---- Figure A: 2D density ----
figA, axes = plt.subplots(1, 2, figsize=(15, 5.5))
_excl = "" if args.min_lam <= 0 else f", |λ|<{args.min_lam:g} excluded"
figA.suptitle(f"Effective |λ| distribution vs performance ({args.cond}, n={len(rows)}{_excl})", fontsize=12)
for ax, xkey, xlab in [(axes[0], "mem", "memory_sum"), (axes[1], "fit", "fitness")]:
    img, xe = density(xkey)
    norm = mpl.colors.LogNorm(vmin=max(img[img > 0].min(), 1e-4), vmax=img.max()) if args.logcolor else None
    im = ax.imshow(img, origin="lower", aspect="auto", cmap="magma", norm=norm,
                   extent=[xe[0], xe[-1], YBINS[0], YBINS[-1]])
    ax.axhline(1.0, color="cyan", ls="--", lw=0.8); ax.text(xe[0], 1.02, "|λ|=1", color="cyan", fontsize=8)
    ax.set_xlabel(xlab, fontsize=11); ax.set_ylabel("|λ| (effective Jacobian)", fontsize=11)
    ax.set_title(f"P(|λ| | {xlab})", fontsize=11)
    figA.colorbar(im, ax=ax, label="probability (global)" if args.global_norm else "probability (per column)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs("Results/plots/misc", exist_ok=True)
outA = f"Results/plots/misc/plot_eig_dist_density_{args.dataset}_{args.cond}.png"
figA.savefig(outA, dpi=150, bbox_inches="tight")

# ---- Figure B: mean/std/max vs memory & fitness ----
figB, axB = plt.subplots(2, 3, figsize=(15, 9))
figB.suptitle(f"Effective |λ| summaries vs performance ({args.cond}, values)", fontsize=12)
for i, (xk, xl) in enumerate([("mem", "memory_sum"), ("fit", "fitness")]):
    for j, (mk, ml) in enumerate([("mean", "mean |λ|"), ("std", "std |λ| (spread)"),
                                  ("max", "max |λ| (spectral radius)")]):
        ax = axB[i, j]
        for s in present:
            xs = [r[xk] for r in rows if r["struct"] == s]
            ys = [r[mk] for r in rows if r["struct"] == s]
            ax.scatter(xs, ys, s=10, color=COLORS[s], alpha=0.5, edgecolors="none", label=s)
        ax.set_xlabel(xl, fontsize=10); ax.set_ylabel(ml, fontsize=10)
        ax.set_title(f"{ml} vs {xl}", fontsize=10)
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.96])
outB = f"Results/plots/misc/plot_eig_dist_summaries_{args.dataset}_{args.cond}.png"
figB.savefig(outB, dpi=150, bbox_inches="tight")
print(f"Saved {outA}\n      {outB}")
plt.show()   # pop up both figure windows
