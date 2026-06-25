# check_alpha_outdegree.py
#
# Diagnostic for the n=100 alpha-controlled OUT-DEGREE skew design.
#
# Idea: keep total edges fixed (so W_hh parameter count is fixed), and control
# skewness with a single knob alpha via a RANK power law on the out-degree:
#     extra_r  proportional to  r^(-alpha)        (r = 1 .. n, rank 1 = biggest)
# To guarantee every node has out-degree >= `floor` (no dead nodes -> recurrent
# core survives), reserve floor*n edges first, then distribute the remaining
# (total - floor*n) edges by the power law.
#
# Key fact: mean out-degree = total/n. "min >= floor" + "mean = floor" => all equal.
# So skew with a floor REQUIRES total > floor*n. With floor=2, total=400 (mean 4)
# leaves 200 edges of headroom for skew. This script quantifies how much skew that
# actually buys across alpha.
#
# Usage:
#   python3 scripts/check_alpha_outdegree.py                         # total=400, floor=2
#   python3 scripts/check_alpha_outdegree.py --total 200 --floor 1   # sparse, allow dead
#   python3 scripts/check_alpha_outdegree.py --n 100 --alphas 0 0.25 0.5 0.75 1.0

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--total", type=int, default=400, help="total out-edges (fixed parameter budget)")
parser.add_argument("--floor", type=int, default=2, help="guaranteed minimum out-degree per node")
parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
args = parser.parse_args()


def alpha_outdegree_sequence(n, total, alpha, floor=2):
    """Rank power-law out-degree sequence with a per-node floor.

    Floor every node, then split the surplus (total - floor*n) by rank weight
    r^(-alpha) (rank 1 = biggest hub). Returns descending out-degrees summing
    EXACTLY to total, each in [floor, n-1]. (See RESULTS_NOTES "n=100 alpha
    out-degree design" for the why behind each step.)
    """
    if total < floor * n:                        # min>=floor needs total>floor*n (else all == floor, no skew)
        raise ValueError(f"total={total} < floor*n={floor*n}: floor infeasible "
                         f"(mean {total/n:.2f} < floor {floor})")
    extra = total - floor * n                    # budget free to create skew

    w = np.arange(1, n + 1) ** (-alpha)          # skew knob: alpha=0 uniform; larger alpha -> front ranks dominate
    w = w / w.sum()                              # shares (divisor = Z = sum_r r^-alpha)
    deg = floor + w * extra                      # fractional target degree per node (length-n array)

    deg_int = np.floor(deg).astype(int)          # round down -> sum too small; hand the remainder back
    frac = deg - deg_int
    rem = int(round(total - deg_int.sum()))      # edges lost to flooring (0 <= rem < n)
    order = np.argsort(-frac)                     # node indices by chopped-off fraction, largest first
    for k in range(rem):                         # largest-remainder: refill so sum == total exactly
        deg_int[order[k]] += 1

    deg_int = np.minimum(deg_int, n - 1)         # no self-loops -> cap at n-1 (drops over-cap hub edges)
    short = int(total - deg_int.sum())
    j = 0
    while short > 0 and j < 1000 * n:            # re-place capped edges on nodes with room (safety valve 1000*n)
        idx = order[j % n]
        if deg_int[idx] < n - 1:
            deg_int[idx] += 1
            short -= 1
        j += 1

    return sorted(deg_int.tolist(), reverse=True)


n, total, floor = args.n, args.total, args.floor
print(f"n={n}  total={total}  floor={floor}  mean out-degree={total/n:.2f}\n")
print(f"{'alpha':>6} | {'min':>4} {'max':>4} | {'dead':>4} | {'top1%':>6} {'top2%':>6} {'top5%':>6}")
print("-" * 48)

rows = []
for a in args.alphas:
    deg = np.array(alpha_outdegree_sequence(n, total, a, floor=floor))
    assert deg.sum() == total, f"sum {deg.sum()} != {total}"
    dead = int((deg == 0).sum())
    top1 = deg[0] / total
    top2 = deg[:2].sum() / total
    top5 = deg[:5].sum() / total
    rows.append({"alpha": a, "deg": deg, "min": deg.min(), "max": deg.max(),
                 "dead": dead, "top1": top1, "top2": top2, "top5": top5})
    print(f"{a:>6.2f} | {deg.min():>4d} {deg.max():>4d} | {dead:>4d} | "
          f"{top1:>5.1%} {top2:>5.1%} {top5:>5.1%}")

# ---------------- plots ----------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(f"alpha-controlled out-degree skew  (n={n}, total={total}, floor={floor}, "
             f"mean={total/n:.0f})", fontsize=13)

cmap = plt.cm.viridis(np.linspace(0, 0.9, len(rows)))
for c, row in zip(cmap, rows):
    axL.plot(np.arange(1, n + 1), row["deg"], color=c, lw=1.6, label=f"alpha={row['alpha']:.2f}")
axL.axhline(floor, color="gray", ls="--", lw=0.9)
axL.text(n * 0.6, floor + 0.3, f"floor={floor}", color="gray", fontsize=8)
axL.set_xlabel("rank (1 = biggest hub)", fontsize=11)
axL.set_ylabel("out-degree", fontsize=11)
axL.set_title("out-degree vs rank", fontsize=11)
axL.legend(fontsize=8)

alphas = [r["alpha"] for r in rows]
axR.plot(alphas, [r["top1"] for r in rows], "o-", label="top-1 share")
axR.plot(alphas, [r["top2"] for r in rows], "s-", label="top-2 share")
axR.plot(alphas, [r["top5"] for r in rows], "^-", label="top-5 share")
axR.set_xlabel("alpha (skew knob)", fontsize=11)
axR.set_ylabel("concentration (top-k share)", fontsize=11)
axR.set_title("hub concentration vs alpha", fontsize=11)
axR.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs("Results/plots/misc", exist_ok=True)
out = f"Results/plots/misc/plot_alpha_outdegree_total{total}_floor{floor}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
