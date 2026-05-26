# analyse_heterogeneity.py
#
# Load trained agents from results/ and plot:
#   Figure 1: heterogeneity (cosine) of W_hh * M_hh vs fitness
#   Figure 2: heterogeneity (cosine) of W_hh * M_hh vs memory sum
# Only Identity and Full masks are included (Zero is excluded).

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
results_dir = "results/"
mask_map    = {"0": "Identity", "1": "Full"}   # skip "2" (Zero)
colors      = {"Identity": "C0", "Full": "C1"}


# --- Heterogeneity (cosine) on effective weights W_hh * M_hh ---
def heterogeneity_cosine(W):
    """
    Heterogeneity based on pairwise cosine similarity of rows of W.
    Z_ij = cosine similarity between row i and row j.
    Heterogeneity = 1 - sum_ij Z_ij / n^2
    """
    n = W.shape[0]
    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            norm_i = np.linalg.norm(W[i])
            norm_j = np.linalg.norm(W[j])
            if norm_i > 0 and norm_j > 0:
                Z[i, j] = np.dot(W[i], W[j]) / (norm_i * norm_j)
            else:
                # If one or both rows are zero vectors, cosine similarity is undefined.
                # We set Z[i,j] = 0 (treat as "completely dissimilar") as a placeholder,
                # but this is arbitrary — alternatives include np.nan or 1.0.
                # In our experiments this branch has never been triggered (checked across
                # all 800 rows in results/), so this choice does not affect current results.
                Z[i, j] = 0
    return 1 - Z.sum() / (n ** 2)


# --- Load results ---
folders = sorted([f for f in os.listdir(results_dir) if f.startswith("test")])
print(f"Found {len(folders)} runs")

records = []  # list of dicts

for folder in folders:
    path = os.path.join(results_dir, folder)
    for fname, mask_name in mask_map.items():
        fpath = os.path.join(path, fname + ".pickle")
        if not os.path.exists(fpath):
            continue
        with open(fpath, "rb") as f:
            agnt = pickle.load(f)

        # Effective weights: W_hh * M_hh
        W = agnt.hidden_to_hidden.weight.detach().numpy()
        M = agnt.M_hh.numpy()
        W_eff = W * M

        h  = heterogeneity_cosine(W_eff)
        fitness     = agnt.results[0]
        memory_sum  = np.nansum(agnt.memory[0])

        records.append({
            "mask":       mask_name,
            "heterogeneity": h,
            "fitness":    fitness,
            "memory_sum": memory_sum,
        })

print(f"Loaded {len(records)} records")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for mask_name, c in colors.items():
    subset = [r for r in records if r["mask"] == mask_name]
    hs  = [r["heterogeneity"] for r in subset]
    fit = [r["fitness"]       for r in subset]
    mem = [r["memory_sum"]    for r in subset]

    axes[0].scatter(hs, fit, color=c, label=mask_name, alpha=0.7,
                    edgecolors='k', linewidths=0.5, s=60)
    axes[1].scatter(hs, mem, color=c, label=mask_name, alpha=0.7,
                    edgecolors='k', linewidths=0.5, s=60)

# Add trend lines for Full mask only
full = [r for r in records if r["mask"] == "Full"]
hs_full  = np.array([r["heterogeneity"] for r in full])
fit_full = np.array([r["fitness"]       for r in full])
mem_full = np.array([r["memory_sum"]    for r in full])

for ax, ys in zip(axes, [fit_full, mem_full]):
    m, b = np.polyfit(hs_full, ys, 1)
    x_line = np.linspace(hs_full.min(), hs_full.max(), 100)
    ax.plot(x_line, m * x_line + b, color="C1", linewidth=2, linestyle="--")

axes[0].set_xlabel("Heterogeneity of W_hh * M_hh (cosine)", fontsize=11)
axes[0].set_ylabel("Fitness", fontsize=11)
axes[0].set_title("Heterogeneity vs Fitness", fontsize=12)
axes[0].legend(fontsize=10)

axes[1].set_xlabel("Heterogeneity of W_hh * M_hh (cosine)", fontsize=11)
axes[1].set_ylabel("Memory sum", fontsize=11)
axes[1].set_title("Heterogeneity vs Memory sum", fontsize=12)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig("heterogeneity_analysis.png", dpi=150)
plt.show()
print("Saved heterogeneity_analysis.png")
