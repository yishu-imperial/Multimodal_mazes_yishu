# test_p_structure.py
#
# Diagnostic: for random graphs with different edge probabilities p,
# plot heterogeneity (cosine) vs reciprocity at different R_TARGETS.
# Goal: find the R_TARGET where heterogeneity spread across p values is largest.
#
# Usage:
#   python3 test_p_structure.py

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import utils

# --- Configuration ---
N_HIDDEN = 10
N_GRAPHS = 10  # masks per p value
P_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
R_TARGETS = [0.1, 0.3, 0.5, 0.7, 0.9]


# --- Generate mask ---
def generate_mask(p, n):
    G = nx.fast_gnp_random_graph(n, p=p, directed=True)
    M = nx.to_numpy_array(G)
    return (M > 0).astype(float)


# --- Initialise W_hh (fan-in normalisation) ---
def init_W_hh(M):
    fan_in = M.sum(axis=0)
    fan_in_safe = np.where(fan_in == 0, 1, fan_in)
    W = np.random.uniform(0, 1, size=M.shape) * M / np.sqrt(fan_in_safe)
    return W


# --- Cosine heterogeneity ---
def heterogeneity_cosine(W):
    n = W.shape[0]
    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            norm_i = np.linalg.norm(W[i])
            norm_j = np.linalg.norm(W[j])
            if norm_i > 0 and norm_j > 0:
                Z[i, j] = np.dot(W[i], W[j]) / (norm_i * norm_j)
            else:
                Z[i, j] = 0
    return 1 - Z.sum() / (n ** 2)


# --- Main loop ---
results = []

for p in P_VALUES:
    for _ in range(N_GRAPHS):
        M = generate_mask(p, N_HIDDEN)
        W_init = init_W_hh(M)

        for r_target in R_TARGETS:
            try:
                W_adj = utils.adjust_reciprocity_weighted(W_init.copy(), r_target)
                r = utils.compute_reciprocity_weighted(W_adj.copy())
                h = heterogeneity_cosine(W_adj)
                results.append({"p": p, "r_target": r_target, "reciprocity": r, "h_cosine": h})
            except Exception as e:
                print(f"Skipped p={p} r_target={r_target}: {e}")

# --- Print spread at each R_TARGET ---
print("\nHeterogeneity spread across p values per R_TARGET:")
print(f"{'R_TARGET':>10} | {'h_cosine range':>20} | {'spread (max-min)':>18}")
print("-" * 55)
for r_target in R_TARGETS:
    vals = [res["h_cosine"] for res in results if res["r_target"] == r_target]
    if vals:
        spread = max(vals) - min(vals)
        print(f"{r_target:>10.1f} | [{min(vals):.4f}, {max(vals):.4f}]       | {spread:>18.4f}")

# --- Plot ---
cmap = plt.cm.viridis
p_norm = {p: i / (len(P_VALUES) - 1) for i, p in enumerate(P_VALUES)}

fig, ax = plt.subplots(figsize=(9, 6))

for p in P_VALUES:
    color = cmap(p_norm[p])
    xs = [res["reciprocity"] for res in results if res["p"] == p]
    ys = [res["h_cosine"]    for res in results if res["p"] == p]
    ax.scatter(xs, ys, color=color, alpha=0.6, s=50)

    # trend line
    if len(xs) > 1:
        z = np.polyfit(xs, ys, 1)
        x_line = np.linspace(min(xs), max(xs), 100)
        ax.plot(x_line, np.polyval(z, x_line), color=color, linewidth=2, label=f"p={p}")

ax.set_xlabel("Reciprocity of W_hh")
ax.set_ylabel("Heterogeneity (cosine)")
ax.set_title("Random graph: Heterogeneity vs Reciprocity for different p values")
ax.legend(title="p", fontsize=9)
plt.tight_layout()
plt.savefig("test_p_structure.png", dpi=150)
plt.show()
print("\nSaved test_p_structure.png")
