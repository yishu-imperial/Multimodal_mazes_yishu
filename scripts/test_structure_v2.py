import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import utils

# --- Configuration ---
N_HIDDEN = 10
N_GRAPHS = 5
R_TARGETS = [0.1, 0.3, 0.5, 0.7, 0.9]

# --- Step 1: Generate M_hh from different graph structures ---
def generate_masks(n, n_graphs):
    """
    Generate binary M_hh matrices from three graph types:
    - Erdos-Renyi (random)
    - Small world (Watts-Strogatz)
    - Modular (stochastic block model)
    Returns a dict of {graph_type: [M_hh, ...]}
    """
    masks = {}

    # Erdos-Renyi: random directed graph
    masks["random"] = [
        nx.to_numpy_array(nx.fast_gnp_random_graph(n, p=0.3, directed=True))
        for _ in range(n_graphs)
    ]

    # Small world: undirected then converted to directed
    masks["small_world"] = [
        nx.to_numpy_array(
            nx.connected_watts_strogatz_graph(n, k=4, p=0.3).to_directed()
        )
        for _ in range(n_graphs)
    ]

    # Modular: stochastic block model with 2 blocks of 5
    sz = [5, 5]
    pr = [[0.8, 0.1], [0.1, 0.8]]
    masks["modular"] = [
        nx.to_numpy_array(nx.stochastic_block_model(sz, pr))
        for _ in range(n_graphs)
    ]

    # Binarise all matrices
    for graph_type in masks:
        processed = []
        for M in masks[graph_type]:
            M = (M > 0).astype(float)
            processed.append(M)
        masks[graph_type] = processed

    return masks


# --- Step 2: Initialise non-negative W_hh from M_hh ---
def init_W_hh(M):
    """
    Initialise W_hh with uniform(0,1) weights only where M_hh == 1.
    """
    W = np.random.uniform(0, 1, size=M.shape) * M
    return W


# --- Step 4a: Motif-based heterogeneity ---
def get_motif_counts(W, threshold=0):
    """
    Count motif participation for each neuron.
    Edge exists if W[i,j] > threshold.
    Motifs: FF (feedforward), cycle, skip.
    """
    n = W.shape[0]
    A = (W > threshold).astype(int)
    counts = [{'FF': 0, 'cycle': 0, 'skip': 0} for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if A[i, j] == 1:
                if A[j, i] == 1:
                    counts[i]['cycle'] += 1
                else:
                    counts[i]['FF'] += 1
            # skip motif: i->j->k and i->k
            for k in range(n):
                if A[i,j]==1 and A[j,k]==1 and A[i,k]==1 and k!=i and k!=j:
                    counts[i]['skip'] += 1

    return counts


def counts_to_set(counts_dict):
    """
    Expand count dict to set for Jaccard similarity.
    e.g. {'cycle': 2, 'FF': 1} -> {'cycle_0', 'cycle_1', 'FF_0'}
    """
    s = set()
    for motif, count in counts_dict.items():
        for c in range(count):
            s.add(f'{motif}_{c}')
    return s


def heterogeneity_motif(W):
    """
    Motif-based heterogeneity: n / sum_ij Z_ij
    where Z_ij = Jaccard similarity of motif count sets.
    """
    n = W.shape[0]
    counts = get_motif_counts(W)
    motif_sets = [counts_to_set(c) for c in counts]

    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = len(motif_sets[i] & motif_sets[j])
            union = len(motif_sets[i] | motif_sets[j])
            Z[i, j] = inter / union if union > 0 else 0

    total = Z.sum()
    return n / total if total > 0 else np.nan


# --- Step 4b: Cosine similarity-based heterogeneity ---
def heterogeneity_cosine(W):
    """
    Cosine similarity-based heterogeneity: 1 - sum_ij Z_ij / n^2
    where Z_ij = cosine similarity between row i and row j of W.
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
                Z[i, j] = 0
    return 1 - Z.sum() / (n ** 2)


# --- Main loop ---
masks = generate_masks(N_HIDDEN, N_GRAPHS)

results = []

for graph_type, M_list in masks.items():
    for M in M_list:
        W_init = init_W_hh(M)

        for r_d in R_TARGETS:
            try:
                W_adj = utils.adjust_reciprocity_weighted(W_init.copy(), r_d)
                r = utils.compute_reciprocity_weighted(W_adj.copy())
                h_motif = heterogeneity_motif(W_adj)
                h_cosine = heterogeneity_cosine(W_adj)

                results.append({
                    "graph_type": graph_type,
                    "r_d": r_d,
                    "reciprocity": r,
                    "h_motif": h_motif,
                    "h_cosine": h_cosine,
                })

                print(f"  {graph_type} | r_d={r_d} | r={r:.3f} | "
                      f"h_motif={h_motif:.3f} | h_cosine={h_cosine:.3f}")

            except Exception as e:
                print(f"  Skipped {graph_type} r_d={r_d}: {e}")

# --- Plot ---
colors = {"random": "C0", "small_world": "C1", "modular": "C2"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for res in results:
    gt = res["graph_type"]
    axes[0].scatter(res["reciprocity"], res["h_motif"],
                    color=colors[gt], alpha=0.7, edgecolors='k', linewidths=0.5, s=60)
    axes[1].scatter(res["reciprocity"], res["h_cosine"],
                    color=colors[gt], alpha=0.7, edgecolors='k', linewidths=0.5, s=60)

# Trend lines for cosine plot (one per graph type)
for gt, c in colors.items():
    xs = np.array([res["reciprocity"] for res in results if res["graph_type"] == gt])
    ys = np.array([res["h_cosine"]    for res in results if res["graph_type"] == gt])
    if len(xs) > 1:
        m, b = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 100)
        axes[1].plot(x_line, m * x_line + b, color=c, linewidth=2)

# Legend
for gt, c in colors.items():
    axes[0].scatter([], [], color=c, label=gt)
    axes[1].scatter([], [], color=c, label=gt)

axes[0].set_xlabel("Reciprocity of W_hh")
axes[0].set_ylabel("Heterogeneity (motif-based)")
axes[0].set_title("Reciprocity vs Heterogeneity (Motif)")
axes[0].legend()

axes[1].set_xlabel("Reciprocity of W_hh")
axes[1].set_ylabel("Heterogeneity (cosine)")
axes[1].set_title("Reciprocity vs Heterogeneity (Cosine)")
axes[1].legend()

plt.tight_layout()
plt.savefig("structure_analysis_v2.png", dpi=150)
plt.show()
print("Saved structure_analysis_v2.png")
