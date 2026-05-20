import numpy as np
import matplotlib.pyplot as plt
import multimodal_mazes

# --- Configuration ---
N_HIDDEN = 10
wm_flags = np.array([0, 1, 0, 0, 0, 0, 0])
N_STRUCTURES = 60
n_swaps_list = [0, 2, 5, 10, 20, 50]

# --- Step 1: Initialise M_hh as block diagonal ---
M_hh_init = np.zeros((N_HIDDEN, N_HIDDEN))
for i in range(5):
    M_hh_init[2*i:2*i+2, 2*i:2*i+2] = 1

# --- Step 2: Swap operation (rank-preserving) ---
def swap_M_hh(M, n_swaps=1):
    """
    Randomly swap one 1 and one 0, preserving sparsity (number of 1s).
    Returns new M_hh.
    """
    M_new = M.copy()

    for _ in range(n_swaps):
        ones = list(zip(*np.where(M_new == 1)))
        zeros = list(zip(*np.where(M_new == 0)))
        pos1 = ones[np.random.randint(len(ones))]
        pos0 = zeros[np.random.randint(len(zeros))]
        M_new[pos1] = 0
        M_new[pos0] = 1

    return M_new

# --- Step 4: Reciprocity (on W_hh, using absolute values) ---
def compute_reciprocity(W):
    """
    Compute reciprocity of W_hh.
    r = sum(min(|W_ij|, |W_ji|)) / sum(|W_ij|)
    """
    W_abs = np.abs(W)
    W_sym = np.minimum(W_abs, W_abs.T)
    r = W_sym.sum() / W_abs.sum()
    return r

# --- Step 5: Heterogeneity (on M_hh) ---

# --- Method 1: Jaccard similarity with motif counts (commented out) ---
# def get_motif_counts(M):
#     n = M.shape[0]
#     counts = [{'FF': 0, 'cycle': 0, 'skip': 0} for _ in range(n)]
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 continue
#             if M[i, j] == 1:
#                 if M[j, i] == 1:
#                     counts[i]['cycle'] += 1
#                 else:
#                     counts[i]['FF'] += 1
#             for k in range(n):
#                 if M[i,j]==1 and M[j,k]==1 and M[i,k]==1 and k!=i and k!=j:
#                     counts[i]['skip'] += 1
#     return counts
#
# def counts_to_set(counts_dict):
#     s = set()
#     for motif, count in counts_dict.items():
#         for c in range(count):
#             s.add(f'{motif}_{c}')
#     return s
#
# def compute_heterogeneity(M):
#     n = M.shape[0]
#     counts = get_motif_counts(M)
#     motif_sets = [counts_to_set(c) for c in counts]
#     Z = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             inter = len(motif_sets[i] & motif_sets[j])
#             union = len(motif_sets[i] | motif_sets[j])
#             Z[i, j] = inter / union if union > 0 else 0
#     return n / Z.sum()

# --- Method 2: Hamming similarity on M_hh rows ---
def compute_heterogeneity(M):
    """
    Compute heterogeneity of M_hh using row-wise Hamming similarity.
    Z_ij = number of matching positions between row i and row j / n
    Heterogeneity = 1 - n^2 / sum_ij Z_ij
    0 = fully homogeneous, 1 = fully heterogeneous.
    """
    n = M.shape[0]

    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            intersection = np.sum((M[i] == 1) & (M[j] == 1))
            union = np.sum((M[i] == 1) | (M[j] == 1))
            Z[i, j] = intersection / union if union > 0 else 0

    return 1 - Z.sum() / (n ** 2)

def compute_heterogeneity_cosine(W):
    """
    Cosine similarity-based heterogeneity on W_hh rows.
    Z_ij = cosine similarity between row i and row j.
    Heterogeneity = 1 - sum_ij Z_ij / n^2
    note: the else branch has not been touched yet, but the Z[i, j] = 0 assumption might not be correct
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

# --- Verify swap is working ---
M_test = swap_M_hh(M_hh_init, n_swaps=5)
print("M changed:", not np.array_equal(M_test, M_hh_init))
print("Rank before:", np.linalg.matrix_rank(M_hh_init))
print("Rank after: ", np.linalg.matrix_rank(M_test))

# --- Step 3: Generate maze (local testing) ---
maze_train = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
maze_train.generate(number=1000)  # use 400000 on HPC

# --- Step 6: Main loop ---
reciprocities = []
heterogeneities = []
n_swaps_labels = []

for n_swaps in n_swaps_list:
    n_per_swap = N_STRUCTURES // len(n_swaps_list)
    for _ in range(n_per_swap):
        # Generate M_hh
        M = swap_M_hh(M_hh_init, n_swaps=n_swaps)

        # Train agent
        agnt = multimodal_mazes.AgentDQN(
            location=None,
            channels=[1, 1],
            sensor_noise_scale=0.05,
            n_hidden_units=N_HIDDEN,
            wm_flags=wm_flags,
            M_hh=M,
        )
        agnt.generate_policy(maze=maze_train, n_steps=20)

        # Extract W_hh
        W_hh = agnt.hidden_to_hidden.weight.detach().numpy()

        # Compute metrics
        r = compute_reciprocity(W_hh)
        h = compute_heterogeneity_cosine(W_hh)

        reciprocities.append(r)
        heterogeneities.append(h)
        n_swaps_labels.append(n_swaps)

        print(f"  n_swaps={n_swaps} | reciprocity={r:.4f} | heterogeneity={h:.4f}")

# --- Plot ---
reciprocities = np.array(reciprocities)
heterogeneities = np.array(heterogeneities)
n_swaps_labels = np.array(n_swaps_labels)

plt.figure(figsize=(7, 5))
scatter = plt.scatter(
    reciprocities,
    heterogeneities,
    c=n_swaps_labels,
    cmap='viridis',
    s=60,
    edgecolors='k',
    linewidths=0.5,
)
plt.colorbar(scatter, label="Number of swaps")
plt.xlabel("Reciprocity of W_hh")
plt.ylabel("Heterogeneity of W_hh (cosine)")
plt.title("Reciprocity vs Heterogeneity")
plt.tight_layout()
plt.savefig("structure_analysis.png", dpi=150)
plt.show()
print("Saved structure_analysis.png")
