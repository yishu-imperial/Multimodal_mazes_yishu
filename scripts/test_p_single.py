# test_p_single.py
#
# Single-agent training script for p-variation experiment (random graph only).
# Each job trains ONE agent with a specified edge probability p and cue_sparsity.
# Run 30 submissions × 7 p values × 2 cue_sparsity = 420 agents in parallel.
#
# Changelog:
#   2026-05-29 (v1): Initial version. Based on test_reciprocity_single.py (v2).
#                    Random graph only, p and cue_sparsity passed as arguments.

import os
import argparse
import pickle
import numpy as np
import networkx as nx
import utils
import multimodal_mazes

# --- Parse arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--p", type=float, default=0.3,
                    help="Edge probability for random graph (e.g. 0.3, 0.5, 0.9)")
parser.add_argument("--cue_sparsity", type=float, default=0.1,
                    help="Fraction of cues removed (e.g. 0.1 or 0.3)")
args = parser.parse_args()
p = args.p
cue_sparsity = args.cue_sparsity
wall_sparsity = 0.0

run_version = "v1_20260529"

job_id = os.environ.get("PBS_JOBID", "local")
print(f"Job {job_id} | p={p} | cue_sparsity={cue_sparsity}")

# --- Configuration ---
N_HIDDEN = 10
R_TARGET = 0.9
wm_flags = np.array([0, 1, 0, 0, 0, 0, 0])


# --- Generate random mask with edge probability p ---
def generate_mask(p, n):
    G = nx.fast_gnp_random_graph(n, p=p, directed=True)
    M = nx.to_numpy_array(G)
    return (M > 0).astype(float)


# --- Reciprocity ---
def compute_reciprocity(W):
    W_abs = np.abs(W)
    W_sym = np.minimum(W_abs, W_abs.T)
    total = W_abs.sum()
    return W_sym.sum() / total if total > 0 else np.nan


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


# --- Generate mazes ---
maze_train = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
maze_train.generate(number=400000, cue_sparsity=cue_sparsity, wall_sparsity=wall_sparsity)

maze_test = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
maze_test.generate(number=1000, cue_sparsity=cue_sparsity, wall_sparsity=wall_sparsity)

exp_config = {"channels": [1, 1], "n_steps": 20}

# --- Generate mask and initialise W_hh ---
M = generate_mask(p, N_HIDDEN)
fan_in = M.sum(axis=0)
fan_in_safe = np.where(fan_in == 0, 1, fan_in)
W_init = np.random.uniform(0, 1, size=M.shape) * M / np.sqrt(fan_in_safe)

try:
    W_09 = utils.adjust_reciprocity_weighted(W_init.copy(), R_TARGET)
except Exception as e:
    print(f"Failed to adjust reciprocity: {e}")
    raise

h_init = heterogeneity_cosine(W_09)

# --- Train agent ---
agnt = multimodal_mazes.AgentDQN(
    location=None,
    channels=[1, 1],
    sensor_noise_scale=0.05,
    n_hidden_units=N_HIDDEN,
    wm_flags=wm_flags,
    M_hh=M,
    W_hh_init=W_09,
    conserve_weight_sum=False,
)
agnt.generate_policy(maze=maze_train, n_steps=20)

# --- Test ---
test_results, _, memory = multimodal_mazes.test_dqn_agent(
    maze_test=maze_test,
    agnt=agnt,
    exp_config=exp_config,
    noises=[0.05],
)
fitness = test_results[0]
memory_sum = np.nansum(memory[0])

# --- Extract trained W_hh ---
W_trained = agnt.hidden_to_hidden.weight.detach().numpy()
r = compute_reciprocity(W_trained)
h = heterogeneity_cosine(W_trained)

result = {
    "p": p,
    "cue_sparsity": cue_sparsity,
    "job_id": job_id,
    "heterogeneity_init": h_init,
    "reciprocity": r,
    "heterogeneity": h,
    "fitness": fitness,
    "memory_sum": memory_sum,
    "memory_decay": memory[0],
    "W_init": W_init,
    "W_09": W_09,
    "W_trained": W_trained,
}

print(f"  p={p} | cs={cue_sparsity} | h_init={h_init:.3f} | r={r:.3f} | h={h:.3f} | fitness={fitness:.3f} | memory_sum={memory_sum:.3f}")

# --- Save ---
out_dir = f"results_p_variation_{run_version}"
os.makedirs(out_dir, exist_ok=True)
p_str = f"{p:.1f}".replace(".", "")
cs_str = f"{cue_sparsity:.1f}".replace(".", "")
out_path = f"{out_dir}/result_p{p_str}_cs{cs_str}_{job_id}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(result, f)
print(f"Saved {out_path}")
