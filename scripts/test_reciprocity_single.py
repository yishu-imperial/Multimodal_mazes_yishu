# test_reciprocity_single.py
#
# Single-agent version of test_reciprocity_init_training.py.
# Designed to be run as a PBS array job (-J 1-3):
#   PBS_ARRAY_INDEX=1 → random
#   PBS_ARRAY_INDEX=2 → small_world
#   PBS_ARRAY_INDEX=3 → modular
#
# Each job trains ONE agent and saves ONE result pkl.
# Run 50 submissions × 3 array jobs = 150 agents in parallel.

import os
import argparse
import pickle
import numpy as np
import networkx as nx
import utils
import multimodal_mazes

# --- Parse arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--conserve", type=str, default="true")
parser.add_argument("--task", type=str, default="M",
                    help="Task to train on: 'M' or 'Msc'")
args = parser.parse_args()
conserve = args.conserve.lower() == "true"
task = args.task
suffix = "conserve" if conserve else "no_conserve"

# --- Task parameters ---
task_params = {
    "M":   {"cue_sparsity": 0.0, "wall_sparsity": 0.0},
    "Msc": {"cue_sparsity": 0.1, "wall_sparsity": 0.0},
}
assert task in task_params, f"Unknown task '{task}'. Choose from: {list(task_params.keys())}"
cue_sparsity  = task_params[task]["cue_sparsity"]
wall_sparsity = task_params[task]["wall_sparsity"]

# --- PBS array index → graph type ---
idx = int(os.environ.get("PBS_ARRAY_INDEX", 1))
graph_types = {1: "random", 2: "small_world", 3: "modular"}
graph_type = graph_types[idx]

job_id = os.environ.get("PBS_JOBID", "local")
print(f"Job {job_id} | task={task} | graph_type={graph_type} | conserve={conserve}")

# --- Configuration ---
N_HIDDEN = 10
R_TARGET = 0.9
wm_flags = np.array([0, 1, 0, 0, 0, 0, 0])


# --- Generate one M_hh ---
def generate_mask(graph_type, n):
    if graph_type == "random":
        G = nx.fast_gnp_random_graph(n, p=0.3, directed=True)
    elif graph_type == "small_world":
        G = nx.connected_watts_strogatz_graph(n, k=4, p=0.3).to_directed()
    elif graph_type == "modular":
        sz = [5, 5]
        pr = [[0.8, 0.1], [0.1, 0.8]]
        G = nx.stochastic_block_model(sz, pr)
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
M = generate_mask(graph_type, N_HIDDEN)
W_init = np.random.uniform(0, 1, size=M.shape) * M

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
    conserve_weight_sum=conserve,
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
    "graph_type": graph_type,
    "task": task,
    "conserve": conserve,
    "job_id": job_id,
    "heterogeneity_init": h_init,
    "reciprocity": r,
    "heterogeneity": h,
    "fitness": fitness,
    "memory_sum": memory_sum,
    "memory_decay": memory[0],
    "W_trained": W_trained,
}

print(f"  {graph_type} | h_init={h_init:.3f} | r={r:.3f} | h={h:.3f} | fitness={fitness:.3f} | memory_sum={memory_sum:.3f}")

# --- Save ---
os.makedirs("results_reciprocity", exist_ok=True)
out_path = f"results_reciprocity/result_{suffix}_{task}_{graph_type}_{job_id}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(result, f)
print(f"Saved {out_path}")
