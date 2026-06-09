# test_degseq_single.py
#
# Single-agent training script for degree sequence experiment.
# Fixed total edges = 18 (corresponding to p=0.2 in random graph).
# Each job trains ONE agent with a specified degree sequence type.
# Run 30 submissions x 4 deg_types = 120 agents in parallel.
#
# Changelog:
#   2026-06-07 (v1): Initial version. Fixed total_edges=18, cs=0.3, noise=0.05.
#                    No reciprocity adjustment. het() from notebook.
#   2026-06-08 (v2): Parametrise --cue_sparsity and --noise (defaults unchanged).
#                    Record training_fitness (maze_test passed to generate_policy).
#                    Store cue_sparsity / noise / task in result dict.
#                    Removed h_trained (het of trained W) and its "heterogeneity"
#                    result-dict key; this experiment only uses mask heterogeneity
#                    (h_init). W_trained is still stored. merge script updated to match.
#                    FIX: medium/high skew coefficients were swapped (0.4/0.35) so
#                    medium was MORE skewed than high. Now medium=0.35, high=0.4 →
#                    monotonic skew regular < medium < high < hub.
#                    NOTE: v1 (v1_20260607_degseq) data has medium/high LABELS swapped
#                    vs this corrected definition; not re-run.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import pickle
import numpy as np
import multimodal_mazes

# --- Parse arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--deg_type", type=str, default="regular",
                    help="Degree sequence type: regular / medium / high / hub")
parser.add_argument("--version", type=str, default="v1_20260607_degseq",
                    help="Version tag for output directory")
parser.add_argument("--cue_sparsity", type=float, default=0.3,
                    help="Cue sparsity (0.0 -> Task M, >0 -> Task Msc)")
parser.add_argument("--noise", type=float, default=0.05,
                    help="Sensor noise scale")
args = parser.parse_args()

deg_type    = args.deg_type
run_version = args.version

job_id = os.environ.get("PBS_JOBID", "local")
print(f"Job {job_id} | deg_type={deg_type}")

# --- Configuration ---
N     = 10
TOTAL = 18    # fixed total edges (corresponds to p=0.2)
CS    = args.cue_sparsity
NOISE = args.noise
TASK  = "M" if CS == 0.0 else "Msc"   # task label derived from cue sparsity
print(f"  cue_sparsity={CS} | noise={NOISE} | task={TASK}")

# --- Heterogeneity (from notebook) ---
def het(M):
    n = M.shape[0]
    assert M.shape == (n, n)
    colnorms = ((M**2).sum(axis=1))**.5
    return 1 - ((M[:, None, :] * M[None, :, :]).sum(axis=2) /
                (np.maximum(1, colnorms[:, None] * colnorms[None, :]))).mean()

# --- Degree sequence mask ---
def degree_sequence_mask(n, out_degrees, num_samples=1):
    masks = []
    for _ in range(num_samples):
        M = np.zeros((n, n), dtype=int)
        for i, d in enumerate(out_degrees):
            available = [j for j in range(n) if j != i]
            if d > 0:
                targets = np.random.choice(available,
                                           size=min(d, len(available)),
                                           replace=False)
                M[i, targets] = 1
        masks.append(M)
    return masks

# --- Degree sequences (total=18, n=10) ---
def make_degree_sequences(n, total):
    max_deg = n - 1
    sequences = {}

    # Regular
    avg = total // n
    remainder = total % n
    regular = sorted([avg + (1 if i < remainder else 0) for i in range(n)], reverse=True)
    sequences['regular'] = regular

    # Medium skew: top 2 nodes get ~35% of edges each
    hub_med = min(max_deg, int(total * 0.35))
    remaining_med = total - 2 * hub_med
    rest_med = [0] * (n - 2)
    per_node = remaining_med // (n - 2)
    leftover = remaining_med % (n - 2)
    for i in range(n - 2):
        rest_med[i] = per_node + (1 if i < leftover else 0)
    medium = [hub_med, hub_med] + rest_med
    if sum(medium) == total and all(0 <= d <= max_deg for d in medium):
        sequences['medium'] = sorted(medium, reverse=True)

    # High skew: top 2 nodes get ~40% of edges each
    hub_high = min(max_deg, int(total * 0.4))
    remaining_high = total - 2 * hub_high
    rest_high = [0] * (n - 2)
    for i in range(min(remaining_high, n - 2)):
        rest_high[i] = 1
    high = [hub_high, hub_high] + rest_high
    if sum(high) == total and all(0 <= d <= max_deg for d in high):
        sequences['high'] = sorted(high, reverse=True)

    # Hub-spoke: 2 hubs take maximum possible edges
    hub2 = min(max_deg, total // 2)
    remaining_hub = total - 2 * hub2
    rest_hub = [0] * (n - 2)
    for i in range(min(remaining_hub, n - 2)):
        rest_hub[i] = 1
    hub_spoke = [hub2, hub2] + rest_hub
    if sum(hub_spoke) == total:
        sequences['hub'] = sorted(hub_spoke, reverse=True)

    return sequences

# --- Generate degree sequences and select ---
deg_seqs = make_degree_sequences(N, TOTAL)
if deg_type not in deg_seqs:
    raise ValueError(f"deg_type '{deg_type}' could not be constructed for total={TOTAL}. "
                     f"Available: {list(deg_seqs.keys())}")

deg = deg_seqs[deg_type]
print(f"  degree_sequence={deg}  sum={sum(deg)}")

# --- Generate M_hh ---
M = degree_sequence_mask(N, deg, num_samples=1)[0]

# --- Initialise W_hh ---
fan_in      = M.sum(axis=0)
fan_in_safe = np.where(fan_in == 0, 1, fan_in)
W_init      = np.random.uniform(0, 1, size=M.shape) * M / np.sqrt(fan_in_safe)

# --- Compute h_init on M_hh ---
h_init = het(M)

# --- Generate mazes ---
wm_flags = np.array([0, 1, 0, 0, 0, 0, 0])

maze_train = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
maze_train.generate(number=400000, cue_sparsity=CS)

maze_test = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
maze_test.generate(number=1000, cue_sparsity=CS)

exp_config = {"channels": [1, 1], "n_steps": 20}

# --- Train agent ---
agnt = multimodal_mazes.AgentDQN(
    location=None,
    channels=[1, 1],
    sensor_noise_scale=NOISE,
    n_hidden_units=N,
    wm_flags=wm_flags,
    M_hh=M,
    W_hh_init=W_init,
    conserve_weight_sum=False,
)
agnt.generate_policy(maze=maze_train, n_steps=20, maze_test=maze_test)

# --- Test ---
test_results, _, memory = multimodal_mazes.test_dqn_agent(
    maze_test=maze_test,
    agnt=agnt,
    exp_config=exp_config,
    noises=[NOISE],
)
fitness    = test_results[0]
memory_sum = np.nansum(memory[0])

# --- Extract trained W_hh ---
W_trained = agnt.hidden_to_hidden.weight.detach().numpy()

print(f"  h_init={h_init:.3f} | "
      f"fitness={fitness:.3f} | memory_sum={memory_sum:.3f}")

# --- Save ---
result = {
    "deg_type":           deg_type,
    "degree_sequence":    deg,
    "job_id":             job_id,
    "cue_sparsity":       CS,
    "noise":              NOISE,
    "task":               TASK,
    "heterogeneity_init": h_init,
    "fitness":            fitness,
    "memory_sum":         memory_sum,
    "memory_decay":       memory[0],
    "training_fitness":   agnt.training_fitness,
    "M_hh":               M,
    "W_init":             W_init,
    "W_trained":          W_trained,
}

out_dir = f"Results/results_degseq_{run_version}"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/result_{deg_type}_{job_id}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(result, f)
print(f"Saved {out_path}")
