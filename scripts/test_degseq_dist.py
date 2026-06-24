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
#   2026-06-08 (v3): Robustness: test across a noise sweep (eval_fitness, fitness-only,
#                    no extra Jacobians) -> save test_noises, fitness_vs_noise,
#                    robustness_auc (AUC of fitness-vs-noise). Also save full agent
#                    state_dict ("agent_state") so future re-tests need no retraining.
#                    FIX: medium/high skew coefficients were swapped (0.4/0.35) so
#                    medium was MORE skewed than high. Now medium=0.35, high=0.4 →
#                    monotonic skew regular < medium < high < hub.
#                    NOTE: v1 (v1_20260607_degseq) data has medium/high LABELS swapped
#                    vs this corrected definition; not re-run.
#   2026-06-21 (v5): n=100 scale-up via a single continuous skew knob --alpha
#                    (rank power law on out-degree, total edges fixed, per-node floor).
#                    --n / --total / --floor parametrise size & budget (defaults keep
#                    the old n=10 named-structure path when --alpha is omitted).
#                    Store alpha / n_hidden / total_edges in the result dict.

import os
import sys
import copy
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
parser.add_argument("--alpha", type=float, default=None,
                    help="If set: n=N skew via rank power law r^(-alpha) on out-degree "
                         "(alpha=0 uniform, larger = more skew). Omit -> old named-structure path.")
parser.add_argument("--n", type=int, default=10, help="number of hidden units (network size)")
parser.add_argument("--total", type=int, default=18, help="total recurrent edges (fixed budget)")
parser.add_argument("--floor", type=int, default=0, help="(alpha path) guaranteed min out-degree")
parser.add_argument("--n_train_mazes", type=int, default=400000, help="# training episodes")
parser.add_argument("--n_test_mazes", type=int, default=1000, help="# test mazes")
parser.add_argument("--n_spectrum_mazes", type=int, default=100,
                    help="# test mazes used for the per-state effective |λ| distribution (bounds file size)")
args = parser.parse_args()

deg_type    = args.deg_type
run_version = args.version

job_id = os.environ.get("PBS_JOBID", "local")
print(f"Job {job_id} | deg_type={deg_type}")

# --- Configuration ---
N     = args.n
TOTAL = args.total    # fixed total edges (n=10 default 18 ~ p=0.2)
FLOOR = args.floor
ALPHA = args.alpha
CS    = args.cue_sparsity
NOISE = args.noise
TASK  = "M" if CS == 0.0 else "Msc"   # task label derived from cue sparsity
print(f"  n={N} | total={TOTAL} | alpha={ALPHA} | cue_sparsity={CS} | noise={NOISE} | task={TASK}")

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

    # d4: 2 hubs of out-degree 4 (between regular and medium; conc 44%)
    hub4 = min(max_deg, 4)
    rem4 = total - 2 * hub4
    rest4 = [0] * (n - 2)
    per4 = rem4 // (n - 2)
    lo4 = rem4 % (n - 2)
    for i in range(n - 2):
        rest4[i] = per4 + (1 if i < lo4 else 0)
    d4 = [hub4, hub4] + rest4
    if sum(d4) == total and all(0 <= d <= max_deg for d in d4):
        sequences['d4'] = sorted(d4, reverse=True)

    # d8: 2 hubs of out-degree 8 (between high and hub; conc 89%)
    hub8 = min(max_deg, 8)
    rem8 = total - 2 * hub8
    rest8 = [0] * (n - 2)
    for i in range(min(rem8, n - 2)):
        rest8[i] = 1
    d8 = [hub8, hub8] + rest8
    if sum(d8) == total and all(0 <= d <= max_deg for d in d8):
        sequences['d8'] = sorted(d8, reverse=True)

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

# --- Alpha-controlled out-degree sequence (n=100 scale-up) ---
# Floor every node, split surplus (total - floor*n) by rank weight r^(-alpha)
# (rank 1 = biggest hub); sum held EXACTLY = total. See RESULTS_NOTES
# "n=100 alpha out-degree design" / scripts/check_alpha_outdegree.py.
def alpha_outdegree_sequence(n, total, alpha, floor=2):
    if total < floor * n:
        raise ValueError(f"total={total} < floor*n={floor*n}: floor infeasible")
    extra = total - floor * n
    w = np.arange(1, n + 1) ** (-alpha)          # skew knob: 0 uniform, larger -> front ranks dominate
    w = w / w.sum()
    deg = floor + w * extra
    deg_int = np.floor(deg).astype(int)          # round down, then refill the remainder
    frac = deg - deg_int
    rem = int(round(total - deg_int.sum()))
    order = np.argsort(-frac)                     # largest chopped-off fraction first
    for k in range(rem):
        deg_int[order[k]] += 1
    deg_int = np.minimum(deg_int, n - 1)         # no self-loops -> cap at n-1
    short = int(total - deg_int.sum())
    j = 0
    while short > 0 and j < 1000 * n:            # re-place capped-hub edges on nodes with room
        idx = order[j % n]
        if deg_int[idx] < n - 1:
            deg_int[idx] += 1
            short -= 1
        j += 1
    return sorted(deg_int.tolist(), reverse=True)

# --- Generate degree sequence and select ---
if ALPHA is not None:                            # n=100 alpha path (deg_type is just a label, e.g. "a10")
    deg = alpha_outdegree_sequence(N, TOTAL, ALPHA, floor=FLOOR)
else:                                            # old n=10 named-structure path
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
maze_train.generate(number=args.n_train_mazes, cue_sparsity=CS)

maze_test = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
maze_test.generate(number=args.n_test_mazes, cue_sparsity=CS)

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

# --- Test (memory/sensitivity computed only at the training noise) ---
test_results, _, memory = multimodal_mazes.test_dqn_agent(
    maze_test=maze_test,
    agnt=agnt,
    exp_config=exp_config,
    noises=[NOISE],
)
fitness    = test_results[0]
memory_sum = np.nansum(memory[0])

# --- Robustness: fitness vs test-noise sweep (cheap, no Jacobians) ---
# Train at one noise, test at many -> fitness-vs-noise curve -> AUC.
# Uses eval_fitness directly (fitness only) to avoid the 7x Jacobian cost.
TEST_NOISES = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
fitness_vs_noise = np.array([
    multimodal_mazes.eval_fitness(
        genome=None, config=None, channels=exp_config["channels"],
        sensor_noise_scale=tn, drop_connect_p=0.0,
        maze=maze_test, n_steps=exp_config["n_steps"], agnt=agnt,
    )
    for tn in TEST_NOISES
])
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy>=2.0 renamed trapz->trapezoid
robustness_auc = float(trapz(fitness_vs_noise, TEST_NOISES))  # area under fitness-vs-noise

# --- Effective (Jacobian) recurrent spectrum along the test trajectory ---
# diag(relu')·(W_hh⊙M) per step via autodiff; |λ| sorted desc.
# effective_spectrum = mean over steps (length n_hidden; [0] = spectral radius).
# eff_lambdas_per_state = every |λ| at every step (n_states, n) over n_spectrum_mazes mazes,
#   so the |λ| DISTRIBUTION / spread can be studied (capped to bound file size).
_, all_states = multimodal_mazes.eval_fitness(
    genome=None, config=None, channels=exp_config["channels"],
    sensor_noise_scale=NOISE, drop_connect_p=0.0,
    maze=maze_test, n_steps=exp_config["n_steps"],
    agnt=copy.deepcopy(agnt), record_states=True,
)
eff = multimodal_mazes.calculate_dqn_effective_spectrum(
    all_states[:args.n_spectrum_mazes], agnt, return_distribution=True)
effective_spectrum   = eff["mean_spectrum"]
eff_lambdas_per_state = eff["lambdas_per_state"]

# --- Extract trained W_hh + full agent weights (so re-tests need no retraining) ---
W_trained = agnt.hidden_to_hidden.weight.detach().numpy()
agent_state = {k: v.detach().cpu().numpy() for k, v in agnt.state_dict().items()}

print(f"  h_init={h_init:.3f} | "
      f"fitness={fitness:.3f} | memory_sum={memory_sum:.3f} | "
      f"robustness_auc={robustness_auc:.3f}")

# --- Save ---
result = {
    "deg_type":           deg_type,
    "degree_sequence":    deg,
    "alpha":              ALPHA,
    "n_hidden":           N,
    "total_edges":        TOTAL,
    "job_id":             job_id,
    "cue_sparsity":       CS,
    "noise":              NOISE,
    "task":               TASK,
    "heterogeneity_init": h_init,
    "fitness":            fitness,
    "memory_sum":         memory_sum,
    "memory_decay":       memory[0],
    "training_fitness":   agnt.training_fitness,
    "test_noises":        TEST_NOISES,
    "fitness_vs_noise":   fitness_vs_noise,
    "robustness_auc":     robustness_auc,
    "effective_spectrum":     effective_spectrum,
    "eff_lambdas_per_state":  eff_lambdas_per_state,   # (n_states, n) every |λ| at every step
    "M_hh":               M,
    "W_init":             W_init,
    "W_trained":          W_trained,
    "agent_state":        agent_state,
}

out_dir = f"Results/results_degseq_{run_version}"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/result_{deg_type}_{job_id}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(result, f)
print(f"Saved {out_path}")
