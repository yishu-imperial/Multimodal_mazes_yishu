import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
results_dir = "results/"
mask_names  = ["Identity", "Full", "Zero"]
n_masks     = len(mask_names)
mask_map    = {"0": 0, "1": 1, "2": 2}
colors      = ["C0", "C1", "C2"]

# --- Load results ---
folders = sorted([f for f in os.listdir(results_dir) if f.startswith("test")])
n_runs  = len(folders)
n_lags  = 20
print(f"Found {n_runs} runs")

fitness  = np.zeros((n_masks, n_runs)) * np.nan
memories = np.zeros((n_masks, 1, n_lags, n_runs)) * np.nan

for a, folder in enumerate(folders):
    path = os.path.join(results_dir, folder)
    for fname in os.listdir(path):
        if fname.endswith(".pickle"):
            key = os.path.splitext(fname)[0]
            if key in mask_map:
                idx = mask_map[key]
                with open(os.path.join(path, fname), "rb") as f:
                    agnt = pickle.load(f)
                fitness[idx, a]        = agnt.results[0]
                memories[idx, 0, :, a] = agnt.memory[0]

print("Loaded. Fitness shape:", fitness.shape)
print("Memory shape:", memories.shape)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Plot 1: Fitness bar chart ---
ax = axes[0]
x      = np.arange(n_masks)
fit_m  = np.nanmedian(fitness, axis=1)
fit_lo = np.nanquantile(fitness, 0.25, axis=1)
fit_hi = np.nanquantile(fitness, 0.75, axis=1)

for i in range(n_masks):
    ax.bar(x[i], fit_m[i], color=colors[i], alpha=0.8, label=mask_names[i])
    ax.errorbar(x[i], fit_m[i],
                yerr=[[fit_m[i] - fit_lo[i]], [fit_hi[i] - fit_m[i]]],
                fmt="none", color="k", capsize=5, linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(mask_names)
ax.set_ylabel("Fitness")
ax.set_title("Performance (Fitness)")

# --- Plot 2: Memory decay per mask ---
ax = axes[1]
x_lags = range(n_lags)

for i in range(n_masks):
    l, m, u = np.nanquantile(memories[i, 0], [0.25, 0.5, 0.75], axis=1)
    ax.plot(x_lags, m[::-1], color=colors[i], label=mask_names[i])
    ax.fill_between(x_lags, l[::-1], u[::-1], color=colors[i], alpha=0.25, edgecolor=None)

ax.hlines(y=1.0, xmin=0, xmax=n_lags - 1, color="k", linestyles="--")
ax.set_xlabel("Time")
ax.set_ylabel("Input-output influence")
ax.set_title("Memory: past input influence")
ax.set_xticks(np.arange(n_lags)[::2])
ax.set_xticklabels(np.arange(n_lags)[::2][::-1] * -1)
ax.legend()

plt.tight_layout()
plt.savefig("mask_analysis.png", dpi=150)
plt.show()
print("Saved mask_analysis.png")

# --- Print summary ---
print("\n--- Fitness summary (median [Q1, Q3]) ---")
for i, name in enumerate(mask_names):
    print(f"  {name:10s}: {fit_m[i]:.4f} [{fit_lo[i]:.4f}, {fit_hi[i]:.4f}]")
