import numpy as np
import matplotlib.pyplot as plt
import multimodal_mazes

# --- 1. Configuration ---
N_HIDDEN = 8
wm_flags_standard_recurrent = np.array([0, 1, 0, 0, 0, 0, 0])

exp_config = {
    "channels": [1, 1],
    "n_steps": 50,
}

# --- 2. Define multiple M_hh structures ---
structures = {
    "Identity": np.eye(N_HIDDEN),
    "Full":     np.ones((N_HIDDEN, N_HIDDEN)),
    "Zero":     np.zeros((N_HIDDEN, N_HIDDEN)),
}

# --- 3. Environment Setup ---
maze_train = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
maze_train.generate(number=1000)

maze_test = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
maze_test.generate(number=100)

# --- 4. Run each structure ---
all_results = {}

for name, M_hh in structures.items():
    print(f"\nTraining: {name}")

    agnt = multimodal_mazes.AgentDQN(
        location=None,
        channels=[1, 1],
        sensor_noise_scale=0.05,
        n_hidden_units=N_HIDDEN,
        wm_flags=wm_flags_standard_recurrent,
        M_hh=M_hh
    )

    agnt.generate_policy(maze=maze_train, n_steps=50)

    results, input_sensitivity, memory = multimodal_mazes.test_dqn_agent(
        maze_test=maze_test,
        agnt=agnt,
        exp_config=exp_config,
        noises=[0.05],
    )

    xs, ys = input_sensitivity[0]
    all_results[name] = {
        "fitness": results[0],
        "memory": memory[0],
        "ys": ys,
    }

    print(f"  Fitness: {results[0]:.4f} | Mean Input Sensitivity: {np.mean(ys):.4f}")

# --- 5. Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

lags = np.arange(exp_config["n_steps"])

for name, res in all_results.items():
    axes[0].plot(lags, res["memory"], label=name)
    axes[1].plot(res["ys"], label=name)

axes[0].set_xlabel("Temporal Lag (steps)")
axes[0].set_ylabel("Normalised Memory")
axes[0].set_title("Memory Decay")
axes[0].legend()

axes[1].set_xlabel("Timestep (all trials concatenated)")
axes[1].set_ylabel("Jacobian Frobenius Norm")
axes[1].set_title("Input Sensitivity per Timestep")
axes[1].legend()

plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
plt.show()