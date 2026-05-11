import numpy as np
import matplotlib.pyplot as plt
import multimodal_mazes

# --- 1. Configuration ---
N_HIDDEN = 8
wm_flags_standard_recurrent = np.array([0, 1, 0, 0, 0, 0, 0])

# --- 2. Define multiple M_hh structures ---
structures = {
    "Identity": np.eye(N_HIDDEN),
    "Full":     np.ones((N_HIDDEN, N_HIDDEN)),
    "Zero":     np.zeros((N_HIDDEN, N_HIDDEN)),
}

# --- 3. Define tasks ---
tasks = {
    # "M": {
    #     "maze_type": "General",
    #     "size": 9,
    #     "n_steps": 20,
    #     "n_train": 400000,
    #     "n_test": 1000,
    #     "wall_sparsity": 0.0,
    #     "cue_sparsity": 0.0,
    #     "gaps": None,
    # },
    "M_sw": {
        "maze_type": "General",
        "size": 9,
        "n_steps": 20,
        "n_train": 400000,
        "n_test": 1000,
        "wall_sparsity": 0.1,
        "cue_sparsity": 0.0,
        "gaps": None,
    },
    "M_sc": {
        "maze_type": "General",
        "size": 9,
        "n_steps": 20,
        "n_train": 400000,
        "n_test": 1000,
        "wall_sparsity": 0.0,
        "cue_sparsity": 0.1,
        "gaps": None,
    },
    "T_sc": {
        "maze_type": "Track",
        "size": 11,
        "n_steps": 6,
        "n_train": 200000,
        "n_test": 1000,
        "wall_sparsity": 0.0,
        "cue_sparsity": 0.0,
        "gaps": 1,
    },
}

# --- 4. Run each task ---
for task_name, task_config in tasks.items():
    print(f"\n===== Task: {task_name} =====")

    exp_config = {
        "channels": [1, 1],
        "n_steps": task_config["n_steps"],
    }

    # Maze setup
    if task_config["maze_type"] == "General":
        maze_train = multimodal_mazes.GeneralMaze(size=task_config["size"], n_channels=2)
        maze_train.generate(
            number=task_config["n_train"],
            wall_sparsity=task_config["wall_sparsity"],
            cue_sparsity=task_config["cue_sparsity"],
        )
        maze_test = multimodal_mazes.GeneralMaze(size=task_config["size"], n_channels=2)
        maze_test.generate(
            number=task_config["n_test"],
            wall_sparsity=task_config["wall_sparsity"],
            cue_sparsity=task_config["cue_sparsity"],
        )
    else:
        maze_train = multimodal_mazes.TrackMaze(size=task_config["size"], n_channels=2)
        maze_train.generate(number=task_config["n_train"], noise_scale=0.0, gaps=task_config["gaps"])
        maze_test = multimodal_mazes.TrackMaze(size=task_config["size"], n_channels=2)
        maze_test.generate(number=task_config["n_test"], noise_scale=0.0, gaps=task_config["gaps"])

    # M_hh structure loop
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
    
        agnt.generate_policy(maze=maze_train, n_steps=task_config["n_steps"])

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

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    lags = np.arange(task_config["n_steps"])

    for name, res in all_results.items():
        axes[0].plot(lags, res["memory"], label=name)
        axes[1].plot(res["ys"], label=name)

    axes[0].set_xlabel("Temporal Lag (steps)")
    axes[0].set_ylabel("Normalised Memory")
    axes[0].set_title(f"Memory Decay — {task_name}")
    axes[0].legend()

    axes[1].set_xlabel("Timestep (all trials concatenated)")
    axes[1].set_ylabel("Jacobian Frobenius Norm")
    axes[1].set_title(f"Input Sensitivity — {task_name}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"comparison_{task_name}.png", dpi=150)
    plt.close()
    print(f"Saved comparison_{task_name}.png")
