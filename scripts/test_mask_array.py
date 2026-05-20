import numpy as np
import pickle
import os
import re
import multimodal_mazes

N_HIDDEN = 8
wm_flags_standard_recurrent = np.array([0, 1, 0, 0, 0, 0, 0])

structures = {
    1: ("Identity", np.eye(N_HIDDEN)),
    2: ("Full",     np.ones((N_HIDDEN, N_HIDDEN))),
    3: ("Zero",     np.zeros((N_HIDDEN, N_HIDDEN))),
}

job_index = int(os.environ["PBS_ARRAY_INDEX"])
name, M_hh = structures[job_index]

exp_config = {"channels": [1, 1], "n_steps": 20}

maze_train = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
maze_train.generate(number=400000)

maze_test = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
maze_test.generate(number=1000)

agnt = multimodal_mazes.AgentDQN(
    location=None,
    channels=[1, 1],
    sensor_noise_scale=0.05,
    n_hidden_units=N_HIDDEN,
    wm_flags=wm_flags_standard_recurrent,
    M_hh=M_hh,
)
agnt.generate_policy(maze=maze_train, n_steps=20)

results, input_sensitivity, memory = multimodal_mazes.test_dqn_agent(
    maze_test=maze_test,
    agnt=agnt,
    exp_config=exp_config,
    noises=[0.05],
)

agnt.results = results
agnt.input_sensitivity = input_sensitivity
agnt.memory = memory
agnt.mask_name = name

save_folder = "../results/" + re.sub(r"\[.*?\]", "", os.environ["PBS_JOBID"])
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, name + ".pickle")
with open(save_path, "wb") as f:
    pickle.dump(agnt, f)

print(f"Saved: {save_path} | Fitness: {results[0]:.4f}")
