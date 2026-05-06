import multimodal_mazes
import matplotlib.pyplot as plt

maze = multimodal_mazes.GeneralMaze(size=33, n_channels=2)
maze.generate(number=10)

agnt = multimodal_mazes.AgentRuleBased(location=None, channels=[1,1], policy="Linear fusion")

n = 0 # select a maze
time, path = multimodal_mazes.maze_trial(mz=maze.mazes[n], mz_start_loc=maze.start_locations[n], mz_goal_loc=maze.goal_locations[n], channels=[1,1], sensor_noise_scale=0.0, drop_connect_p=0.0, n_steps=100, agnt=agnt) 

multimodal_mazes.plot_path(path, mz=maze.mazes[n], mz_goal_loc=maze.goal_locations[n], n_steps=100, style="gradients")

plt.show()