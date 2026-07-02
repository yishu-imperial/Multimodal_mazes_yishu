# DQN agent + fixed delay-line memory (a simplified linear SSM).
#
# Adds an explicit memory buffer m(t) that stores the last T steps of (x, h):
#     m(t)   = M_in x(t) + M_h h(t) + M_m m(t-1)          (M_* fixed / not learned)
#     h(t)   = A x(t) + B h(t-1) + W m(t-1)                (LINEAR, ReLU or not depends on relu_state)
#     y(t)   = C h(t)                                      (softmax readout)
# Fixed (constant): M_in, M_h (=I_n), M_m (shift).  Learned: A, B, W, C.
# See notes/delay_line_memory_ssm.md for the design + why it is a simplified SSM.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multimodal_mazes
from multimodal_mazes.agents.agent_dqn import AgentDQN


class AgentDQNMemory(AgentDQN):
    def __init__(self, location, channels, sensor_noise_scale, n_hidden_units, T,
                 use_B=True, nonlinear_readout=False, relu_state=False):
        """
        DQN agent with a fixed delay-line memory of length T.
        A/B/W/C are learned; the memory matrices M_in, M_h, M_m are fixed.
        use_B: if False, drop the hidden->hidden term (h = A x + W m) for stability.
        nonlinear_readout: if True, output = C * ReLU(D * h) (extra nonlinear layer in
               the OUTPUT path only). State (h, m) evolution stays linear (still an SSM).
        """
        self.use_B = bool(use_B)
        self.nonlinear_readout = bool(nonlinear_readout)
        self.relu_state = bool(relu_state)   # if True: h = ReLU(A x + B h + W m) (nonlinear state)
        wm_flags = np.array([0, 1, 0, 0, 0, 0, 0])   # hidden->hidden recurrence on (dense B)
        super().__init__(location=location, channels=channels,
                         sensor_noise_scale=sensor_noise_scale,
                         n_hidden_units=n_hidden_units, wm_flags=wm_flags)
        self.type = "AgentDQN"                       # so maze_trial resets prev_input/hidden/...
        self.T = int(T)
        d_in = self.n_input_units
        n = self.n_hidden_units
        slot = d_in + n                              # each slot stores [x ; h]
        self.slot_dim = slot
        self.mem_size = slot * self.T

        # Fixed memory matrices M_in, M_h, M_m (see notes/delay_line_memory_ssm.md)
        # are implemented efficiently in forward() via slicing/cat instead of dense
        # matmuls (M_m would be mem_size x mem_size = huge for large n). The op is
        # identical: write [x;h] into slot 0, shift the rest back one slot.

        # --- Learned memory readout W: (n x mem_size) ---
        self.memory_read = nn.Linear(self.mem_size, n, bias=False)
        nn.init.uniform_(self.memory_read.weight, a=-self.ks[1], b=self.ks[1])

        # Optional nonlinear readout head (output path only; state stays linear)
        if self.nonlinear_readout:
            self.readout_mlp = nn.Linear(n, n)
            nn.init.uniform_(self.readout_mlp.weight, a=-self.ks[1], b=self.ks[1])

        # Current memory buffer (reset per episode)
        self.mem = torch.zeros(self.mem_size)

    def forward(self, prev_input, hidden, prev_output, mem, tensor_input=False):
        # Input at t
        if tensor_input:
            new_input = self.channel_inputs
        else:
            new_input = torch.from_numpy(self.channel_inputs.reshape(-1)).to(torch.float32)

        # Hidden update (LINEAR, no ReLU):  h(t) = A x(t) [+ B h(t-1)] + W m(t-1)
        new_hidden = self.input_to_hidden(new_input)          # A x
        if self.use_B:
            new_hidden = new_hidden + self.hidden_to_hidden(hidden)  # B h(t-1)  (dense)
        new_hidden = new_hidden + self.memory_read(mem)        # W m(t-1)
        if self.relu_state:
            new_hidden = torch.relu(new_hidden)                # nonlinear state (optional)

        # Memory update (efficient equivalent of M_in x + M_h h + M_m m):
        # slot 0 = [x ; h]; slots 1..T-1 = old slots 0..T-2 (shift), oldest dropped.
        new_mem = torch.cat([new_input, new_hidden, mem[:-self.slot_dim]])

        # Output (softmax over Q-values); optional nonlinear head (state stays linear)
        h_read = torch.relu(self.readout_mlp(new_hidden)) if self.nonlinear_readout else new_hidden
        output = self.hidden_to_output(h_read)
        output = output + torch.rand(len(output)) / 1000
        output = torch.nn.Softmax(dim=0)(output)
        return output, new_input, new_hidden, output, new_mem

    def policy(self):
        # Called once per step by maze_trial; threads memory via self.mem.
        with torch.no_grad():
            (self.outputs, self.prev_input, self.hidden,
             self.prev_output, self.mem) = self.forward(
                self.prev_input, self.hidden, self.prev_output, self.mem)

    def generate_policy(self, maze, n_steps, maze_test=None):
        """Deep Q-learning, threading the memory buffer m alongside hidden."""
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        gamma = 0.9
        epsilons = np.repeat(
            np.linspace(start=0.95, stop=0.25, num=10), repeats=len(maze.mazes) // 10)
        self.training_fitness = []

        for a, n in enumerate(np.random.permutation(len(maze.mazes))):
            if (a % (len(maze.mazes) // 100) == 0) and (maze_test is not None):
                with torch.no_grad():
                    fitness = multimodal_mazes.eval_fitness(
                        genome=None, config=None, channels=self.channels,
                        sensor_noise_scale=self.sensor_noise_scale, drop_connect_p=0.0,
                        maze=maze_test, n_steps=n_steps, agnt=self)
                    self.training_fitness.append(fitness)

            # Reset agent (including memory)
            prev_input = torch.zeros(self.n_input_units)
            hidden = torch.zeros(self.n_hidden_units)
            prev_output = torch.zeros(self.n_output_units)
            mem = torch.zeros(self.mem_size)
            self.location = np.copy(maze.start_locations[n])
            self.outputs = torch.zeros(self.n_output_units)
            loss = 0.0

            starting_reward = (
                maze.d_maps[n].max()
                - maze.d_maps[n][self.location[0], self.location[1]]) / maze.d_maps[n].max()

            for _ in range(n_steps):
                self.sense(maze.mazes[n])
                if torch.rand(1) < epsilons[a]:
                    action = torch.randint(low=0, high=self.n_output_units, size=(1,)).item()
                else:
                    with torch.no_grad():
                        q_values, _, _, _, _ = self.forward(prev_input, hidden, prev_output, mem)
                        action = torch.argmax(q_values).item()

                q_values, prev_input, hidden, prev_output, mem = self.forward(
                    prev_input, hidden, prev_output, mem)
                predicted = q_values[action]

                self.outputs *= 0.0
                self.outputs[action] = 1.0
                self.act(maze.mazes[n])

                reward = (
                    maze.d_maps[n].max()
                    - maze.d_maps[n][self.location[0], self.location[1]]) / maze.d_maps[n].max()
                reward = (reward - starting_reward) * 2

                self.sense(maze.mazes[n])
                with torch.no_grad():
                    next_q_values, _, _, _, _ = self.forward(prev_input, hidden, prev_output, mem)
                    target = reward + (gamma * torch.max(next_q_values)) - 0.1

                loss = loss + criterion(predicted, target)
                if np.array_equal(self.location, maze.goal_locations[n]):
                    break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
            optimizer.step()
