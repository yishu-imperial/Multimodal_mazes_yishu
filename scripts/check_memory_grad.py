# check_memory_grad.py
#
# Sanity check: do gradients actually flow through the delay-line
# memory update `new_mem = torch.cat([new_input, new_hidden, mem[:-slot]])`?
#
# If gradients could NOT flow through the memory, the readout W would never be
# trained -> the memory would look "useless" purely because of a bug, not because
# memory doesn't help. This script rules that out.
#
# Method: set use_B=False so the ONLY channel carrying information across timesteps
# is the memory buffer. Run several steps, put a loss on the LAST step's output,
# backprop, and check that W (memory readout) and A (input->hidden) get nonzero
# gradients. Nonzero => gradients propagate across time THROUGH the memory (the cat).
#
# NOTE on the loss: the agent's `output` is a Softmax, so `output.sum()` is always 1
# (a constant) -> its gradient is 0. That is NOT a broken gradient; it just means you
# must use a single element (like the DQN target on q_values[action]), not the sum.
#
# Usage: python3 scripts/check_memory_grad.py

import numpy as np
import torch
import multimodal_mazes

torch.manual_seed(0)
np.random.seed(0)

N, T, STEPS = 6, 4, 3

# use_B=False: no hidden->hidden recurrence, so the only cross-step path is the memory buffer.
agnt = multimodal_mazes.AgentDQNMemory(
    location=None, channels=[1, 1], sensor_noise_scale=0.05,
    n_hidden_units=N, T=T, use_B=False)

for p in agnt.parameters():
    p.grad = None

prev_i = torch.zeros(agnt.n_input_units)
hid = torch.zeros(N)
prev_o = torch.zeros(agnt.n_output_units)
mem = torch.zeros(agnt.mem_size)

# Roll out STEPS steps, threading the memory (this builds the graph across time).
out = None
for t in range(STEPS):
    agnt.channel_inputs = np.ones_like(agnt.channel_inputs) * (t + 1)
    out, prev_i, hid, prev_o, mem = agnt.forward(prev_i, hid, prev_o, mem)

# Loss on ONE element of the LAST step's output (varies; NOT the softmax sum which is constant 1).
loss = out[0]
loss.backward()


def report(name, w):
    g = w.grad
    val = 0.0 if g is None else g.abs().sum().item()
    return f"  {name:<22}: |grad| = {val:.5g}"


print(f"use_B=False (only cross-step channel = memory buffer), {STEPS} steps, loss = out[0]:")
print(report("W (memory_read)", agnt.memory_read.weight))
print(report("A (input_to_hidden)", agnt.input_to_hidden.weight))
print(report("C (hidden_to_output)", agnt.hidden_to_output.weight))

w_ok = agnt.memory_read.weight.grad is not None and agnt.memory_read.weight.grad.abs().sum() > 0
a_ok = agnt.input_to_hidden.weight.grad is not None and agnt.input_to_hidden.weight.grad.abs().sum() > 0
print()
if w_ok and a_ok:
    print("PASS: gradients propagate across timesteps THROUGH the memory buffer (the cat).")
    print("      -> the memory readout W is trainable; underperformance is not a gradient bug.")
else:
    print("FAIL: gradient does NOT reach the memory readout -> a real gradient-flow bug.")

# Extra: the softmax-sum pitfall, shown explicitly.
# Why we care about the softmax: the agent's forward() ends with
# `output = torch.nn.Softmax(dim=0)(output)`, so the returned output is a
# probability vector that ALWAYS sums to 1. Therefore loss=output.sum() is the
# constant 1 -> its gradient is 0, which looks (wrongly) like "gradients don't
# flow". The fix is to use a single element output[k] (as real training does with
# q_values[action]), which varies and gives a correct nonzero gradient. The block
# below prints both so the trap is documented, not just avoided.
agnt.zero_grad()
agnt.channel_inputs = np.ones_like(agnt.channel_inputs)
o = agnt.forward(prev_i.detach(), hid.detach(), prev_o.detach(), mem.detach())[0]
gsum = torch.autograd.grad(o.sum(), agnt.memory_read.weight, retain_graph=True, allow_unused=True)[0]
gone = torch.autograd.grad(o[0], agnt.memory_read.weight, allow_unused=True)[0]
print("\nWhy output.sum() gives 0 (softmax sums to 1, constant):")
print(f"  |grad| from loss=output.sum() : {0.0 if gsum is None else gsum.abs().sum().item():.5g}  (misleading 0)")
print(f"  |grad| from loss=output[0]    : {0.0 if gone is None else gone.abs().sum().item():.5g}  (correct, nonzero)")
