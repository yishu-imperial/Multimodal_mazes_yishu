# check_jacobian_zeros.py
#
# most effective-Jacobian eigenvalues are 0 (effectively feedforward).
# Question: Is that set by the MASK (connectivity graph) or does it EMERGE during training?
#
# How the effective Jacobian relates to the masked weight matrix
# --------------------------------------------------------------
# Recurrent update (with ReLU):
#     h_t = ReLU( W_masked · h_{t-1} + W_in · x_t ),   W_masked = W_hh ⊙ M
# Jacobian d h_t / d h_{t-1} by the chain rule:
#     let z = W_masked · h_{t-1} + W_in x_t  (pre-activation)
#     d h_t/dz       = diag(relu'(z)) = G   (0/1 diagonal: 1 if z_i>0 (active), else 0)
#     dz/d h_{t-1}   = W_masked              (the input term W_in x_t does not depend on h_{t-1})
#     => J = G · W_masked                    (ReLU's derivative G is the ONLY difference from W_masked)
# Consequences for zero eigenvalues:
#   - No ReLU / all neurons active (G=I):  J = W_masked  -> SAME eigenvalues, SAME zeros.
#   - With gating (G != I):  J = G·W_masked zeros the inactive neurons' rows -> J has >= as many zeros
#     (gating can only cut cycles, never add them -> more 'feedforward' -> more zero eigenvalues).
#   - G_t changes every timestep (state-dependent), so J is a FAMILY of matrices; we read its |λ|
#     per step from the stored eff_lambdas_per_state.
#
# Decompose the fraction of ~0 eigenvalues across three weight matrices (per agent, per structure):
#   1. M_hh            : binary connectivity (mask).
#   2. W_init  ⊙ M     : random weights on the mask (pre-training).
#   3. W_trained ⊙ M   : trained weights on the mask (no gating).
# Columns equal -> MASK sets the zeros (weight values/training don't change the count).
# (The gated effective Jacobian J = G·W_masked is a different operator; not compared here.)
#
# Usage: python3 scripts/check_jacobian_zeros.py --cond cs0_n010

import glob
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--cond", default="cs0_n010")
parser.add_argument("--thr", type=float, default=1e-6, help="|λ| below this counts as 'zero'")
args = parser.parse_args()

STRUCTS = ["regular", "d4", "medium", "high", "d8"]
VER = "v6_20260622_degseq"


def frac_zero(A):
    """fraction of eigenvalues of square matrix A with |λ| < thr."""
    w = np.abs(np.linalg.eigvals(np.asarray(A, float)))
    return np.mean(w < args.thr)


print(f"cond={args.cond}  thr={args.thr}  (fraction of |λ| < thr = 'zero')\n")
print(f"{'struct':>8} | {'M_hh':>7} {'W_init⊙M':>9} {'W_train⊙M':>10}")
print("-" * 42)

for s in STRUCTS:
    fM, fI, fT = [], [], []
    for f in sorted(glob.glob(f"Results/results_degseq_{VER}_{s}_{args.cond}/*.pkl")):
        r = pickle.load(open(f, "rb"))
        M = np.asarray(r["M_hh"], float)
        fM.append(frac_zero(M))                       # zeros of the mask graph
        fI.append(frac_zero(r["W_init"]))             # zeros of W_init ⊙ M  (random weights)
        fT.append(frac_zero(r["W_trained"]))          # zeros of W_trained ⊙ M (trained weights)
    print(f"{s:>8} | {np.mean(fM):>7.2f} {np.mean(fI):>9.2f} {np.mean(fT):>10.2f}")

print("\nReading: three columns equal -> MASK sets the zeros (weight values / training don't change the count).")
