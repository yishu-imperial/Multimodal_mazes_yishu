# merge_reciprocity_results.py
#
# Merge all individual pkl files from results_reciprocity/ into summary pkl files.
# Supports --task M or --task Msc.
#
# Usage:
#   python3 merge_reciprocity_results.py --task M
#   python3 merge_reciprocity_results.py --task Msc

import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="M")
args = parser.parse_args()
task = args.task

results_dir = "results_reciprocity"
files = sorted(os.listdir(results_dir))

for suffix in ["conserve", "no_conserve"]:
    results = []
    for fname in files:
        if fname.endswith(".pkl") and fname.startswith(f"result_{suffix}_{task}_"):
            with open(os.path.join(results_dir, fname), "rb") as f:
                result = pickle.load(f)
            results.append(result)
    out_path = f"results_reciprocity_{task}_{suffix}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"{suffix}: {len(results)} results → {out_path}")
    for r in results:
        print(f"  {r['graph_type']} | h_init={r['heterogeneity_init']:.3f} | fitness={r['fitness']:.3f} | memory_sum={r['memory_sum']:.3f}")
