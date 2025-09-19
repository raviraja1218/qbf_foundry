# src/compare_mitigated_vs_raw.py
"""
Compare raw vs mitigated counts from reports/mitigated_hw_runs.
Writes:
  - reports/mitigated_hw_runs/metrics_per_run.csv
  - reports/mitigated_hw_runs/metrics_summary.json
  - reports/mitigated_hw_runs/plots/summary_violin.png
Usage:
  python src/compare_mitigated_vs_raw.py --dir reports/mitigated_hw_runs
"""
import os
import json
import argparse
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, ttest_rel, wilcoxon

def dict_to_probvec(counts: dict, ordered_keys=None, shots: int = None):
    if counts is None:
        return [], []
    keys = sorted(counts.keys())
    if ordered_keys is not None:
        keys = ordered_keys
    vec = np.array([counts.get(k, 0) for k in keys], dtype=float)
    if shots is None:
        shots = vec.sum() if vec.sum() > 0 else 1.0
    prob = vec / shots
    return keys, prob

def classical_fidelity(p, q):
    # Bhattacharyya-like classical fidelity: sum sqrt(p_i q_i)
    p = np.asarray(p); q = np.asarray(q)
    return float(np.sum(np.sqrt(p * q)))

def l1_l2(p, q):
    p = np.asarray(p); q = np.asarray(q)
    return float(np.sum(np.abs(p - q))), float(np.linalg.norm(p - q))

def safe_entropy(p, q):
    # KL(p||q) with small regularization
    eps = 1e-12
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    return float(np.sum(p * np.log(p / q)))

def load_debug_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="reports/mitigated_hw_runs")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    ddir = args.dir
    if not os.path.isdir(ddir):
        raise SystemExit(f"Directory not found: {ddir}")

    # find summary CSV and debug JSONs
    summary_csv = os.path.join(ddir, "summary.csv")
    if not os.path.exists(summary_csv):
        raise SystemExit("Missing summary.csv in " + ddir)

    # read rows
    rows = []
    with open(summary_csv, "r") as f:
        r = csv.DictReader(f)
        for rr in r:
            rows.append(rr)

    per_run = []
    # find all debug jsons and compute metrics
    for row in rows:
        dbg = row.get("debug_json")
        if not dbg:
            continue
        if not os.path.exists(dbg):
            print("Warning: debug json missing:", dbg)
            continue
        j = load_debug_json(dbg)
        raw = j.get("raw_counts", {}) or {}
        mit = j.get("mitigated_counts", {}) or {}
        shots = int(j.get("shots", row.get("shots") or 0) or 0)
        # unify key space (all bitstrings present in union)
        all_keys = sorted(set(list(raw.keys()) + list(mit.keys())))
        _, p_raw = dict_to_probvec(raw, ordered_keys=all_keys, shots=shots or sum(raw.values()) or 1)
        _, p_mit = dict_to_probvec(mit, ordered_keys=all_keys, shots=shots or sum(mit.values()) or 1)
        # metrics
        try:
            js = float(jensenshannon(p_raw, p_mit, base=2.0))
        except Exception:
            js = None
        try:
            kl = safe_entropy(p_raw, p_mit)
        except Exception:
            kl = None
        l1, l2 = l1_l2(p_raw, p_mit)
        fid = classical_fidelity(p_raw, p_mit)
        per_run.append({
            "run": j.get("run"),
            "job_id": j.get("job_id"),
            "shots": shots,
            "js_div": js,
            "kl": kl,
            "l1": l1,
            "l2": l2,
            "classical_fidelity": fid,
            "raw_counts_len": len(raw),
            "mitigated_counts_len": len(mit),
            "debug_json": dbg
        })

    # write per-run CSV
    out_csv = os.path.join(ddir, "metrics_per_run.csv")
    keys = ["run","job_id","shots","js_div","kl","l1","l2","classical_fidelity","raw_counts_len","mitigated_counts_len","debug_json"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in per_run:
            w.writerow(r)
    print("Wrote per-run metrics to", out_csv)

    # summary statistics
    metrics = {}
    def stats_from_list(name):
        vals = [x[name] for x in per_run if x.get(name) is not None]
        if not vals:
            return {"n": 0, "mean": None}
        arr = np.array(vals, dtype=float)
        return {"n": len(arr), "mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1) if len(arr)>1 else 0.0),
                "min": float(np.min(arr)), "max": float(np.max(arr))}

    for nm in ("js_div","kl","l1","l2","classical_fidelity"):
        metrics[nm] = stats_from_list(nm)

    out_summary = os.path.join(ddir, "metrics_summary.json")
    with open(out_summary, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Wrote summary JSON to", out_summary)

    # simple violin/box plot for chosen metrics
    os.makedirs(os.path.join(ddir, "plots"), exist_ok=True)
    metric_for_plot = "classical_fidelity"
    vals = [x[metric_for_plot] for x in per_run if x.get(metric_for_plot) is not None]
    if vals:
        plt.figure(figsize=(4,4))
        plt.violinplot(vals, showmeans=True)
        plt.title(f"{metric_for_plot} across runs (n={len(vals)})")
        plt.ylabel(metric_for_plot)
        plt.savefig(os.path.join(ddir, "plots", f"{metric_for_plot}_violin.png"))
        plt.close()
        print("Wrote violin plot for", metric_for_plot)

    # Paired t-test between raw vs mitigated classical fidelity is not directly possible here (we only have fidelities computed per run),
    # but we can perform a simple t-test on the fidelity values if you have two arrays (e.g. fidelity_vs_simulator and fidelity_vs_raw).
    # For now return per_run list and metrics.
    print("Done.")

if __name__ == "__main__":
    main()
