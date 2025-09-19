# src/analyze_hw_vs_sim_stats.py
"""
Analyze hw_vs_sim results: summary stats, bootstrap CI, and quick plots.

Usage:
    export PYTHONPATH="$PWD"
    python src/analyze_hw_vs_sim_stats.py --csv reports/hw_vs_sim_results.csv --out reports/hw_vs_sim_analysis
"""
import argparse
import os
import json
import math
import numpy as np
import csv
import logging
from typing import List, Dict
import matplotlib.pyplot as plt
from statistics import mean, stdev

LOG = logging.getLogger("analyze_hw_vs_sim")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def read_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # cast numeric fields
            for k in ("js_div", "kl_avg", "l1", "l2", "classical_fidelity"):
                if k in r:
                    try:
                        r[k] = float(r[k]) if r[k] != "" else float("nan")
                    except Exception:
                        r[k] = float("nan")
            rows.append(r)
    return rows

def bootstrap_ci(data: List[float], n_bootstrap: int = 10000, alpha: float = 0.05):
    data = np.array([x for x in data if not math.isnan(x)])
    if data.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(0)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=data.size, replace=True)
        means.append(sample.mean())
    lo = np.percentile(means, 100 * (alpha / 2))
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)

def save_json(path, obj):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="reports/hw_vs_sim_results.csv")
    p.add_argument("--out", type=str, default="reports/hw_vs_sim_analysis")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    rows = read_csv(args.csv)
    if not rows:
        LOG.error("No rows read from %s", args.csv)
        return

    metrics = ["js_div", "kl_avg", "l1", "l2", "classical_fidelity"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in rows if not math.isnan(r[m])]
        if len(vals) == 0:
            summary[m] = {"n": 0, "mean": None, "std": None, "ci95": (None, None)}
            continue
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci_lo, ci_hi = bootstrap_ci(vals, n_bootstrap=5000)
        summary[m] = {"n": len(vals), "mean": mu, "std": sd, "ci95": (ci_lo, ci_hi)}
        LOG.info("%s: n=%d mean=%.6f std=%.6f ci95=(%.6f, %.6f)", m, len(vals), mu, sd, ci_lo, ci_hi)

    # Save summary JSON
    save_json(os.path.join(args.out, "summary.json"), summary)

    # Create boxplots / distribution plots
    for m in metrics:
        vals = [r[m] for r in rows if not math.isnan(r[m])]
        if not vals:
            continue
        plt.figure(figsize=(6,4))
        plt.boxplot(vals, vert=True)
        plt.title(f"{m} (n={len(vals)})")
        plt.ylabel(m)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(args.out, f"box_{m}.png"), dpi=200)
        plt.close()

        # histogram
        plt.figure(figsize=(6,4))
        plt.hist(vals, bins=min(20, max(4, len(vals))), alpha=0.8)
        plt.title(f"{m} histogram")
        plt.savefig(os.path.join(args.out, f"hist_{m}.png"), dpi=200)
        plt.close()

    # If simulator was missing (sim_top_probs blank or '[]'), report
    sim_missing = any((r.get("sim_top_probs") in ("", "null", "[]", None)) for r in rows)
    if sim_missing:
        LOG.warning("Some simulator outputs appear missing. Install qiskit-aer to run simulator comparisons locally.")
        # print debug json paths from CSV rows if present
        debug_files = []
        for r in rows:
            rid = r.get("run_id")
            pth = os.path.join("reports", f"hw_vs_sim_debug_{rid}.json")
            if os.path.exists(pth):
                debug_files.append(pth)
        if debug_files:
            LOG.info("Found debug JSONs: %s", debug_files)

    LOG.info("Analysis written to %s", args.out)
    print("Done. Summary saved to", os.path.join(args.out, "summary.json"))

if __name__ == "__main__":
    main()
