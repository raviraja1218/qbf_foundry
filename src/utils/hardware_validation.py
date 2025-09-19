# src/utils/hardware_validation.py
"""
Hardware validation helpers: distance metrics, fidelity approximations,
safe extraction & normalization utilities, plotting helpers.
"""
from typing import Dict, Tuple, Optional, Sequence
import numpy as np
from math import log2
import json
import os
import logging

LOG = logging.getLogger("hw_val")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- probability utilities ---
def safe_normalize_counts(counts: Dict[str, int], shots: Optional[int] = None) -> Dict[str, float]:
    """Return probability mapping from counts; if shots provided use it, else sum counts."""
    if not counts:
        return {}
    total = shots if (shots is not None) else sum(counts.values())
    if total <= 0:
        total = 1
    probs = {k: float(v) / float(total) for k, v in counts.items()}
    return probs

def align_distributions(a: Dict[str, float], b: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given two probability maps over bitstrings, return aligned probability arrays and the union keys list."""
    keys = sorted(set(a.keys()).union(b.keys()))
    pa = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    pb = np.array([b.get(k, 0.0) for k in keys], dtype=float)
    return pa, pb, np.array(keys, dtype=object)

# --- divergences & distances ---
def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL divergence D(p||q) for discrete distributions (base e). Adds tiny eps for stability."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    return float(np.sum(p * np.log(p / q)))

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(p - q)))

def l2_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.sum((p - q) ** 2)))

# --- simple classical "fidelity" surrogate ---
def classical_state_fidelity(p: np.ndarray, q: np.ndarray) -> float:
    """
    A classical surrogate for fidelity between diagonal density matrices:
    F = (sum_sqrt(p_i * q_i))**2
    This equals fidelity if p and q are eigenvalue vectors of diagonal states.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(np.sqrt(p * q)) ** 2)

# --- result saving helpers ---
def dump_json(path: str, obj):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

# --- simple plotting helpers (matplotlib) ---
def plot_histograms(pa: np.ndarray, pb: np.ndarray, keys: Sequence[str], title: str, savepath: str, top_k: int = 40):
    import matplotlib.pyplot as plt
    # show top_k bitstrings by combined probability
    combined = pa + pb
    idx = np.argsort(-combined)[:top_k]
    labels = [keys[i] for i in idx]
    a_vals = pa[idx]
    b_vals = pb[idx]
    x = np.arange(len(idx))
    width = 0.4
    plt.figure(figsize=(max(6, len(idx) * 0.15), 4))
    plt.bar(x - width/2, a_vals, width, label='Simulator')
    plt.bar(x + width/2, b_vals, width, label='Hardware')
    plt.xticks(x, labels, rotation=90, fontsize=6)
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    d = os.path.dirname(savepath)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    plt.savefig(savepath, dpi=200)
    plt.close()

def save_result_summary_csv(rows, csv_path):
    import csv
    d = os.path.dirname(csv_path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    keys = [
        "run_id", "seed", "shots", "backend", "job_id",
        "js_div", "kl_avg", "l1", "l2", "classical_fidelity",
        "sim_top_probs", "hw_top_probs"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})
