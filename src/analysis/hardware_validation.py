# src/analysis/hardware_validation.py
"""
Hardware validation analysis helpers:
 - js_divergence between two probability maps
 - quantum_fidelity for density matrices (requires scipy)
 - error_mitigation_factor
 - statistical tests (paired t-test, Cohen's d, bootstrap CI)
 - helpers to align bitstring distributions and normalize counts
 - simple plotting helpers (matplotlib)
"""

from typing import Dict, Tuple, List, Sequence, Any
import numpy as np
from scipy.stats import entropy, ttest_rel
from scipy.linalg import sqrtm
import math
import random
import matplotlib.pyplot as plt

# -----------------------
# Basic distribution helpers
# -----------------------
def align_and_normalize_counts(a: Dict[str, int], b: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two count dicts on the union of keys and return normalized probability vectors (same order).
    """
    keys = sorted(set(a.keys()) | set(b.keys()))
    ca = np.array([a.get(k, 0) for k in keys], dtype=float)
    cb = np.array([b.get(k, 0) for k in keys], dtype=float)
    # If total counts are zero, return zeros
    ta = ca.sum() if ca.sum() > 0 else 1.0
    tb = cb.sum() if cb.sum() > 0 else 1.0
    pa = ca / ta
    pb = cb / tb
    return pa, pb

def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    return float(np.sum(p * np.log(p / q)))

def js_divergence_from_counts(a: Dict[str, int], b: Dict[str, int]) -> float:
    pa, pb = align_and_normalize_counts(a, b)
    m = 0.5 * (pa + pb)
    return 0.5 * (kl_div(pa, m) + kl_div(pb, m))

# -----------------------
# Quantum fidelity (density-matrix)
# -----------------------
def quantum_fidelity(rho_sim: np.ndarray, rho_hw: np.ndarray) -> float:
    """
    Uhlmann fidelity for density matrices:
      F(rho, sigma) = (Tr sqrt( sqrt(rho) sigma sqrt(rho) ))^2
    Accepts small numerical tolerances. Returns float in [0,1].
    """
    # Cast to complex
    rho_sim = np.asarray(rho_sim, dtype=complex)
    rho_hw = np.asarray(rho_hw, dtype=complex)
    # Ensure Hermitian (symmetrize)
    rho_sim = 0.5 * (rho_sim + rho_sim.conj().T)
    rho_hw = 0.5 * (rho_hw + rho_hw.conj().T)
    # Compute sqrtm and fidelity
    s = sqrtm(rho_sim)
    inner = s @ rho_hw @ s
    root = sqrtm(inner)
    tr = np.trace(root)
    # numerical noise -> real part
    val = np.real(tr)
    fidelity = float(np.clip(val, 0.0, None))**2
    # Bound to [0,1]
    return min(1.0, fidelity)

# -----------------------
# Error mitigation effectiveness
# -----------------------
def error_mitigation_factor(ideal_prob: Dict[str, float], mitigated_prob: Dict[str, float], raw_prob: Dict[str, float]) -> float:
    """
    Returns ratio describing mitigation improvement toward ideal:
      factor = (dist(ideal, mitigated) ) / (dist(ideal, raw))
    If factor < 1 -> mitigated is closer to ideal than raw.
    Here we return relative improvement in JS divergence:
      improvement = (raw_js - mitigated_js) / raw_js
    """
    # align vectors
    ia = {k: ideal_prob.get(k, 0.0) for k in set(ideal_prob) | set(mitigated_prob) | set(raw_prob)}
    ma = {k: mitigated_prob.get(k, 0.0) for k in ia.keys()}
    ra = {k: raw_prob.get(k, 0.0) for k in ia.keys()}

    # Convert to arrays
    p_ideal = np.array([ia[k] for k in ia], dtype=float)
    p_mit = np.array([ma[k] for k in ia], dtype=float)
    p_raw = np.array([ra[k] for k in ia], dtype=float)
    # normalize
    if p_ideal.sum() > 0: p_ideal /= p_ideal.sum()
    if p_mit.sum() > 0: p_mit /= p_mit.sum()
    if p_raw.sum() > 0: p_raw /= p_raw.sum()

    raw_js = 0.5 * (kl_div(p_ideal, 0.5*(p_ideal+p_raw)) + kl_div(p_raw, 0.5*(p_ideal+p_raw)))
    mit_js = 0.5 * (kl_div(p_ideal, 0.5*(p_ideal+p_mit)) + kl_div(p_mit, 0.5*(p_ideal+p_mit)))
    if raw_js <= 0:
        return float('nan')
    improvement = (raw_js - mit_js) / raw_js
    return float(improvement)

# -----------------------
# Statistical tests & bootstrap
# -----------------------
def paired_ttest(quantum_scores: Sequence[float], classical_scores: Sequence[float]) -> Dict[str, float]:
    t_stat, p_value = ttest_rel(quantum_scores, classical_scores)
    mean_diff = float(np.mean(np.array(quantum_scores) - np.array(classical_scores)))
    # pooled std for Cohen's d
    diffs = np.array(quantum_scores) - np.array(classical_scores)
    pooled_sd = float(np.std(diffs, ddof=1))
    cohen_d = mean_diff / (pooled_sd if pooled_sd > 0 else 1e-8)
    return {"t_stat": float(t_stat), "p_value": float(p_value), "mean_diff": mean_diff, "cohen_d": cohen_d}

def bootstrap_ci(differences: Sequence[float], n_bootstrap: int = 10000, alpha: float = 0.05, rng: Any = None) -> Tuple[float,float]:
    """
    Non-parametric bootstrap CI (percentile).
    differences: array of paired differences (quantum - classical)
    """
    arr = np.array(differences)
    rng = np.random.default_rng() if rng is None else rng
    n = arr.size
    boots = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boots[i] = sample.mean()
    lower = float(np.percentile(boots, 100*alpha/2))
    upper = float(np.percentile(boots, 100*(1-alpha/2)))
    return (lower, upper)

# -----------------------
# Plot helpers
# -----------------------
def plot_bitstring_distributions(hw_counts: Dict[str,int], sim_counts: Dict[str,int], top_k: int = 10, ax=None, title: str = None):
    pa, pb = align_and_normalize_counts(hw_counts, sim_counts)
    keys = sorted(set(hw_counts.keys()) | set(sim_counts.keys()))
    # show top_k keys by avg probability
    avg = (pa + pb) / 2
    order = np.argsort(avg)[::-1][:top_k]
    labels = [keys[i] for i in order]
    hw_vals = pa[order]
    sim_vals = pb[order]
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, top_k*0.5), 4))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, sim_vals, width, label="sim")
    ax.bar(x + width/2, hw_vals, width, label="hardware")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, fontsize=9)
    ax.set_ylabel("Probability")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax

def plot_js_over_runs(js_values: Sequence[float], ax=None, title: str = "JS divergence across runs"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(js_values, marker='o')
    ax.set_ylabel("Jensen-Shannon")
    ax.set_xlabel("run index")
    ax.set_title(title)
    return ax

# -----------------------
# Utility: convert counts -> probability dict (float)
# -----------------------
def counts_to_prob_dict(counts: Dict[str,int]) -> Dict[str,float]:
    tot = sum(counts.values()) if counts else 0
    if tot == 0:
        return {k: 0.0 for k in counts}
    return {k: float(v)/tot for k,v in counts.items()}
