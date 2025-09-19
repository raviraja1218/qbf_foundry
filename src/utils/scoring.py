# src/utils/scoring.py
"""
Protein sequence scoring helpers.

Provides:
 - Kyte-Doolittle hydrophobicity scoring (higher = more hydrophobic).
 - A simple solubility proxy (more charged/polar residues increase solubility).
 - Combined score helper for experiments.

Sequence input: string using one-letter amino-acid codes. Unknowns ('?', '-') handled conservatively.
"""

from typing import Tuple
import numpy as np

# Kyte-Doolittle hydrophobicity scale (one-letter)
_KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5
}

# Residues considered polar/charged for solubility proxy
_POLAR_CHARGED = set(["D","E","K","R","H","S","T","N","Q","Y","C","W"])  # conservative

def _clean_seq(seq: str) -> str:
    if seq is None:
        return ""
    # remove whitespace, convert to uppercase
    return "".join([c.upper() for c in seq if not c.isspace()])

def score_sequence_hydrophobicity(seq: str) -> float:
    """
    Returns mean Kyte-Doolittle hydrophobicity for seq.
    Higher values => more hydrophobic.
    """
    s = _clean_seq(seq)
    if len(s) == 0:
        return 0.0
    vals = []
    for c in s:
        if c in _KD:
            vals.append(_KD[c])
        else:
            # unknown or gap: treat as neutral slightly hydrophilic
            vals.append(-0.5)
    return float(np.mean(vals))

def score_sequence_solubility_proxy(seq: str) -> float:
    """
    Heuristic solubility proxy:
     - fraction of polar/charged residues (higher => more soluble)
    """
    s = _clean_seq(seq)
    if len(s) == 0:
        return 0.0
    count = sum(1 for c in s if c in _POLAR_CHARGED)
    return float(count / len(s))

def score_sequence_combined(seq: str, weights: Tuple[float, float] = (0.7, 0.3)) -> float:
    """
    Combined score: weighted sum of normalized hydrophobicity and solubility proxy.
    We normalize KD to roughly [-1, +1] by dividing by 4.5 (max).
    By default we weight hydrophobicity heavier.
    """
    kd = score_sequence_hydrophobicity(seq)
    sol = score_sequence_solubility_proxy(seq)
    # normalize kd to approx [-1,1]
    kd_norm = kd / 4.5
    return float(weights[0] * kd_norm + weights[1] * sol)
