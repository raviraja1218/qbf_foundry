# src/evaluate_bio.py
"""
Evaluate biological scores (hydrophobicity + solubility proxy) for
classical vs quantum sequences in reports/hybrid_batch_results.csv.

Usage:
    export PYTHONPATH="$PWD"
    python -m src.evaluate_bio
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.scoring import score_sequence_combined

REPORTS_DIR = Path("reports")
INPUT_CSV = REPORTS_DIR / "hybrid_batch_results.csv"
OUT_CSV = REPORTS_DIR / "hybrid_results_with_bio.csv"
SUMMARY_TXT = REPORTS_DIR / "bio_score_summary.txt"
HIST_PNG = REPORTS_DIR / "bio_delta_hist.png"

def safe_score(s):
    try:
        return score_sequence_combined(s)
    except Exception:
        return np.nan

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"{INPUT_CSV} not found. Run run_hybrid_batch first.")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Ensure classical_seq and quantum_seq columns exist
    if "classical_seq" not in df.columns or "quantum_seq" not in df.columns:
        raise ValueError("CSV missing 'classical_seq' or 'quantum_seq' columns")

    df["bio_classical"] = df["classical_seq"].fillna("").astype(str).map(safe_score)
    df["bio_quantum"] = df["quantum_seq"].fillna("").astype(str).map(safe_score)
    df["delta_bio"] = df["bio_quantum"] - df["bio_classical"]

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved with biological scores: {OUT_CSV}")

    # Summary
    mean_delta = df["delta_bio"].mean()
    std_delta = df["delta_bio"].std()
    improved = (df["delta_bio"] > 0).sum()
    worsened = (df["delta_bio"] < 0).sum()

    summary_text = (
        f"Biological score summary\n"
        f"Rows: {len(df)}\n"
        f"Mean delta (quantum - classical): {mean_delta:.4f} ± {std_delta:.4f}\n"
        f"Improved: {improved}, Worsened: {worsened}\n"
        f"\nTop improvements:\n"
    )

    top = df.sort_values("delta_bio", ascending=False).head(10)
    for _, r in top.iterrows():
        summary_text += f"- {r['file']}: Δ={r['delta_bio']:.4f} (classical={r['bio_classical']:.4f}, quantum={r['bio_quantum']:.4f})\n"

    SUMMARY_TXT.write_text(summary_text)
    print(f"Saved summary to {SUMMARY_TXT}")

    # Plot histogram
    plt.figure(figsize=(6,4))
    df["delta_bio"].hist(bins=25)
    plt.title("Distribution of biological score Δ (quantum - classical)")
    plt.xlabel("Δ bio score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(HIST_PNG)
    print(f"Saved histogram: {HIST_PNG}")

if __name__ == "__main__":
    main()
