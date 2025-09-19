# src/plot_hybrid_results.py
"""
Visualize hybrid classical vs quantum results.
Usage:
    export PYTHONPATH="$PWD"
    python -m src.plot_hybrid_results
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_CSV = "reports/hybrid_batch_results.csv"
OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(RESULTS_CSV)

    # --- Bar Plot: Classical vs Quantum per protein ---
    plt.figure(figsize=(12,6))
    x = range(len(df))
    plt.bar([i-0.2 for i in x], df["classical_score"], width=0.4, label="Classical")
    plt.bar([i+0.2 for i in x], df["quantum_score"], width=0.4, label="Quantum")
    plt.xticks(x, [f.replace(".npz","") for f in df["file"]], rotation=45, ha="right")
    plt.ylabel("Hydrophobicity Score (Kyte-Doolittle)")
    plt.title("Classical vs Quantum Protein Variants")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hybrid_scores_bar.png", dpi=300)
    plt.close()

    # --- Histogram of Δ Scores (Quantum - Classical) ---
    df["delta"] = df["quantum_score"] - df["classical_score"]
    plt.figure(figsize=(8,6))
    plt.hist(df["delta"], bins=10, color="purple", alpha=0.7)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Δ Score (Quantum - Classical)")
    plt.ylabel("Count")
    plt.title("Quantum Impact on Hydrophobicity")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hybrid_score_deltas.png", dpi=300)
    plt.close()

    # --- Save delta summary table ---
    df.to_csv(OUT_DIR / "hybrid_results_with_delta.csv", index=False)
    print(f"✅ Saved plots and updated CSV in {OUT_DIR}/")

if __name__ == "__main__":
    main()
