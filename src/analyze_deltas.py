# src/analyze_deltas.py
"""
Analyze delta improvements between classical and quantum scores.
Usage: PYTHONPATH="$PWD" python -m src.analyze_deltas
"""

import pandas as pd
from pathlib import Path

IN_CSV = Path("reports/hybrid_results_with_delta.csv")
OUT_SUMMARY = Path("reports/delta_summary.txt")

def main():
    df = pd.read_csv(IN_CSV)

    # Basic stats
    mean_delta = df["delta"].mean()
    std_delta = df["delta"].std()
    improved = (df["delta"] > 0).sum()
    worsened = (df["delta"] < 0).sum()

    # Top 5 improvements
    top5 = df.sort_values("delta", ascending=False).head(5)

    # Save summary
    with open(OUT_SUMMARY, "w") as f:
        f.write("=== Quantum vs Classical Delta Analysis ===\n\n")
        f.write(f"Total proteins: {len(df)}\n")
        f.write(f"Mean Δ: {mean_delta:.4f} ± {std_delta:.4f}\n")
        f.write(f"Improved: {improved}, Worsened: {worsened}\n\n")
        f.write("Top 5 improvements:\n")
        for _, row in top5.iterrows():
            f.write(f"- {row['file']}: Δ={row['delta']:.4f}\n")

    print(f"✅ Summary saved to {OUT_SUMMARY}")
    print("\n=== Preview ===")
    print(open(OUT_SUMMARY).read())

if __name__ == "__main__":
    main()
