#!/usr/bin/env python3
"""
Inspect and extract results from reports/quantum_opt_batch_raw.npz

Outputs:
 - reports/quantum_opt_batch_summary_from_npz.csv  (file, classical_score, best_score, delta)
 - reports/quantum_opt_sequences.fasta            (FASTA: >file|classical and >file|quantum)
 - prints brief summary to stdout
"""
import os
from pathlib import Path
import numpy as np
import csv
import math
import sys

ROOT = Path(__file__).resolve().parents[2]  # repo root
IN_PATH = ROOT / "reports" / "quantum_opt_batch_raw.npz"
OUT_CSV = ROOT / "reports" / "quantum_opt_batch_summary_from_npz.csv"
OUT_FASTA = ROOT / "reports" / "quantum_opt_sequences.fasta"

# try to import your AA decoder; if not available use a fallback
try:
    from src.utils.aa_map import decode_indices_to_seq
    HAVE_DECODER = True
except Exception:
    HAVE_DECODER = False

def safe_float(x):
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        return float(np.asarray(x).astype(float))
    except Exception:
        return float("nan")

def load_npz(path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    d = np.load(path, allow_pickle=True)
    return d

def write_csv(summary_arr, out_csv):
    fieldnames = ["file", "classical_score", "best_score", "delta"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_arr:
            cs = row.get("classical_score")
            bs = row.get("best_score")
            try:
                delta = float(bs) - float(cs)
            except Exception:
                delta = ""
            writer.writerow({
                "file": row.get("file", ""),
                "classical_score": float(cs) if cs is not None and not math.isnan(float(cs)) else "",
                "best_score": float(bs) if bs is not None and not math.isnan(float(bs)) else "",
                "delta": delta
            })
    print(f"Saved CSV -> {out_csv}")

def write_fasta(results_list, out_fasta):
    with open(out_fasta, "w") as fh:
        for r in results_list:
            name = r.get("file", "unnamed")
            cls_idx = r.get("classical_seq_idx")
            best_idx = r.get("best_seq_idx")
            if cls_idx is not None and len(cls_idx):
                fh.write(f">{name}|classical\n")
                try:
                    if HAVE_DECODER:
                        seq = decode_indices_to_seq(list(cls_idx))
                        fh.write(seq + "\n")
                    else:
                        fh.write(" ".join(map(str, list(cls_idx))) + "\n")
                except Exception:
                    fh.write(" ".join(map(str, list(cls_idx))) + "\n")
            if best_idx is not None and len(best_idx):
                fh.write(f">{name}|quantum\n")
                try:
                    if HAVE_DECODER:
                        seq = decode_indices_to_seq(list(best_idx))
                        fh.write(seq + "\n")
                    else:
                        fh.write(" ".join(map(str, list(best_idx))) + "\n")
                except Exception:
                    fh.write(" ".join(map(str, list(best_idx))) + "\n")
    print(f"Saved FASTA -> {out_fasta}")

def print_summary(results, summary_rows):
    n = len(results)
    improvements = 0
    worsened = 0
    deltas = []
    print("\n=== Quantum Opt Results Summary ===\n")
    for r, s in zip(results, summary_rows):
        cs = safe_float(s.get("classical_score", np.nan))
        bs = safe_float(s.get("best_score", np.nan))
        delta = None
        if not math.isnan(cs) and not math.isnan(bs):
            delta = bs - cs
        if delta is not None:
            deltas.append(delta)
            if delta > 0:
                improvements += 1
            elif delta < 0:
                worsened += 1
        print(f"file: {s.get('file')}\n  classical_score: {cs:.6f}  best_score: {bs:.6f}  delta: {delta}\n")
    if deltas:
        mean = float(np.mean(deltas))
        std = float(np.std(deltas, ddof=0))
        print(f"Total proteins: {n}")
        print(f"Mean Δ: {mean:.6f} ± {std:.6f}")
        print(f"Improved: {improvements}, Worsened: {worsened}")
        # top improvements
        top = sorted(zip([r["file"] for r in summary_rows], deltas), key=lambda x: -x[1])[:5]
        if top:
            print("\nTop improvements:")
            for fname, d in top:
                print(f" - {fname}: Δ={d:.6f}")
    else:
        print("No numeric deltas found.")

def main():
    try:
        d = load_npz(IN_PATH)
    except Exception as e:
        print("ERROR loading npz:", e)
        sys.exit(1)

    # results is likely an object-array of dicts
    results_obj = d.get("results", None)
    summary_obj = d.get("summary", None)
    if results_obj is None:
        print("No 'results' key in npz.")
        sys.exit(1)

    try:
        results = results_obj.tolist()
    except Exception:
        # fallback if already list-like
        results = list(results_obj)

    # if summary exists use it; otherwise build
    if summary_obj is not None:
        try:
            summary_rows = list(summary_obj.tolist())
        except Exception:
            summary_rows = list(summary_obj)
    else:
        summary_rows = []
        for r in results:
            cs = r.get("classical_score", np.nan)
            bs = r.get("best_score", np.nan)
            summary_rows.append({"file": r.get("file", "unnamed"), "classical_score": cs, "best_score": bs})

    # Ensure output dir exists
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)

    # write CSV and FASTA
    write_csv(summary_rows, OUT_CSV)
    write_fasta(results, OUT_FASTA)

    # print summary to stdout
    print_summary(results, summary_rows)

if __name__ == "__main__":
    main()
