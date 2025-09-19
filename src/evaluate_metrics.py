#!/usr/bin/env python3
"""
src/evaluate_metrics.py

Evaluate the trained ProteinVAE:
 - per-residue accuracy (ignoring PAD)
 - per-sequence accuracy
 - amino-acid frequency comparison
 - confusion matrix (saved as image)
Outputs:
 - reports/eval_metrics.csv
 - reports/aa_freq.png
 - reports/confusion_matrix.png
 - reports/accuracy_hist.png
"""

import os
from pathlib import Path
import glob
import numpy as np
import torch
import math
import csv
import matplotlib.pyplot as plt

# Project import (ensure you run with PYTHONPATH="$PWD")
from src.models.classical_nn import ProteinVAE, PAD_IDX, VOCAB_SIZE

# Paths
ROOT = Path.cwd()
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "interim_models" / "protein_vae.pt"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(device=DEVICE):
    model = ProteinVAE().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_dataset(files):
    """Return list of dict {seq_enc (np int array length L), phys_props (LxF)}"""
    items = []
    for f in sorted(files):
        d = np.load(f, allow_pickle=True)
        seq = d["seq_enc"].astype(int)
        phys = d.get("phys_props", None)
        items.append({"file": f, "seq": seq, "phys": phys})
    return items

def evaluate(model, items, device=DEVICE, max_len=300):
    all_true = []
    all_pred = []
    seq_accuracies = []

    for it in items:
        seq = it["seq"][:max_len]
        phys = it["phys"]
        if phys is not None:
            phys = phys[:max_len]
        # prepare tensors
        x = torch.tensor(seq[None, :], dtype=torch.long, device=device)
        phys_t = None
        if phys is not None:
            phys_t = torch.tensor(phys[None, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, mu, logvar = model(x, phys_t) if phys_t is not None else model(x)
            # logits shape: [1, L, VOCAB_SIZE+maybe PAD?]
            pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy().astype(int)

        true = seq.copy()
        # compute mask: ignore PAD_IDX and out-of-range values (<0)
        mask = (true != PAD_IDX) & (true >= 0) & (true < VOCAB_SIZE)
        if mask.sum() == 0:
            seq_acc = float("nan")
        else:
            seq_acc = (pred[mask] == true[mask]).sum() / float(mask.sum())
        seq_accuracies.append(seq_acc)

        all_true.append(true)
        all_pred.append(pred)

    return all_true, all_pred, seq_accuracies

def flatten_lists(list_of_arrays):
    return np.concatenate([a.flatten() for a in list_of_arrays]).astype(int)

def compute_confusion(y_true, y_pred, vocab_size=VOCAB_SIZE, pad_idx=20):
    cm = np.zeros((vocab_size, vocab_size), dtype=int)
    for a, b in zip(y_true, y_pred):
        if a == pad_idx or b == pad_idx:  # skip PAD tokens
            continue
        cm[a, b] += 1
    return cm


def plot_aa_freq(true_flat, pred_flat, vocab_size=20, out_path=None):
    # compute frequencies
    tcounts = np.bincount(true_flat.clip(0, vocab_size-1), minlength=vocab_size)
    pcounts = np.bincount(pred_flat.clip(0, vocab_size-1), minlength=vocab_size)
    tfreq = tcounts / (tcounts.sum() + 1e-12)
    pfreq = pcounts / (pcounts.sum() + 1e-12)
    aa_idxs = np.arange(vocab_size)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(aa_idxs - width/2, tfreq, width, label="True")
    ax.bar(aa_idxs + width/2, pfreq, width, label="Pred")
    ax.set_xlabel("Amino-acid index")
    ax.set_ylabel("Frequency")
    ax.set_title("Amino-acid frequency: True vs Pred")
    ax.legend()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return tcounts, pcounts

def plot_confusion(cm, out_path=None):
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (counts)")
    fig.colorbar(im, ax=ax)
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def save_metrics_csv(files, seq_accuracies, out_csv):
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "seq_accuracy"])
        for f,acc in zip(files, seq_accuracies):
            writer.writerow([os.path.basename(f), "" if acc is None or math.isnan(acc) else f"{acc:.6f}"])
    print("Saved CSV:", out_csv)

def main():
    print("Device:", DEVICE)
    files = sorted(glob.glob(str(DATA_DIR / "*.npz")))
    if len(files) == 0:
        print("No processed files found in", DATA_DIR)
        return

    model = load_model()
    items = load_dataset(files)
    print("Loaded", len(items), "items")

    all_true, all_pred, seq_accuracies = evaluate(model, items)
    files_names = [Path(x["file"]).name for x in items]

    true_flat = flatten_lists(all_true)
    pred_flat = flatten_lists(all_pred)

    # metrics
    overall_mask = (true_flat >= 0) & (true_flat < VOCAB_SIZE) & (true_flat != PAD_IDX)
    overall_acc = (pred_flat[overall_mask] == true_flat[overall_mask]).sum() / float(overall_mask.sum())
    mean_seq_acc = np.nanmean(seq_accuracies)

    print(f"Overall per-residue accuracy: {overall_acc:.4f}")
    print(f"Mean per-sequence accuracy: {mean_seq_acc:.4f}")

    # Save CSV of per-sequence accuracies
    save_metrics_csv(files_names, seq_accuracies, REPORT_DIR / "eval_metrics.csv")

    # Plot AA frequency
    plot_aa_freq(true_flat[overall_mask], pred_flat[overall_mask], VOCAB_SIZE, out_path=REPORT_DIR / "aa_freq.png")
    print("Saved:", REPORT_DIR / "aa_freq.png")

    # Confusion
    cm = compute_confusion(true_flat[overall_mask], pred_flat[overall_mask], vocab_size=VOCAB_SIZE)
    plot_confusion(cm, out_path=REPORT_DIR / "confusion_matrix.png")
    print("Saved:", REPORT_DIR / "confusion_matrix.png")

    # Save histogram of per-sequence accuracy
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist([a for a in seq_accuracies if not math.isnan(a)], bins=20)
    ax.set_xlabel("Per-sequence accuracy")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of per-sequence accuracy")
    fig.savefig(REPORT_DIR / "accuracy_hist.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", REPORT_DIR / "accuracy_hist.png")

    # Summary file
    summary = {
        "overall_per_residue_accuracy": float(overall_acc),
        "mean_per_sequence_accuracy": float(mean_seq_acc),
        "n_sequences": len(items),
        "n_residues_evaluated": int(overall_mask.sum()),
    }
    with open(REPORT_DIR / "summary.txt", "w") as fh:
        for k,v in summary.items():
            fh.write(f"{k}: {v}\n")
    print("Saved summary:", REPORT_DIR / "summary.txt")
    print("Done.")

if __name__ == "__main__":
    main()
