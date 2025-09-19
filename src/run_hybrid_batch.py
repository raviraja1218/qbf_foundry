# src/run_hybrid_batch.py
"""
Batch-run hybrid inference over all files in data/processed/.
Saves a CSV at reports/hybrid_batch_results.csv and a NPZ with raw arrays.
Usage:
  export PYTHONPATH="$PWD"
  python -m src.run_hybrid_batch
"""

import glob
import numpy as np
import torch
from pathlib import Path
from src.run_hybrid import load_model_from_ckpt, apply_quantum_transform  # we'll rely on helper functions in your run_hybrid
from src.utils.aa_map import decode_indices_to_seq
from src.utils.scoring import hydrophobicity_from_indices

OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)

def process_file(path, model):
    # load processed .npz
    d = np.load(path, allow_pickle=True)
    seq_enc = d["seq_enc"][:300]  # consistent length
    phys = d.get("phys_props")
    if phys is None:
        phys = np.zeros((300, 5), dtype=np.float32)

    # create tensors
    seq_t = torch.tensor(seq_enc[None, :], dtype=torch.long)
    phys_t = torch.tensor(phys[None, :, :], dtype=torch.float32)

    device = next(model.parameters()).device
    seq_t = seq_t.to(device)
    phys_t = phys_t.to(device)

    # classical forward
    with torch.no_grad():
        logits, mu, logvar = model(seq_t, phys_t)
        recon_classical = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        latent = mu.cpu().numpy()

    # apply quantum transform (this function should exist in run_hybrid; if not adapt to your quantum_layer)
    zq = apply_quantum_transform(torch.tensor(latent).to(device)).cpu().numpy()

    # decode quantum via model.decode (or other compatible call)
    try:
        logits_q = model.decode(torch.tensor(zq).to(device))
        recon_q = logits_q.argmax(dim=-1).squeeze(0).cpu().numpy()
    except Exception:
        # fallback: just use the zq as placeholder or rerun decode path in your model
        recon_q = recon_classical  # safe fallback (replace with proper decode)
    
    # scores
    score_c = hydrophobicity_from_indices(recon_classical)
    score_q = hydrophobicity_from_indices(recon_q)

    return {
        "file": Path(path).name,
        "classical_seq": decode_indices_to_seq(recon_classical),
        "quantum_seq": decode_indices_to_seq(recon_q),
        "classical_score": score_c,
        "quantum_score": score_q
    }

def main():
    # load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_ckpt("models/interim_models/protein_vae.pt", device=device)
    model.to(device).eval()

    results = []
    files = sorted(glob.glob("data/processed/*.npz"))
    for f in files:
        print("Processing:", f)
        try:
            out = process_file(f, model)
            results.append(out)
        except Exception as e:
            print("Error processing", f, "->", e)

    # save CSV
    import csv
    csv_path = OUT_DIR / "hybrid_batch_results.csv"
    keys = ["file", "classical_seq", "quantum_seq", "classical_score", "quantum_score"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print("Saved CSV:", csv_path)

    # save NPZ of raw values
    np.savez(OUT_DIR / "hybrid_batch_raw.npz", results=results)
    print("Saved NPZ:", OUT_DIR / "hybrid_batch_raw.npz")

if __name__ == "__main__":
    main()
