# src/evaluate_vae.py
import torch
import numpy as np
from pathlib import Path
from src.models.classical_nn import ProteinVAE

MODEL_PATH = Path("models/interim_models/protein_vae.pt")
DATA_DIR = Path("data/processed")

def idxs_to_aa(idxs):
    AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(AA_LIST[i] if (i >= 0 and i < len(AA_LIST)) else "-" for i in idxs)

def load_model(device="cpu"):
    model = ProteinVAE(vocab_size=20, seq_len=300)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def reconstruct_one(model, seq_enc, device="cpu"):
    x = torch.tensor(seq_enc, dtype=torch.long).unsqueeze(0).to(device)  # [1, L]
    with torch.no_grad():
        recon, mu, logvar = model(x)
        pred = recon.argmax(dim=-1).squeeze(0).cpu().numpy()
    return pred

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = load_model(device=device)

    files = sorted(DATA_DIR.glob("*.npz"))
    for f in files:
        d = np.load(f, allow_pickle=True)
        seq_enc = d["seq_enc"]
        raw = d["raw_seq"].tolist() if isinstance(d["raw_seq"], np.ndarray) else d["raw_seq"]
        pred = reconstruct_one(model, seq_enc, device=device)
        print("----", f.name)
        print("RAW  (first 80):", raw[:80])
        print("PRED (first 80):", idxs_to_aa(pred[:80]))

if __name__ == "__main__":
    main()
