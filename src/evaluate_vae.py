# src/evaluate_vae.py
import torch
import numpy as np
from pathlib import Path
from src.models.classical_nn import ProteinVAE

MODEL_PATH = Path("models/interim_models/protein_vae.pt")
DATA_DIR = Path("data/processed")

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"  # 20 amino acids

def idxs_to_aa(idxs):
    # map indices to amino acids; out-of-range -> '-'
    return "".join(AA_LIST[i] if (0 <= int(i) < len(AA_LIST)) else "-" for i in idxs)

def load_model(device="cpu"):
    model = ProteinVAE(vocab_size=20, seq_len=300)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def reconstruct_one(model, seq_enc, device="cpu"):
    """
    seq_enc : 1D array-like of ints (length L)
    Returns: predicted index sequence (1D numpy array)
    """
    x = torch.tensor(seq_enc, dtype=torch.long).unsqueeze(0).to(device)  # [1, L]
    with torch.no_grad():
        recon, mu, logvar = model(x)
        pred = recon.argmax(dim=-1).squeeze(0).cpu().numpy()
    return pred

def safe_seq_from_npz_entry(entry):
    """Ensure seq_enc is a 1D integer numpy array and clamp to valid indices."""
    seq = np.asarray(entry, dtype=np.int64).ravel()
    # Model uses vocab_size=20 and pad_id == 20 in training code (embedding has vocab_size+1 rows).
    # Valid token indices for amino acids: 0..19. pad_id may be 20; clamp to [0,20].
    vocab_plus_pad = 20  # if you used pad_id = vocab_size
    seq = np.clip(seq, 0, vocab_plus_pad)  # clamp any stray values
    return seq

def pretty_raw(raw_entry):
    # raw_entry might be a numpy scalar, bytes, or str
    if isinstance(raw_entry, np.ndarray):
        try:
            val = raw_entry.tolist()
        except Exception:
            val = str(raw_entry)
    else:
        val = raw_entry
    # if bytes, decode
    if isinstance(val, (bytes, bytearray)):
        try:
            val = val.decode("utf-8", errors="ignore")
        except Exception:
            val = str(val)
    return str(val)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    try:
        model = load_model(device=device)
    except FileNotFoundError:
        print("Model file not found at", MODEL_PATH)
        return
    except Exception as e:
        print("Failed to load model:", e)
        return

    files = sorted(DATA_DIR.glob("*.npz"))
    if not files:
        print("No processed files found in", DATA_DIR)
        return

    for f in files:
        d = np.load(f, allow_pickle=True)
        if "seq_enc" not in d.files:
            print(f"Skipping {f.name}: 'seq_enc' key not found in npz")
            continue

        raw = d["raw_seq"] if "raw_seq" in d.files else None
        raw_text = pretty_raw(raw)

        # sanitize / clamp sequence indices
        seq_enc = safe_seq_from_npz_entry(d["seq_enc"])

        # attempt reconstruction, skip sample on runtime errors
        try:
            pred = reconstruct_one(model, seq_enc, device=device)
        except RuntimeError as e:
            print(f"⚠️ Skipping {f.name} due to runtime error during reconstruction: {e}")
            continue
        except Exception as e:
            print(f"⚠️ Skipping {f.name} due to unexpected error: {e}")
            continue

        print("----", f.name)
        print("RAW  (first 80):", raw_text[:80])
        print("PRED (first 80):", idxs_to_aa(pred[:80]))

if __name__ == "__main__":
    main()

