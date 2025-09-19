# src/train.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from src.models.classical_nn import ProteinVAE, vae_loss, PAD_IDX

DATA_DIR = Path("data/processed")
MODEL_OUT = Path("models/interim_models")
MODEL_OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_OUT / "protein_vae.pt"

SEQ_LEN = 300

class ProteinDataset(Dataset):
    def __init__(self, files, seq_len=SEQ_LEN):
        self.files = files
        self.seq_len = seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        seq = d["seq_enc"].astype(np.int64)
        phys = d["phys_props"].astype(np.float32)
        # ensure shapes
        if seq.shape[0] < self.seq_len:
            pad = np.full((self.seq_len - seq.shape[0],), PAD_IDX, dtype=np.int64)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[: self.seq_len]
        if phys.shape[0] < self.seq_len:
            pad_phys = np.zeros((self.seq_len - phys.shape[0], phys.shape[1]), dtype=np.float32)
            phys = np.concatenate([phys, pad_phys], axis=0)
        else:
            phys = phys[: self.seq_len]
        return {"seq": seq, "phys": phys}

def collate_fn(batch):
    seqs = torch.tensor([b["seq"] for b in batch], dtype=torch.long)
    phys = torch.tensor([b["phys"] for b in batch], dtype=torch.float32)
    return seqs, phys

def train_vae(epochs=10, batch_size=8, lr=1e-3, device=None, kl_beta=1.0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted([str(p) for p in DATA_DIR.glob("*.npz")])
    dataset = ProteinDataset(files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = ProteinVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kld = 0.0
        for seqs, phys in loader:
            seqs = seqs.to(device)
            phys = phys.to(device)
            opt.zero_grad()
            logits, mu, logvar = model(seqs, phys)
            loss, ce_val, kld_val = vae_loss(logits, seqs, mu, logvar, pad_idx=PAD_IDX, kl_beta=kl_beta)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_ce += ce_val
            total_kld += kld_val
        n_batches = len(loader)
        print(f"Epoch {epoch+1}/{epochs} — Avg Loss: {total_loss / n_batches:.4f} — CE: {total_ce:.1f} — KLD: {total_kld:.1f}")
    torch.save(model.state_dict(), str(MODEL_PATH))
    print("✅ VAE training complete. Model saved at", MODEL_PATH)

if __name__ == "__main__":
    train_vae(epochs=10, batch_size=8, lr=1e-3)
