"""
Run a quick classical vs quantum-augmented comparison.

Usage:
    PYTHONPATH="$PWD" python -m src.run_hybrid
"""

import os
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Project model import (expects src/models/classical_nn.py present)
from src.models.classical_nn import ProteinVAE

# try to import a dedicated quantum layer, otherwise provide a small fallback
try:
    from src.models.quantum_layer import QuantumLayer  # user-provided quantum layer (preferred)
except Exception:
    QuantumLayer = None

# If PennyLane is available, fallback QuantumLayer uses it.
try:
    import pennylane as qml  # type: ignore
    _PL_AVAILABLE = True
except Exception:
    _PL_AVAILABLE = False


MODEL_PATH = "models/interim_models/protein_vae.pt"
DATA_GLOB = "data/processed/*.npz"
REPORTS_DIR = Path("reports")


# -------------------------
# Utilities
# -------------------------
def _adapt_tensor_to_shape(src: torch.Tensor, target_shape):
    src = src.detach().cpu()
    tgt_shape = tuple(int(x) for x in target_shape)
    if tuple(src.shape) == tgt_shape:
        return src

    if src.ndim == 1 and len(tgt_shape) == 1:
        s_len, t_len = src.shape[0], tgt_shape[0]
        out = torch.zeros(t_len, dtype=src.dtype)
        out[:min(s_len, t_len)] = src[:min(s_len, t_len)]
        return out

    if src.ndim == 2 and len(tgt_shape) == 2:
        s0, s1 = src.shape
        t0, t1 = tgt_shape
        out = torch.zeros((t0, t1), dtype=src.dtype)
        out[:min(s0, t0), :min(s1, t1)] = src[:min(s0, t0), :min(s1, t1)]
        return out

    flat = src.view(-1)
    target_size = np.prod(tgt_shape)
    if flat.shape[0] >= target_size:
        flat2 = flat[:target_size].clone()
    else:
        pad = torch.zeros(target_size - flat.shape[0], dtype=flat.dtype)
        flat2 = torch.cat([flat, pad], dim=0)
    return flat2.view(*tgt_shape)


def inspect_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    info = {"vocab": None, "embed_dim": None, "seq_len": None, "latent_dim": None}
    for k, v in sd.items():
        if "embedding" in k and isinstance(v, torch.Tensor) and v.ndim == 2:
            info["vocab"], info["embed_dim"] = v.shape
        if ("fc_mu" in k or "fc_logvar" in k) and isinstance(v, torch.Tensor) and v.ndim == 2:
            info["latent_dim"] = v.shape[0]
        if ("fc_dec" in k) and isinstance(v, torch.Tensor) and v.ndim == 2 and info["vocab"]:
            possible_seq_len = v.shape[0] // info["vocab"]
            info["seq_len"] = possible_seq_len
    if info["vocab"] is None: info["vocab"] = 21
    if info["embed_dim"] is None: info["embed_dim"] = 64
    if info["seq_len"] is None: info["seq_len"] = 300
    if info["latent_dim"] is None: info["latent_dim"] = 64
    return info


def load_model_from_ckpt(path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    info = inspect_checkpoint(path)
    print("[run_hybrid] inferred from checkpoint:", info)

    model = ProteinVAE(vocab_size=info["vocab"],
                       seq_len=info["seq_len"],
                       embed_dim=info["embed_dim"],
                       latent_dim=info["latent_dim"])
    model_sd = model.state_dict()
    adapted_sd = {}
    for k, v in state_dict.items():
        if k in model_sd and isinstance(v, torch.Tensor):
            adapted_sd[k] = _adapt_tensor_to_shape(v, model_sd[k].shape)
    final_sd = {k: adapted_sd.get(k, v) for k, v in model_sd.items()}
    model.load_state_dict(final_sd, strict=True)
    model.to(device).eval()
    print("[run_hybrid] Loaded checkpoint with shape-adaptation fallback.")
    return model


# -------------------------
# Fallback QuantumLayer
# -------------------------
if QuantumLayer is None:
    class QuantumLayer(nn.Module):
        def __init__(self, input_dim=64, n_qubits=4, n_layers=1):
            super().__init__()
            self.linear = nn.Linear(input_dim, input_dim)
        def forward(self, x): return self.linear(x)


# -------------------------
# Quantum transform helper
# -------------------------
def apply_quantum_transform(z_tensor, n_qubits=8, n_layers=1):
    """
    Apply a QuantumLayer to a latent tensor.
    Returns transformed tensor of same shape.
    """
    latent_dim = z_tensor.shape[-1]
    qlayer = QuantumLayer(input_dim=latent_dim,
                          n_qubits=min(n_qubits, latent_dim),
                          n_layers=n_layers)
    qlayer.to(z_tensor.device).eval()
    with torch.no_grad():
        return qlayer(z_tensor)


# -------------------------
# Data/sample loader
# -------------------------
def load_sample(max_len=300):
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError("No processed files found in data/processed/")
    f = files[0]
    d = np.load(f, allow_pickle=True)
    seq = d["seq_enc"][:max_len]
    phys = d.get("phys_props", np.zeros((max_len, 5), dtype=np.float32))[:max_len]
    seq_t = torch.tensor(seq[None, :], dtype=torch.long)
    phys_t = torch.tensor(phys[None, :, :], dtype=torch.float32)
    return seq_t, phys_t, f


# -------------------------
# Main flow
# -------------------------
def main(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = load_model_from_ckpt(MODEL_PATH, device=device)
    seq_t, phys_t, sample_file = load_sample()
    seq_t, phys_t = seq_t.to(device), phys_t.to(device)

    with torch.no_grad():
        logits, mu, _ = model(seq_t, phys_t)
        z = mu
        recon_classical = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
    print("[run_hybrid] sample file:", sample_file)
    print("[run_hybrid] classical latent sample (first 8):", z[0, :8].cpu().numpy())

    zq = apply_quantum_transform(z)
    if hasattr(model, "decode"):
        logits_q = model.decode(zq)
    else:
        lin = nn.Linear(zq.shape[-1], model.seq_len * model.vocab_size).to(device)
        out_flat = lin(zq)
        logits_q = out_flat.view(1, model.seq_len, model.vocab_size)
    recon_quantum = logits_q.argmax(dim=-1).squeeze(0).cpu().numpy()

    print("---- Comparison (first 80 tokens) ----")
    print("CLASSICAL:", " ".join([f"{int(x):02d}" for x in recon_classical[:80]]))
    print("QUANTUM :", " ".join([f"{int(x):02d}" for x in recon_quantum[:80]]))

    REPORTS_DIR.mkdir(exist_ok=True)
    np.savez(REPORTS_DIR / "hybrid_sample.npz",
             sample_file=str(sample_file),
             classical=recon_classical,
             quantum=recon_quantum,
             latent_classical=z.cpu().numpy(),
             latent_quantum=zq.cpu().numpy())
    print("[run_hybrid] Saved", REPORTS_DIR / "hybrid_sample.npz")


if __name__ == "__main__":
    main()
