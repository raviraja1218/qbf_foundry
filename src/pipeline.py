# src/pipeline.py
import numpy as np
import torch
from pathlib import Path

# imports from your project
from src.models.classical_nn import ProteinVAE
from src.models.quantum_module import build_qaoa

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("models/interim_models/protein_vae.pt")

# index -> amino acid map (20 canonical amino acids)
IDX_TO_AA = {
    0: "A", 1: "C", 2: "D", 3: "E", 4: "F",
    5: "G", 6: "H", 7: "I", 8: "K", 9: "L",
    10: "M", 11: "N", 12: "P", 13: "Q", 14: "R",
    15: "S", 16: "T", 17: "V", 18: "W", 19: "Y"
}

def idxs_to_seq(idxs):
    """Convert integer index array to amino-acid string."""
    return "".join(IDX_TO_AA.get(int(i), "X") for i in idxs)

def load_vae():
    model = ProteinVAE(vocab_size=20, seq_len=300).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def sample_from_vae(model, seq_len=50):
    """Sample a latent vector, decode with the VAE and return integer sequence (len seq_len)."""
    # create random latent vector with size matching model
    latent_dim = getattr(model, "latent_dim", None)
    if latent_dim is None:
        # fallback — try common attribute names
        latent_dim = 64
    z = torch.randn((1, latent_dim), device=DEVICE)
    # the decode() implementation may return logits; adapt as needed
    logits, _, _ = model.decode(z, seq_len=seq_len)
    seq_idxs = logits.argmax(dim=-1).cpu().numpy()[0]
    return seq_idxs

def run_quantum_sampler(num_qubits=4, p=1):
    run_qaoa, run_random = build_qaoa(num_qubits=num_qubits, p=p)
    params, cost = run_random(steps=15, lr=0.2)
    return params, cost

def run_pipeline():
    # 1) Load VAE
    model = load_vae()
    # 2) Sample a sequence (indices) from VAE
    seq_idxs = sample_from_vae(model, seq_len=50)
    seq_str = idxs_to_seq(seq_idxs)
    print("Sampled (indices):", seq_idxs[:50])
    print("Sampled (FASTA-style):")
    print(">sample_from_vae")
    print(seq_str)

    # 3) Run quantum sampler (toy)
    params, cost = run_quantum_sampler(num_qubits=4, p=1)
    print("\nQuantum-optimized params:", params)
    print("Quantum sampling cost:", cost)

if __name__ == "__main__":
    run_pipeline()

