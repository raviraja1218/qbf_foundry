# src/view_hybrid_npz.py
"""
Simple viewer for reports/hybrid_sample.npz
Run with:
    export PYTHONPATH="$PWD"
    python src/view_hybrid_npz.py
"""

import numpy as np
from pathlib import Path

# import our decoder
try:
    from src.utils.aa_map import decode_indices_to_seq
except Exception as e:
    # fallback: simple mapping A..T for 0..19 (shouldn't happen if src/utils/aa_map.py exists)
    print("Warning: couldn't import src.utils.aa_map.decode_indices_to_seq:", e)
    def decode_indices_to_seq(i): 
        order = list("ACDEFGHIKLMNPQRSTVWY")
        return "".join([order[int(x)] if 0 <= int(x) < 20 else "-" for x in i])

NPZ = Path("reports/hybrid_sample.npz")

def main():
    if not NPZ.exists():
        print("File not found:", NPZ)
        return
    d = np.load(NPZ, allow_pickle=True)
    print("Keys:", d.files)
    sample_file = d.get("sample_file", "unknown")
    print("Sample file used:", sample_file)

    classical = d["classical"]
    quantum = d["quantum"]
    latent_c = d["latent_classical"]
    latent_q = d["latent_quantum"]

    # decode to letters
    seq_c = decode_indices_to_seq(classical)
    seq_q = decode_indices_to_seq(quantum)

    # print short summary
    print("\n--- Short summary ---")
    print("Classical (first 120 aa):")
    print(seq_c[:120])
    print("\nQuantum  (first 120 aa):")
    print(seq_q[:120])

    # Save FASTA files for inspection
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    fasta_c = out_dir / "hybrid_classical.fasta"
    fasta_q = out_dir / "hybrid_quantum.fasta"
    fasta_c.write_text(">classical_from_hybrid_sample\n" + seq_c + "\n")
    fasta_q.write_text(">quantum_from_hybrid_sample\n" + seq_q + "\n")
    print("\nSaved FASTA files:")
    print(" -", fasta_c)
    print(" -", fasta_q)

    # also print latent shapes / small numeric preview
    print("\nLatent (classical) shape:", latent_c.shape)
    print("Latent (quantum) shape:   ", latent_q.shape)
    print("Latent classical (first row, first 8):", latent_c.ravel()[:8])
    print("Latent quantum    (first row, first 8):", latent_q.ravel()[:8])

if __name__ == "__main__":
    main()
