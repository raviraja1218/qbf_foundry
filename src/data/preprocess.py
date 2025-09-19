# src/data/preprocess.py
"""
Enhanced preprocessing for QBF Phase 2:
 - reads FASTA / PDB
 - creates seq_enc (ints 0..19, padded to seq_len)
 - extracts coords adjacency (if PDB present)
 - computes physico-chemical properties per residue -> phys_props (seq_len x D)
 - saves .npz with keys: seq_enc, coords, adj, raw_seq, phys_props
"""

from pathlib import Path
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser
import math

RAW_DIR = Path("data/raw/benchmark")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# basic AA alphabet and index map (20 standard AAs)
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
PAD_IDX = len(AA_ORDER)   # pad index = 20

SEQ_LEN = 300  # fixed length we used earlier; same convention

# Physicochemical properties lookup table (per-residue)
# Columns: [hydropathy (Kyte-Doolittle), volume (Å^3, approximate), polarity, charge, aromaticity]
# Values are illustrative and commonly used; you can refine later with precise scales
PHYS_PROP = {
    'A': [1.8,  88.6,  0.0,  0.0, 0.0],
    'C': [2.5,  108.5, 0.0,  0.0, 0.0],
    'D': [-3.5, 111.1, 1.0, -1.0, 0.0],
    'E': [-3.5, 138.4, 1.0, -1.0, 0.0],
    'F': [2.8,  189.9, 0.0,  0.0, 1.0],
    'G': [-0.4, 60.1,  0.0,  0.0, 0.0],
    'H': [-3.2, 153.2, 1.0,  0.0, 0.0],
    'I': [4.5,  166.7, 0.0,  0.0, 0.0],
    'K': [-3.9, 168.6, 1.0,  1.0, 0.0],
    'L': [3.8,  166.7, 0.0,  0.0, 0.0],
    'M': [1.9,  162.9, 0.0,  0.0, 0.0],
    'N': [-3.5, 114.1, 1.0,  0.0, 0.0],
    'P': [-1.6, 112.7, 0.0,  0.0, 0.0],
    'Q': [-3.5, 143.8, 1.0,  0.0, 0.0],
    'R': [-4.5, 173.4, 1.0,  1.0, 0.0],
    'S': [-0.8,  89.0, 1.0,  0.0, 0.0],
    'T': [-0.7, 116.1, 1.0,  0.0, 0.0],
    'V': [4.2,  140.0, 0.0,  0.0, 0.0],
    'W': [-0.9, 227.8, 0.0,  0.0, 1.0],
    'Y': [-1.3, 193.6, 0.0,  0.0, 1.0],
}

# Normalization helpers (we'll normalize phys props per column)
PROP_NAMES = ["hydropathy", "volume", "polarity", "charge", "aromaticity"]

def seq_to_indices(seq, seq_len=SEQ_LEN):
    seq = seq.upper()
    arr = np.full(seq_len, PAD_IDX, dtype=np.int64)
    for i, aa in enumerate(seq[:seq_len]):
        arr[i] = AA_TO_IDX.get(aa, PAD_IDX)  # unknown -> pad idx
    return arr

def phys_props_for_sequence(seq, seq_len=SEQ_LEN):
    seq = seq.upper()
    D = len(PROP_NAMES)
    props = np.zeros((seq_len, D), dtype=np.float32)
    for i, aa in enumerate(seq[:seq_len]):
        if aa in PHYS_PROP:
            props[i] = np.array(PHYS_PROP[aa], dtype=np.float32)
        else:
            props[i] = np.zeros(D, dtype=np.float32)
    # Normalize columns: mean 0, std 1 (avoid dividing by zero)
    mean = props.mean(axis=0)
    std = props.std(axis=0)
    std[std == 0] = 1.0
    props = (props - mean) / std
    return props

def pdb_coords_and_adj(pdb_path):
    """Return Nx3 coords (CA) and adjacency matrix NxN (distance-based graph)
       If no PDB or parsing fails, return (None, None)
    """
    if not pdb_path.exists():
        return None, None
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("X", str(pdb_path))
        # collect CA atoms in first model, first chain(s)
        coords = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if 'CA' in res:
                        ca = res['CA'].get_coord()
                        coords.append(ca)
            break
        if len(coords) == 0:
            return None, None
        coords = np.array(coords, dtype=np.float32)
        # adjacency: simple distance threshold graph (8.0 Å)
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(coords))
        adj = (dists < 8.0).astype(np.float32)
        return coords, adj
    except Exception as e:
        print("PDB parse error:", e)
        return None, None

def preprocess_one(fasta_path, pdb_path=None, out_path=None):
    record = SeqIO.read(str(fasta_path), "fasta")
    raw_seq = str(record.seq)
    seq_enc = seq_to_indices(raw_seq, SEQ_LEN)
    phys_props = phys_props_for_sequence(raw_seq, SEQ_LEN)
    coords, adj = None, None
    if pdb_path and pdb_path.exists():
        coords, adj = pdb_coords_and_adj(pdb_path)
    out = {
        "seq_enc": seq_enc,
        "coords": coords,
        "adj": adj,
        "raw_seq": raw_seq,
        "phys_props": phys_props,
    }
    if out_path is None:
        out_path = OUT_DIR / (fasta_path.stem + ".npz")
    np.savez_compressed(out_path, **out)
    print("Saved:", out_path)

def main():
    # find FASTA files, pair with similarly named PDB if exists
    fasta_files = sorted(RAW_DIR.glob("*.fasta"))
    if not fasta_files:
        print("No FASTA files in", RAW_DIR)
        return
    for f in fasta_files:
        pdb_candidate = RAW_DIR / (f.stem + ".pdb")
        print("Preprocessing", f.stem)
        preprocess_one(f, pdb_candidate, OUT_DIR / (f.stem + ".npz"))

if __name__ == "__main__":
    main()
