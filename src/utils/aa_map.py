# src/utils/aa_map.py
# Simple index <-> amino-acid mapping used by the project.
# Make sure the vocab and PAD index match your models (here vocab indices 0..20, PAD=20)

AA_LIST = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
]
# If your model uses a PAD or special token appended (index 20), keep this consistent:
PAD_IDX = 20
PAD_CHAR = "-"

# build maps
IDX_TO_AA = {i: aa for i, aa in enumerate(AA_LIST)}
IDX_TO_AA[PAD_IDX] = PAD_CHAR

AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
AA_TO_IDX[PAD_CHAR] = PAD_IDX

def decode_indices_to_seq(indices):
    """Turn numeric indices (iterable) into FASTA-style sequence string."""
    return "".join(IDX_TO_AA.get(int(i), "?") for i in indices)

def encode_seq_to_indices(seq, max_len=None):
    """Encode ASCII protein sequence to indices; unknown->PAD; optionally pad/truncate to max_len."""
    arr = [AA_TO_IDX.get(ch, PAD_IDX) for ch in seq]
    if max_len is not None:
        if len(arr) < max_len:
            arr = arr + [PAD_IDX] * (max_len - len(arr))
        else:
            arr = arr[:max_len]
    return arr
