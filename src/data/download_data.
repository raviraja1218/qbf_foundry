#!/usr/bin/env python3
"""
download_data.py - Fetch benchmark protein sequences + structures
Phase 2.1: scalable dataset builder (UniProt FASTA + AlphaFold PDB)
"""

import requests
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw/benchmark")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Start with a seed set (toy scale). We can later extend this list automatically via UniProt API.
UNIPROT_IDS = [
    "P00734",  # SampleA
    "P0A8T7",  # SampleB
    "P01009",  # SampleC
    "P69905",  # Hemoglobin subunit alpha
    "P68871",  # Hemoglobin subunit beta
    "P04637",  # p53 tumor suppressor
    "P38398",  # BRCA1
    "P01308",  # Insulin
    "P35579",  # Fibrillin
]

def fetch_fasta(uniprot_id: str, out_file: Path):
    """Download FASTA sequence from UniProt"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    r = requests.get(url)
    if r.status_code == 200:
        out_file.write_text(r.text)
    else:
        print(f"⚠️ Failed FASTA for {uniprot_id} ({r.status_code})")

def fetch_pdb(uniprot_id: str, out_file: Path):
    """Download AlphaFold predicted structure (PDB)"""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    r = requests.get(url)
    if r.status_code == 200:
        out_file.write_text(r.text)
    else:
        print(f"⚠️ Failed PDB for {uniprot_id} ({r.status_code})")

def main():
    for uid in tqdm(UNIPROT_IDS, desc="Downloading proteins"):
        fasta_out = RAW_DIR / f"{uid}.fasta"
        pdb_out = RAW_DIR / f"{uid}.pdb"

        if not fasta_out.exists():
            fetch_fasta(uid, fasta_out)
        else:
            print(f"✔️ FASTA already exists: {fasta_out}")

        if not pdb_out.exists():
            fetch_pdb(uid, pdb_out)
        else:
            print(f"✔️ PDB already exists: {pdb_out}")

    print(f"\n✅ Done. Files saved to {RAW_DIR}")

if __name__ == "__main__":
    main()

