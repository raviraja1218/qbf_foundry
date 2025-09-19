# paste into src/run_hardware_batch.py
"""
Small driver: runs 2-4 qubit ansatz on hardware for first N files.
"""
import argparse, glob, logging
from pathlib import Path
from src.models.ibm_hardware_wrapper import run_small_ansatz_on_ibm
from qiskit import QuantumCircuit

LOG = logging.getLogger("run_hardware_batch")
logging.basicConfig(level=logging.INFO)

def make_test_ansatz(n_qubits=2):
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    qc.cx(0, 1) if n_qubits > 1 else None
    return qc

def main(limit=2, backend="ibm_brisbane", shots=256):
    files = sorted(glob.glob("data/processed/*.npz"))[:limit]
    for f in files:
        LOG.info("Submitting small run for file: %s", f)
        qc = make_test_ansatz(n_qubits=2)
        try:
            res = run_small_ansatz_on_ibm(qc, shots=shots, backend=backend)
            LOG.info("Result: %s", res["counts"])
        except Exception as e:
            LOG.error("Hardware run failed: %s", e)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--backend", type=str, default="ibm_brisbane")
    p.add_argument("--shots", type=int, default=256)
    args = p.parse_args()
    main(limit=args.limit, backend=args.backend, shots=args.shots)
