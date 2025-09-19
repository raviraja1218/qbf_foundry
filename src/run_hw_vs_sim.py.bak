# src/run_hw_vs_sim.py
"""
Run hardware vs simulator comparisons for a small ansatz.

Usage example:
    export PYTHONPATH="$PWD"
    python src/run_hw_vs_sim.py --backend ibm_torino --shots 128 --seeds 5 --qubits 2
"""
import argparse
import time
import json
import os
import logging
from typing import Dict

LOG = logging.getLogger("run_hw_vs_sim")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# local imports (your repo)
from src.models.ibm_hardware_wrapper import run_small_ansatz_on_ibm
from src.utils.hardware_validation import (
    safe_normalize_counts,
    align_distributions,
    js_divergence,
    kl_divergence,
    l1_distance,
    l2_distance,
    classical_state_fidelity,
    dump_json,
    plot_histograms,
    save_result_summary_csv,
)

# qiskit imports for local simulator
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.providers.aer import AerSimulator
    QISKIT_AVAILABLE = True
except Exception as e:
    LOG.error("qiskit / Aer not available: %s", e)
    QISKIT_AVAILABLE = False

import numpy as np
import uuid

def make_test_ansatz(n_qubits: int, seed: int = 0):
    """Simple hardware-friendly ansatz: H on first qubit, then ladder of CXs and Ry with random angles."""
    from qiskit import QuantumCircuit
    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    # entangle chain
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    # add small random rotations
    for q in range(n_qubits):
        theta = float(rng.uniform(0, 2 * np.pi))
        qc.ry(theta, q)
    return qc

def run_simulator(qc, shots=1024):
    """Run qc on AerSimulator in shot mode, return counts as dict(bitstring->int)."""
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit/Aer not available in this environment.")
    backend = AerSimulator()
    # transpile to simulator (no noise model here)
    tq = transpile(qc, backend=backend, optimization_level=1)
    job = backend.run(tq, shots=shots)
    result = job.result()
    # get_counts on first circuit
    try:
        counts = result.get_counts()
    except Exception:
        # attempt parsing primitive-style result
        counts = {}
        try:
            rd = result.to_dict()
            if "counts" in rd:
                counts = rd["counts"]
        except Exception:
            pass
    # normalize keys to standard order (qiskit returns keys with qubit 0 last)
    # We'll keep qiskit's bitstring ordering consistent between sim/hw
    return counts

def analyze_pair(sim_counts: Dict[str,int], hw_counts: Dict[str,int], shots:int):
    psim = safe_normalize_counts(sim_counts, shots=shots)
    phw = safe_normalize_counts(hw_counts, shots=shots)
    pa, pb, keys = align_distributions(psim, phw)
    js = js_divergence(pa, pb)
    # mean KL as symmetric measure
    kl_avg = 0.5 * (kl_divergence(pa, pb) + kl_divergence(pb, pa))
    l1 = l1_distance(pa, pb)
    l2 = l2_distance(pa, pb)
    fidelity = classical_state_fidelity(pa, pb)
    # pack top probabilities (for CSV readability)
    top_idx = np.argsort(- (pa + pb))[:10]
    sim_top = {str(keys[i]): float(pa[i]) for i in top_idx}
    hw_top = {str(keys[i]): float(pb[i]) for i in top_idx}
    return dict(js_div=js, kl_avg=kl_avg, l1=l1, l2=l2, fidelity=fidelity,
                sim_top_probs=sim_top, hw_top_probs=hw_top, pa=list(pa), pb=list(pb), keys=list(map(str, keys)))

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="ibm_torino")
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default="reports")
    parser.add_argument("--plot-dir", type=str, default="reports/hw_vs_sim_plots")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    rows = []
    for s in range(args.seeds):
        run_id = str(uuid.uuid4())[:8]
        LOG.info("=== RUN %s seed=%d qubits=%d ===", run_id, s, args.qubits)
        qc = make_test_ansatz(args.qubits, seed=s)
        # simulator
        try:
            sim_counts = run_simulator(qc, shots=args.shots)
        except Exception as e:
            LOG.error("Simulator run failed: %s", e)
            sim_counts = {}
        # hardware (use your wrapper)
        try:
            hw_out = run_small_ansatz_on_ibm(qc, shots=args.shots, backend=args.backend)
            hw_counts = hw_out.get("counts", {}) or {}
            job_id = hw_out.get("job_id")
        except Exception as e:
            LOG.error("Hardware run failed: %s", e)
            hw_counts = {}
            job_id = None

        stats = analyze_pair(sim_counts, hw_counts, shots=args.shots)
        # write debug JSON for this run
        debug = {
            "run_id": run_id,
            "seed": s,
            "qubits": args.qubits,
            "shots": args.shots,
            "backend": args.backend,
            "job_id": job_id,
            "stats": stats,
            "raw_sim_counts": sim_counts,
            "raw_hw_counts": hw_counts
        }
        debug_path = os.path.join(args.out_dir, f"hw_vs_sim_debug_{run_id}.json")
        dump_json(debug_path, debug)
        LOG.info("Wrote debug dump %s", debug_path)

        # produce histogram plot
        try:
            import numpy as np
            pa = np.array(stats["pa"])
            pb = np.array(stats["pb"])
            keys = stats["keys"]
            plot_path = os.path.join(args.plot_dir, f"hist_{run_id}.png")
            plot_histograms(pa, pb, keys, title=f"HW vs SIM run {run_id} seed={s}", savepath=plot_path)
            LOG.info("Wrote plot %s", plot_path)
        except Exception as e:
            LOG.warning("Plot failed: %s", e)

        # Append row summary for CSV
        row = {
            "run_id": run_id,
            "seed": s,
            "shots": args.shots,
            "backend": args.backend,
            "job_id": job_id,
            "js_div": stats["js_div"],
            "kl_avg": stats["kl_avg"],
            "l1": stats["l1"],
            "l2": stats["l2"],
            "classical_fidelity": stats["fidelity"],
            "sim_top_probs": json.dumps(stats["sim_top_probs"]),
            "hw_top_probs": json.dumps(stats["hw_top_probs"])
        }
        rows.append(row)

    csv_path = os.path.join(args.out_dir, "hw_vs_sim_results.csv")
    save_result_summary_csv(rows, csv_path)
    LOG.info("Wrote CSV summary to %s (rows=%d)", csv_path, len(rows))
    print("DONE")

if __name__ == "__main__":
    main()
