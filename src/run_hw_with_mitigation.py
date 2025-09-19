# src/run_hw_with_mitigation.py
"""
Batch-run hardware circuits and apply readout mitigation.

Usage:
    python src/run_hw_with_mitigation.py --backend ibm_torino --shots 256 --n_runs 5 --calfile reports/calibration_M_q2_shots256.npz
"""
import os
import json
import time
import argparse
import numpy as np
import logging
from qiskit import QuantumCircuit
from src.models.ibm_hardware_wrapper import run_small_ansatz_on_ibm, check_qiskit_env
from src.readout_calibration import mitigate_counts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("run_hw_mitigate")

def build_bell_circuit(n_qubits=2):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def counts_to_probdist(counts, shots):
    # return numpy array ordered by lexicographic bitstrings (msb..lsb)
    nbits = int(np.log2(sum([1<<0 for _ in range(1)]) or 1))  # dummy fallback
    # better: infer nbits from keys
    if counts:
        nbits = max(len(k) for k in counts.keys())
    all_keys = sorted(counts.keys(), reverse=False)
    vec = np.array([counts.get(k, 0) / shots for k in all_keys], dtype=float)
    return all_keys, vec

def plot_hist(counts, outpath):
    # simple matplotlib histogram (no color styling)
    import matplotlib.pyplot as plt
    keys = list(counts.keys())
    vals = [counts[k] for k in keys]
    plt.figure(figsize=(6,3))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=60)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="ibm_torino")
    p.add_argument("--shots", type=int, default=256)
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--qubits", type=int, default=2)
    p.add_argument("--calfile", default="reports/calibration_M_q2_shots256.npz")
    p.add_argument("--outdir", default="reports/mitigated_hw_runs")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    LOG.info("qiskit env: %s", check_qiskit_env())

    # load calibration
    if not os.path.exists(args.calfile):
        raise SystemExit(f"Calibration file not found: {args.calfile}")
    dat = np.load(args.calfile, allow_pickle=True)
    if "M_inv" in dat:
        M_inv = dat["M_inv"]
    else:
        meta = dat.get("meta")
        if meta is not None:
            try:
                meta_dict = meta.item() if hasattr(meta, "item") else dict(meta)
            except Exception:
                meta_dict = dict(meta)
            M_inv = meta_dict.get("M_inv")
        else:
            M_inv = None
    if M_inv is None:
        raise SystemExit("M_inv not found in calibration file")

    rows = []
    for run_i in range(args.n_runs):
        seed = int(np.random.randint(0, 2**31-1))
        LOG.info("=== RUN %02d seed=%s qubits=%d shots=%d ===", run_i, seed, args.qubits, args.shots)
        qc = build_bell_circuit(n_qubits=args.qubits)
        start = time.time()
        out = run_small_ansatz_on_ibm(qc, shots=args.shots, backend=args.backend)
        elapsed = time.time() - start

        job_id = out.get("job_id")
        raw_counts = out.get("counts", {})
        mitigated = mitigate_counts(raw_counts, M_inv, shots=args.shots)

        # write per-run debug
        run_name = f"run_{run_i:02d}_{job_id or 'nojid'}"
        debug = {
            "run": run_i,
            "seed": seed,
            "backend": args.backend,
            "shots": args.shots,
            "job_id": job_id,
            "elapsed_sec": elapsed,
            "raw_counts": raw_counts,
            "mitigated_counts": mitigated,
            "debug_summary": out.get("debug_summary")
        }
        debug_path = os.path.join(args.outdir, f"debug_{run_name}.json")
        save_json(debug_path, debug)
        # plot histograms
        try:
            plot_hist(raw_counts, os.path.join(args.outdir, f"hist_raw_{run_name}.png"))
            plot_hist(mitigated, os.path.join(args.outdir, f"hist_mitigated_{run_name}.png"))
        except Exception as e:
            LOG.warning("Plotting failed: %s", e)

        # summary row
        rows.append({
            "run": run_i,
            "job_id": job_id,
            "seed": seed,
            "shots": args.shots,
            "raw_counts_len": len(raw_counts),
            "mitigated_counts_len": len(mitigated),
            "elapsed_sec": elapsed,
            "debug_json": debug_path
        })

    # save CSV summary
    import csv
    csv_path = os.path.join(args.outdir, "summary.csv")
    keys = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    LOG.info("Wrote summary CSV to %s", csv_path)

if __name__ == "__main__":
    main()
