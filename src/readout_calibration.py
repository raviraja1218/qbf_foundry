# src/readout_calibration.py
"""
Readout calibration + mitigation for IBM runtime Sampler results.

Usage examples:
    from readout_calibration import calibration_for_backend, mitigate_counts_from_job

    # create service as in ibm_hardware_wrapper._create_service(...) or let the helper create it
    M, meta = calibration_for_backend(service=svc, backend_obj=backend_obj, qubits=2, shots=256)
    counts_mitigated = mitigate_counts(raw_counts, M, shots=128)

This file is written to be robust: it will attempt to call 'get_counts', 'get_int_counts',
or fall back to DataBin->meas.get_counts if present in the runtime result objects.
"""
import numpy as np
import math
import logging
import time
from typing import Dict, Tuple, Any, Optional, List

LOG = logging.getLogger("readout_cal")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Helper to build computational basis circuits for n qubits
def _basis_state_bitstrings(n: int) -> List[str]:
    return [format(i, f"0{n}b") for i in range(2 ** n)]

def _counts_to_prob_vec(counts: Dict[str, int], bitstrings: List[str], shots: int) -> np.ndarray:
    vec = np.zeros(len(bitstrings), dtype=float)
    for i, b in enumerate(bitstrings):
        vec[i] = counts.get(b, 0) / float(shots)
    return vec

def _prob_vec_to_counts(vec: np.ndarray, shots: int, bitstrings: List[str]) -> Dict[str, int]:
    # clip negatives, renormalize
    v = np.copy(vec)
    v[v < 0] = 0.0
    s = v.sum()
    if s <= 0:
        return {b: 0 for b in bitstrings}
    v = v / s
    # scale to shots and round with adjustment to make sum==shots
    scaled = v * shots
    rounded = np.floor(scaled).astype(int)
    delta = shots - rounded.sum()
    # distribute remaining delta to top fractional parts
    fracs = scaled - np.floor(scaled)
    order = np.argsort(-fracs)
    i = 0
    while delta > 0 and i < len(order):
        rounded[order[i]] += 1
        delta -= 1
        i += 1
    return {bitstrings[i]: int(rounded[i]) for i in range(len(bitstrings))}

# Robust extractor for executable result objects (DataBin/BitArray wrappers)
def extract_counts_from_result(result_obj: Any) -> Dict[str, int]:
    """
    Attempt many common access patterns to extract mapping bitstring->counts.
    Returns {} if none found.
    """
    # If mapping already
    if isinstance(result_obj, dict):
        # assume key->count or key->prob
        vals = list(result_obj.values())
        if vals and all(isinstance(v, (int,)) for v in vals):
            return {k: int(v) for k, v in result_obj.items()}
        # floats -> can't know shots; caller must pass shots
        return {k: int(round(float(v))) for k, v in result_obj.items()}

    # try common methods
    for name in ("get_counts", "get_int_counts", "counts", "get_counts_dict"):
        if hasattr(result_obj, name):
            try:
                fn = getattr(result_obj, name)
                val = fn() if callable(fn) else fn
                if isinstance(val, dict):
                    return {k: int(v) for k, v in val.items()}
            except Exception:
                pass

    # many DataBin-like objects have 'meas' attribute or .data().meas
    try:
        if hasattr(result_obj, "data"):
            d = result_obj.data() if callable(getattr(result_obj, "data")) else result_obj.data
            if d is not None and hasattr(d, "meas"):
                meas = d.meas
                # check meas.get_counts or meas.get_int_counts
                for mn in ("get_counts", "get_int_counts", "get_counts_map", "get_counts_dict"):
                    if hasattr(meas, mn):
                        try:
                            fn = getattr(meas, mn)
                            val = fn() if callable(fn) else fn
                            if isinstance(val, dict):
                                return {k: int(v) for k, v in val.items()}
                        except Exception:
                            pass
                # fallback: some meas objects have .as_array() or to_samples
                try:
                    if hasattr(meas, "get_bitstrings"):
                        # returns list/array of bitstrings for each shot -> compute counts
                        arr = meas.get_bitstrings()
                        if arr is not None:
                            # arr might be list of lists
                            bitstrings = ["".join(map(str, a)) if not isinstance(a, str) else a for a in arr]
                            counts = {}
                            for b in bitstrings:
                                counts[b] = counts.get(b, 0) + 1
                            return counts
                except Exception:
                    pass
    except Exception:
        pass
    return {}

def regularized_inverse(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    U, s, Vt = np.linalg.svd(M)
    s_inv = np.array([1.0 / (si + eps) for si in s])
    return (Vt.T @ np.diag(s_inv) @ U.T)

def calibration_for_backend(service: Any = None,
                            backend_obj: Any = None,
                            sampler: Any = None,
                            qubits: int = 2,
                            shots: int = 1024,
                            regularize_eps: float = 1e-6) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run calibration circuits on the provided backend/service/sampler and return confusion matrix M
    (shape 2^n x 2^n) and meta dictionary.
    Either pass (service & backend_obj) or pass a sampler instance (already constructed).
    """
    if sampler is None:
        if service is None or backend_obj is None:
            raise RuntimeError("Provide either sampler or (service and backend_obj).")
        # instantiate sampler similarly to ibm_hardware_wrapper
        try:
            from qiskit_ibm_runtime import Sampler, SamplerV2  # type: ignore
            try:
                sampler = Sampler(session=service, backend=backend_obj)
            except Exception:
                sampler = Sampler(backend=backend_obj)
        except Exception:
            # attempt SamplerV2 fallback
            try:
                from qiskit_ibm_runtime import SamplerV2  # type: ignore
                sampler = SamplerV2(backend_obj)
            except Exception as e:
                raise RuntimeError("Could not create sampler for calibration: " + str(e))

    # Build basis circuits (we will create simple circuits that prepare each computational basis state)
    from qiskit import QuantumCircuit  # type: ignore
    bitstrings = _basis_state_bitstrings(qubits)
    circuits = []
    for b in bitstrings:
        qc = QuantumCircuit(qubits)
        # prepare basis state: put X on bits with '1'
        for i, ch in enumerate(reversed(b)):  # Qiskit ordering, reverse to map index->bitstring
            if ch == "1":
                qc.x(i)
        qc.measure_all()
        circuits.append(qc)

    LOG.info("Submitting calibration job with %d circuits for %d qubits (shots=%d)", len(circuits), qubits, shots)
    # Submit as a single batch job if sampler.run accepts list
    try:
        job = sampler.run(circuits, shots=shots)
    except Exception as e:
        # try single-run per circuit fallback (slower)
        LOG.warning("Sampler.run(circuits) failed (%s), trying per-circuit submission", e)
        results = []
        for c in circuits:
            j = sampler.run([c], shots=shots)
            res = j.result()
            results.append(res)
        # Build confusion matrix from results list
        # Each result might be a PrimitiveResult -> iterable; handle with extract_counts_from_result
        M = np.zeros((len(bitstrings), len(bitstrings)), dtype=float)
        for i, res in enumerate(results):
            # res may be PrimitiveResult([SamplerPubResult(...)]); attempt extraction
            counts = extract_counts_from_result(res)
            # if counts empty, try iterating res
            if not counts and isinstance(res, (list, tuple)):
                for elem in res:
                    counts = extract_counts_from_result(elem)
                    if counts:
                        break
            prob_vec = _counts_to_prob_vec(counts, bitstrings, shots)
            M[:, i] = prob_vec
        invM = regularized_inverse(M, eps=regularize_eps)
        meta = {"method": "per_circuit_fallback", "qubits": qubits, "shots": shots}
        return M, {"M_inv": invM, "meta": meta}

    # blocking get result
    res = job.result()
    LOG.info("Calibration job finished, extracting counts")
    # res might be PrimitiveResult with many SamplerPubResult elements. Try to extract counts per circuit index.
    # Some runtime results return list-like container where each element corresponds to a circuit.
    counts_list = []
    try:
        # Typical: res.quasi_dists OR res[0], but safest is to iterate over res as sequence
        if isinstance(res, (list, tuple)):
            for elem in res:
                counts = extract_counts_from_result(elem)
                if not counts:
                    # try elem.data()
                    try:
                        d = elem.data() if callable(getattr(elem, "data")) else getattr(elem, "data", None)
                        counts = extract_counts_from_result(d)
                    except Exception:
                        counts = {}
                counts_list.append(counts)
        else:
            # If res is PrimitiveResult, try indexing or .data()
            try:
                # Some PrimitiveResult is iterable
                for elem in res:
                    counts = extract_counts_from_result(elem)
                    counts_list.append(counts)
            except Exception:
                # final fallback: try res.data()
                d = res.data() if callable(getattr(res, "data")) else getattr(res, "data", None)
                if isinstance(d, (list, tuple)):
                    for elem in d:
                        counts = extract_counts_from_result(elem)
                        counts_list.append(counts)
    except Exception as e:
        LOG.warning("Unexpected format for calibration result extraction: %s", e)

    # If still empty, try .quasi_dists
    if not counts_list:
        try:
            if hasattr(res, "quasi_dists"):
                qds = getattr(res, "quasi_dists")
                for qd in qds:
                    # qd might have probabilities_dict method or to_dict
                    # try qd.get_counts / qd.probabilities_dict
                    if hasattr(qd, "get_counts"):
                        c = qd.get_counts()
                        counts_list.append({k: int(v) for k, v in c.items()})
                    else:
                        d = None
                        for name in ("probabilities_dict", "probabilities", "to_dict", "binary_probabilities"):
                            if hasattr(qd, name):
                                try:
                                    val = getattr(qd, name)()
                                    if isinstance(val, dict):
                                        counts_list.append({k: int(round(float(v))) for k, v in val.items()})
                                        break
                                except Exception:
                                    pass
        except Exception:
            pass

    # Build matrix M
    n = len(bitstrings)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        counts = counts_list[i] if i < len(counts_list) else {}
        prob_vec = _counts_to_prob_vec(counts, bitstrings, shots)
        M[:, i] = prob_vec

    # regularized inverse
    invM = regularized_inverse(M, eps=regularize_eps)
    meta = {"method": "sampler_batch", "qubits": qubits, "shots": shots}
    LOG.info("Calibration matrix built (shape %s).", M.shape)
    return M, {"M_inv": invM, "meta": meta}

def mitigate_counts(raw_counts: Dict[str, int], M_inv: np.ndarray, shots: int, bitstrings: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Apply measurement mitigation to raw_counts using the inverse calibration matrix M_inv.
    bitstrings: expected ordering; if None infer ordering from M_inv shape (0..2^n-1)
    """
    n = M_inv.shape[0]
    if bitstrings is None:
        bitstrings = _basis_state_bitstrings(int(math.log2(n)))
    p_raw = _counts_to_prob_vec(raw_counts, bitstrings, shots)
    p_mitig = M_inv @ p_raw
    # clip small negatives and renormalize via prob->counts conversion
    mitig_counts = _prob_vec_to_counts(p_mitig, shots, bitstrings)
    return mitig_counts

# small helper: nicely print matrix (for debugging)
def pretty_matrix(M: np.ndarray) -> str:
    return "\n".join(["\t".join([f"{x:.4f}" for x in row]) for row in M])

if __name__ == "__main__":
    LOG.info("Module readout_calibration loaded — intended as importable helper.")
