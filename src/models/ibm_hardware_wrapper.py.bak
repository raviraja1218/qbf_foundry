"""
Small wrapper to run a single (small) ansatz circuit on IBM Quantum hardware
via qiskit-ibm-runtime. Robust extraction of counts from a variety of runtime
result shapes (PrimitiveResult / SamplerPubResult / DataBin, quasi_dists, etc).

Usage:
    python -m src.run_hardware_batch --limit 1 --backend ibm_torino --shots 128
"""
import os
import time
import json
import logging
from typing import Dict, Any, List, Iterable, Tuple, Optional

LOG = logging.getLogger("ibm_hw")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Try guarded imports so this module fails gracefully if qiskit not installed
IBM_AVAILABLE = False
_qiskit_info: Dict[str, Any] = {}
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, SamplerV2  # type: ignore
    from qiskit import QuantumCircuit, transpile  # type: ignore
    IBM_AVAILABLE = True
    _qiskit_info["qiskit_ibm_runtime"] = True
    _qiskit_info["qiskit"] = True
except Exception as e:
    LOG.debug("[ibm_hw] guarded import failed: %s", e)
    # Collect available metadata if possible
    try:
        import importlib.metadata as _md  # type: ignore
        for pkg in ("qiskit", "qiskit-ibm-runtime", "qiskit-ibm-provider"):
            try:
                _qiskit_info[pkg + "_version"] = _md.version(pkg)
            except Exception:
                _qiskit_info[pkg + "_version"] = None
    except Exception:
        pass


def check_qiskit_env() -> Dict[str, Any]:
    """
    Return a small dict describing available qiskit packages (helpful for debug/log).
    """
    info = dict(_qiskit_info)
    try:
        import importlib.metadata as _md  # type: ignore
        for pkg in ("qiskit", "qiskit-ibm-runtime", "qiskit-ibm-provider"):
            try:
                info[f"{pkg}_version"] = _md.version(pkg)
            except Exception:
                info[f"{pkg}_version"] = None
    except Exception:
        pass
    return info


# ---------- small helpers ----------
def _is_mapping_like(x) -> bool:
    return isinstance(x, dict)


def _try_call(obj, attr: str):
    """
    Call obj.attr() if callable or return obj.attr if present.
    """
    if not hasattr(obj, attr):
        return None
    a = getattr(obj, attr)
    if callable(a):
        try:
            return a()
        except Exception:
            # sometimes attribute is callable but requires args — fall back to returning the attribute object
            return a
    return a


def _scale_probabilities_to_counts(prob_map: Dict[str, float], shots: int) -> Dict[str, int]:
    """
    Convert a probability mapping (bitstring -> probability) to integer counts
    that sum approximately to shots.
    """
    if not prob_map:
        return {}
    scaled = {k: float(v) * shots for k, v in prob_map.items()}
    rounded = {k: int(round(v)) for k, v in scaled.items()}
    total = sum(rounded.values())
    delta = shots - total
    if delta != 0:
        # allocate delta based on fractional parts
        frac_items = []
        for k, v in prob_map.items():
            frac = (v * shots) - int(v * shots)
            frac_items.append((frac, k))
        frac_items.sort(reverse=(delta > 0))
        i = 0
        while delta != 0 and i < len(frac_items):
            _, key = frac_items[i]
            rounded[key] = rounded.get(key, 0) + (1 if delta > 0 else -1)
            delta = shots - sum(rounded.values())
            i += 1
    for k in list(rounded.keys()):
        if rounded[k] < 0:
            rounded[k] = 0
    return rounded


def _unpack_bytes_to_bitstr(b: bytes, nbits: int, msb_first: bool = True) -> str:
    """
    Convert bytes to a flat bitstring of length nbits. msb_first chooses bit ordering.
    """
    if not b:
        return ""
    # build bits from bytes
    bits = []
    for byte in b:
        for i in range(8):
            if msb_first:
                bits.append("1" if (byte & (1 << (7 - i))) else "0")
            else:
                bits.append("1" if (byte & (1 << i)) else "0")
    flat = "".join(bits)[:nbits]
    return flat


def _rows_from_flat_bitstr(flat: str, shots: int, nbits: int) -> List[str]:
    if not flat or shots <= 0 or nbits <= 0:
        return []
    if len(flat) != shots * nbits:
        return []
    rows = [flat[i * nbits:(i + 1) * nbits] for i in range(shots)]
    return rows


def _rows_from_array_like(arr_like, shots: Optional[int], nbits: Optional[int]) -> List[str]:
    """
    Try to interpret numpy-like / list-like object as per-shot booleans or ints.
    Returns list of bitstring rows or empty list.
    """
    try:
        # treat as 2D array where outer dimension = shots
        # common shapes: (shots, nbits) with booleans/0/1 or (nbits, shots)
        import numpy as _np  # type: ignore

        a = _np.array(arr_like)
        if a.size == 0:
            return []
        # If shape equals (shots, nbits)
        if shots and nbits and a.shape == (shots, nbits):
            rows = []
            for r in a:
                bits = "".join("1" if bool(x) else "0" for x in r)
                rows.append(bits)
            return rows
        # If shape equals (nbits, shots) — transpose
        if shots and nbits and a.shape == (nbits, shots):
            a2 = a.T
            rows = []
            for r in a2:
                bits = "".join("1" if bool(x) else "0" for x in r)
                rows.append(bits)
            return rows
        # If flattened and length == shots*nbits
        flat = "".join("1" if bool(x) else "0" for x in a.flatten())
        if shots and nbits and len(flat) == shots * nbits:
            return _rows_from_flat_bitstr(flat, shots, nbits)
    except Exception:
        # fallback: try list of strings
        try:
            if isinstance(arr_like, (list, tuple)) and arr_like and all(isinstance(x, str) for x in arr_like):
                return list(arr_like)
        except Exception:
            pass
    return []


def _detect_num_shots_bits(meas_obj) -> Tuple[Optional[int], Optional[int]]:
    """Try to detect num_shots and num_bits from object attributes."""
    ns = None
    nb = None
    try:
        if hasattr(meas_obj, "num_shots"):
            try:
                ns = int(meas_obj.num_shots)
            except Exception:
                try:
                    ns = int(meas_obj.num_shots())
                except Exception:
                    ns = None
    except Exception:
        ns = None
    try:
        if hasattr(meas_obj, "num_bits"):
            try:
                nb = int(meas_obj.num_bits)
            except Exception:
                try:
                    nb = int(meas_obj.num_bits())
                except Exception:
                    nb = None
    except Exception:
        nb = None
    # fallback: try shape attribute
    try:
        shp = getattr(meas_obj, "shape", None)
        if shp:
            try:
                if isinstance(shp, tuple):
                    # shape like (shots, bits) or (bits, shots)
                    if len(shp) >= 2:
                        a, b = int(shp[0]), int(shp[1])
                        # heuristics: if both small choose a as shots
                        ns = ns or a
                        nb = nb or b
            except Exception:
                pass
    except Exception:
        pass
    return ns, nb


# ---------- Quasi-dist extraction ----------
def _extract_from_quasidist(qd, shots: int):
    """
    Try common quasi-dist interfaces (binary_probabilities, probabilities_dict, to_dict).
    Return dict or None.
    """
    try:
        if hasattr(qd, "binary_probabilities"):
            try:
                val = qd.binary_probabilities()
                if _is_mapping_like(val):
                    s = sum(val.values()) if val else 0
                    if 0.999 <= s <= 1.001:
                        return _scale_probabilities_to_counts(val, shots)
                    return {k: int(round(v)) for k, v in val.items()}
            except Exception:
                pass
        for attr in ("probabilities_dict", "probabilities", "to_dict"):
            val = _try_call(qd, attr)
            if _is_mapping_like(val):
                s = sum(val.values()) if val else 0
                if 0.999 <= s <= 1.001:
                    return _scale_probabilities_to_counts(val, shots)
                if all(isinstance(x, int) for x in val.values()):
                    return {k: int(v) for k, v in val.items()}
                return {k: int(round(float(v))) for k, v in val.items()}
    except Exception:
        pass
    return None


# ---------- Deep meas methods (targeted for BitArray/DataBin.meas) ----------
def _deep_try_meas_methods(mv, shots: int, debug_summary: Dict[str, Any]) -> Dict[str, int]:
    """
    Try candidate methods/attributes on meas (BitArray-like). Records debug info and
    returns counts if successful.
    """
    try:
        debug_summary["meas_dir"] = sorted([a for a in dir(mv) if not a.startswith("_")])
    except Exception:
        debug_summary["meas_dir"] = ["<dir-failed>"]
    try:
        debug_summary["meas_repr"] = repr(mv)[:2000]
    except Exception:
        debug_summary["meas_repr"] = "<repr-failed>"

    num_shots, num_bits = _detect_num_shots_bits(mv)
    debug_summary["num_shots_detected"] = num_shots
    debug_summary["num_bits_detected"] = num_bits

    # 1) get_counts()
    try:
        if hasattr(mv, "get_counts"):
            try:
                gc = mv.get_counts()
            except Exception:
                try:
                    gc = mv.get_counts() if callable(getattr(mv, "get_counts")) else None
                except Exception:
                    gc = None
            if isinstance(gc, dict) and gc:
                try:
                    out = {str(k): int(v) for k, v in gc.items()}
                    debug_summary["method"] = "meas.get_counts"
                    return out
                except Exception:
                    pass
    except Exception:
        pass

    # 2) get_int_counts()
    try:
        if hasattr(mv, "get_int_counts"):
            try:
                ic = mv.get_int_counts()
            except Exception:
                ic = None
            if isinstance(ic, dict) and ic:
                if num_bits:
                    out = {}
                    for k, v in ic.items():
                        b = format(int(k), "b").rjust(num_bits, "0")
                        out[b] = out.get(b, 0) + int(v)
                    debug_summary["method"] = "meas.get_int_counts"
                    return out
    except Exception:
        pass

    # 3) get_bitstrings()
    try:
        if hasattr(mv, "get_bitstrings"):
            try:
                bs = mv.get_bitstrings()
            except Exception:
                try:
                    bs = mv.get_bitstrings() if callable(getattr(mv, "get_bitstrings")) else None
                except Exception:
                    bs = None
            if isinstance(bs, (list, tuple)) and bs:
                if all(isinstance(x, str) for x in bs):
                    counts = {}
                    for r in bs:
                        counts[r] = counts.get(r, 0) + 1
                    debug_summary["method"] = "meas.get_bitstrings"
                    return counts
                if all(isinstance(x, (bytes, bytearray)) for x in bs) and num_bits and num_shots:
                    rows = []
                    for b in bs:
                        rows.append(_unpack_bytes_to_bitstr(bytes(b), num_bits, msb_first=True))
                    counts = {}
                    for r in rows:
                        counts[r] = counts.get(r, 0) + 1
                    debug_summary["method"] = "meas.get_bitstrings_bytes"
                    return counts
    except Exception:
        pass

    # 4) to_bool_array()
    try:
        if hasattr(mv, "to_bool_array"):
            try:
                tb = mv.to_bool_array()
            except Exception:
                try:
                    tb = mv.to_bool_array() if callable(getattr(mv, "to_bool_array")) else None
                except Exception:
                    tb = None
            if tb is not None:
                rows = _rows_from_array_like(tb, num_shots or shots, num_bits or None)
                if rows:
                    counts = {}
                    for r in rows:
                        counts[r] = counts.get(r, 0) + 1
                    debug_summary["method"] = "meas.to_bool_array"
                    return counts
    except Exception:
        pass

    # Generic candidate methods (best-effort fallback)
    candidate_methods = [
        "to01s", "to01", "to01_list", "to_bitstrings", "to_bitstring", "to_strings", "to_list",
        "tolist", "as_array", "as_numpy", "as_uints", "asbytes", "tobytes", "to_bytes",
        "raw", "bits", "as_binary", "as_bitarray", "to_vector", "to_array", "to_ndarray", "to_numpy",
        "asarray", "as_array", "to_numpy_array", "asbytes_le", "asbytes_be", "get_samples", "get_counts"
    ]
    for meth in candidate_methods:
        if hasattr(mv, meth):
            try:
                val = getattr(mv, meth)
                ret = val() if callable(val) else val
            except Exception as e:
                debug_summary.setdefault("meas_method_errors", {})[meth] = repr(e)[:400]
                continue

            # str handling
            if isinstance(ret, str):
                flat = ret.strip()
                if "\n" in flat or " " in flat or "," in flat:
                    parts = [p for p in (flat.replace(",", " ").replace("\n", " ").split()) if p]
                    if parts:
                        counts = {}
                        for r in parts:
                            counts[r] = counts.get(r, 0) + 1
                        debug_summary["method"] = f"meas.{meth}()_split"
                        return counts
                if num_shots and num_bits and len(flat) == num_shots * num_bits:
                    rows = _rows_from_flat_bitstr(flat, num_shots, num_bits)
                    if rows:
                        counts = {}
                        for r in rows:
                            counts[r] = counts.get(r, 0) + 1
                        debug_summary["method"] = f"meas.{meth}()_flat"
                        return counts
                if len(flat) == num_bits and (num_shots == 1 or not num_shots):
                    debug_summary["method"] = f"meas.{meth}()_single"
                    return {flat: 1}

            # bytes handling
            if isinstance(ret, (bytes, bytearray)):
                if not num_shots or not num_bits:
                    ns, nb = _detect_num_shots_bits(mv)
                    num_shots = num_shots or ns
                    num_bits = num_bits or nb
                if num_shots and num_bits:
                    flat = _unpack_bytes_to_bitstr(bytes(ret), num_shots * num_bits, msb_first=True)
                    rows = _rows_from_flat_bitstr(flat, num_shots, num_bits)
                    if rows:
                        counts = {r: rows.count(r) for r in set(rows)}
                        debug_summary["method"] = f"meas.{meth}()_bytes_msb"
                        return counts
                    flat2 = _unpack_bytes_to_bitstr(bytes(ret), num_shots * num_bits, msb_first=False)
                    rows2 = _rows_from_flat_bitstr(flat2, num_shots, num_bits)
                    if rows2:
                        counts = {r: rows2.count(r) for r in set(rows2)}
                        debug_summary["method"] = f"meas.{meth}()_bytes_lsb"
                        return counts

            # list/tuple of strings
            if isinstance(ret, (list, tuple)) and ret and all(isinstance(x, str) for x in ret):
                if num_bits and all(len(x) == num_bits for x in ret):
                    counts = {}
                    for r in ret:
                        counts[r] = counts.get(r, 0) + 1
                    debug_summary["method"] = f"meas.{meth}()_list_of_str"
                    return counts

            rows = _rows_from_array_like(ret, num_shots or shots, num_bits or None)
            if rows:
                counts = {}
                for r in rows:
                    counts[r] = counts.get(r, 0) + 1
                debug_summary["method"] = f"meas.{meth}()_array_like"
                return counts

    # try attr 'bits'
    if hasattr(mv, "bits"):
        try:
            bits_val = mv.bits
            rows = _rows_from_array_like(bits_val, num_shots or shots, num_bits or None)
            if rows:
                counts = {}
                for r in rows:
                    counts[r] = counts.get(r, 0) + 1
                debug_summary["method"] = "meas.bits_attr"
                return counts
        except Exception:
            pass

    # fallback: quasi-dist interface on mv
    extracted = _extract_from_quasidist(mv, shots)
    if extracted:
        debug_summary["method"] = "meas.quasidist"
        return extracted

    return {}


def _safe_extract_counts(result_obj, shots: int, debug_summary: Dict[str, Any]) -> Dict[str, int]:
    """
    Try layered extraction of bitstring->counts from a runtime 'result' object.
    """
    LOG.info("[ibm_hw] Inspecting result type=%s", type(result_obj))

    # 1) If result is iterable (PrimitiveResult), try each element
    try:
        if isinstance(result_obj, (list, tuple, Iterable)) and not isinstance(result_obj, dict):
            for elem in result_obj:
                try:
                    counts = _safe_extract_counts(elem, shots, debug_summary)
                    if counts:
                        return counts
                except Exception:
                    continue
    except Exception:
        pass

    # 2) If has 'quasi_dists' attribute: inspect first quasi-dist
    try:
        qds = getattr(result_obj, "quasi_dists", None)
        if qds:
            try:
                if len(qds) > 0:
                    qd = qds[0]
                    extracted = _extract_from_quasidist(qd, shots)
                    if extracted:
                        debug_summary["method"] = "result.quasi_dists"
                        return extracted
            except Exception:
                pass
    except Exception:
        pass

    # 3) Check result.data (SamplerPubResult has .data -> DataBin)
    try:
        data_attr = getattr(result_obj, "data", None)
        if data_attr is not None:
            try:
                data_val = data_attr() if callable(data_attr) else data_attr
            except Exception:
                data_val = data_attr
            # data_val can be dict or DataBin-like
            if isinstance(data_val, dict):
                for key_candidate in ("probabilities", "counts", "probabilities_dict"):
                    if key_candidate in data_val and isinstance(data_val[key_candidate], dict):
                        inner = data_val[key_candidate]
                        s = sum(inner.values()) if inner else 0
                        if 0.999 <= s <= 1.001:
                            debug_summary["method"] = f"data.{key_candidate}"
                            return _scale_probabilities_to_counts({kk: float(vv) for kk, vv in inner.items()}, shots)
                        return {kk: int(round(float(vv))) for kk, vv in inner.items()}
                # data_val might directly be mapping
                vals = list(data_val.values())
                if vals and all(isinstance(v, int) for v in vals):
                    debug_summary["method"] = "data.dict_ints"
                    return {k: int(v) for k, v in data_val.items()}
                if vals and all(isinstance(v, float) for v in vals):
                    s = sum(vals)
                    if 0.999 <= s <= 1.001:
                        debug_summary["method"] = "data.dict_probs"
                        return _scale_probabilities_to_counts({k: float(v) for k, v in data_val.items()}, shots)
                    return {k: int(round(float(v))) for k, v in data_val.items()}
            else:
                # data_val may be DataBin-like with .meas
                mv = getattr(data_val, "meas", None)
                if mv is not None:
                    extracted = _deep_try_meas_methods(mv, shots, debug_summary)
                    if extracted:
                        return extracted
                # as fallback try quasidist extraction on data_val
                extracted = _extract_from_quasidist(data_val, shots)
                if extracted:
                    debug_summary["method"] = "data_val.quasidist"
                    return extracted
    except Exception:
        pass

    # 4) result.get_counts()
    try:
        if hasattr(result_obj, "get_counts"):
            try:
                counts = result_obj.get_counts()
            except Exception:
                try:
                    counts = result_obj.get_counts() if callable(getattr(result_obj, "get_counts")) else None
                except Exception:
                    counts = None
            if isinstance(counts, dict):
                vals = list(counts.values())
                if vals and any(isinstance(v, float) for v in vals):
                    s = sum(vals)
                    if 0.999 <= s <= 1.001:
                        debug_summary["method"] = "result.get_counts_probs"
                        return _scale_probabilities_to_counts({k: float(v) for k, v in counts.items()}, shots)
                    return {k: int(round(float(v))) for k, v in counts.items()}
                debug_summary["method"] = "result.get_counts"
                return {k: int(v) for k, v in counts.items()}
    except Exception:
        pass

    # 5) result.to_dict()
    try:
        if hasattr(result_obj, "to_dict"):
            try:
                rd = result_obj.to_dict()
            except Exception:
                rd = None
            if isinstance(rd, dict):
                for k in ("counts", "quasi_dists", "probabilities", "probabilities_dict"):
                    if k in rd and isinstance(rd[k], dict):
                        inner = rd[k]
                        vals = list(inner.values())
                        if vals and any(isinstance(v, float) for v in vals):
                            s = sum(vals)
                            if 0.999 <= s <= 1.001:
                                debug_summary["method"] = f"to_dict.{k}_probs"
                                return _scale_probabilities_to_counts({kk: float(vv) for kk, vv in inner.items()}, shots)
                            return {kk: int(round(float(vv))) for kk, vv in inner.items()}
                        debug_summary["method"] = f"to_dict.{k}"
                        return {kk: int(vv) for kk, vv in inner.items()}
                if "quasi_dists" in rd and isinstance(rd["quasi_dists"], (list, tuple)) and rd["quasi_dists"]:
                    inner = rd["quasi_dists"][0]
                    if isinstance(inner, dict):
                        for ik in ("probabilities", "probabilities_dict", "binary_probabilities"):
                            if ik in inner and isinstance(inner[ik], dict):
                                ip = inner[ik]
                                s = sum(ip.values()) if ip else 0
                                if 0.999 <= s <= 1.001:
                                    debug_summary["method"] = "to_dict.quasi_probs"
                                    return _scale_probabilities_to_counts({k: float(v) for k, v in ip.items()}, shots)
                                return {k: int(round(float(v))) for k, v in ip.items()}
    except Exception:
        pass

    # 6) result.data() fallback
    try:
        if hasattr(result_obj, "data"):
            try:
                rd = result_obj.data()
            except Exception:
                try:
                    rd = result_obj.data() if callable(getattr(result_obj, "data")) else None
                except Exception:
                    rd = None
            if isinstance(rd, dict):
                for k in ("counts", "probabilities"):
                    if k in rd and isinstance(rd[k], dict):
                        inner = rd[k]
                        vals = list(inner.values())
                        if vals and any(isinstance(v, float) for v in vals):
                            s = sum(vals)
                            if 0.999 <= s <= 1.001:
                                debug_summary["method"] = "result.data_probs"
                                return _scale_probabilities_to_counts({kk: float(vv) for kk, vv in inner.items()}, shots)
                            return {kk: int(round(float(vv))) for kk, vv in inner.items()}
                        debug_summary["method"] = "result.data_counts"
                        return {kk: int(vv) for kk, vv in inner.items()}
    except Exception:
        pass

    LOG.debug("[ibm_hw] No counts found in result object of type %s", type(result_obj))
    return {}


# ---------- Main runner ----------
def _create_service(token: str, channel: str = "ibm_cloud"):
    if not IBM_AVAILABLE:
        raise RuntimeError(
            "qiskit_ibm_runtime / qiskit not available in this Python environment. "
            "Install qiskit, qiskit-ibm-runtime and qiskit-ibm-provider in the same env."
        )
    try:
        svc = QiskitRuntimeService(channel=channel, token=token)
        LOG.info("[ibm_hw] Created QiskitRuntimeService via (channel, token).")
        return svc
    except TypeError:
        svc = QiskitRuntimeService(token=token)
        LOG.info("[ibm_hw] Created QiskitRuntimeService via (token).")
        return svc


def run_small_ansatz_on_ibm(ansatz: "QuantumCircuit", shots: int = 1024, backend: str = "ibm_torino",
                            optimization_level: int = 1, transpile_try: bool = True,
                            debug_dump_path: Optional[str] = "ibm_result_summary.json") -> Dict[str, Any]:
    """
    Submit a small ansatz to IBM runtime and return normalized integer counts.

    Returns dict:
      { "counts": {bitstring: int_count, ...},
        "backend": backend,
        "shots": shots,
        "timestamp": time.time(),
        "job_id": str or None,
        "raw_result": result_object,
        "debug_summary": { ... }  # diagnostic info
      }
    """
    LOG.info("[ibm_hw] check qiskit env: %s", check_qiskit_env())

    if not IBM_AVAILABLE:
        raise RuntimeError("qiskit / qiskit_ibm_runtime not importable in this environment.")

    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"Set environment var {token_env} with your IBM Quantum token.")

    service = _create_service(token=token, channel=channel)

    # get backend object
    try:
        backend_obj = service.backend(backend)
    except Exception as e:
        LOG.error("[ibm_hw] could not obtain backend %s: %s", backend, e)
        raise

    # transpile for the backend if possible
    circ_to_run = ansatz
    try:
        if transpile_try:
            circ_to_run = transpile(ansatz, backend=backend_obj, optimization_level=optimization_level)
            LOG.info("[ibm_hw] Transpiled circuit for backend %s", backend)
    except Exception as e:
        LOG.warning("[ibm_hw] transpile failed, proceeding with original circuit: %s", e)
        circ_to_run = ansatz

    # ensure measurements
    try:
        if getattr(circ_to_run, "num_clbits", 0) == 0 or circ_to_run.count_ops().get("measure", 0) == 0:
            circ_to_run = circ_to_run.copy()
            circ_to_run.measure_all()
            LOG.info("[ibm_hw] added measure_all() to circuit")
    except Exception:
        try:
            circ_to_run = circ_to_run.copy()
            circ_to_run.measure_all()
            LOG.info("[ibm_hw] added measure_all() (fallback)")
        except Exception as e:
            LOG.warning("[ibm_hw] could not add measurements: %s", e)

    pubs = [circ_to_run]

    # instantiate sampler with multiple fallbacks
    sampler = None
    sampler_errs: List[Exception] = []
    try:
        sampler = Sampler(session=service, backend=backend_obj)
        LOG.info("[ibm_hw] Sampler(session=service, backend=backend_obj) created")
    except Exception as e:
        sampler_errs.append(e)
        try:
            sampler = Sampler(backend=backend_obj)
            LOG.info("[ibm_hw] Sampler(backend=backend_obj) created")
        except Exception as e2:
            sampler_errs.append(e2)
            try:
                sampler = SamplerV2(backend_obj)  # type: ignore
                LOG.info("[ibm_hw] SamplerV2(backend_obj) created")
            except Exception as e3:
                sampler_errs.append(e3)
                LOG.error("[ibm_hw] sampler instantiation failed: %s", sampler_errs)
                raise RuntimeError(f"Could not instantiate Sampler: {sampler_errs}")

    # try running sampler with multiple call styles
    job = None
    run_errs: List[Exception] = []
    try:
        job = sampler.run(pubs, shots=shots)
    except Exception as e:
        run_errs.append(e)
        try:
            job = sampler.run([circ_to_run], shots=shots)
        except Exception as e2:
            run_errs.append(e2)
            try:
                job = sampler.run(pubs)
            except Exception as e3:
                run_errs.append(e3)
                LOG.error("[ibm_hw] sampler.run failed with styles: %s", run_errs)
                raise RuntimeError(f"Sampler.run failed: {run_errs}")

    # get result (blocking)
    try:
        result = job.result()
    except Exception as e:
        LOG.error("[ibm_hw] job.result() error: %s", e)
        raise

    # Try to extract job id robustly (callable or attribute)
    _job_id = None
    try:
        jid = getattr(job, "job_id", None)
        if callable(jid):
            try:
                _job_id = jid()
            except Exception:
                _job_id = None
        else:
            _job_id = jid or getattr(job, "id", None)
            if callable(_job_id):
                try:
                    _job_id = _job_id()
                except Exception:
                    _job_id = None
    except Exception:
        _job_id = None

    debug_summary: Dict[str, Any] = {
        "backend": backend,
        "job_id": _job_id,
        "result_type": repr(type(result)),
        "method": None,
    }

    # Extract counts using robust logic
    counts = _safe_extract_counts(result, shots=shots, debug_summary=debug_summary)

    # If counts empty, try deeper iteration over result elements
    if not counts:
        try:
            if isinstance(result, Iterable) and not isinstance(result, dict):
                for elem in result:
                    try:
                        counts = _safe_extract_counts(elem, shots=shots, debug_summary=debug_summary)
                        if counts:
                            break
                    except Exception:
                        continue
        except Exception:
            pass

    # If still empty, attempt to deep-inspect any DataBin-like 'data().meas' nested path
    if not counts:
        try:
            data_attr = getattr(result, "data", None)
            if data_attr is not None:
                try:
                    data_val = data_attr() if callable(data_attr) else data_attr
                except Exception:
                    data_val = data_attr
                mv = getattr(data_val, "meas", None) or getattr(data_val, "measures", None)
                if mv is not None:
                    counts = _deep_try_meas_methods(mv, shots, debug_summary)
        except Exception:
            pass

    # If counts still empty, log debug info to file for offline inspection
    if debug_dump_path:
        try:
            summary = {
                "backend": backend,
                "job_id": _job_id,
                "result_type": repr(type(result)),
                "result_summary": repr(result)[:4000],
                "counts_extracted": bool(counts),
                "counts_len": len(counts) if counts else 0,
                "debug_summary": debug_summary,
            }
            with open(debug_dump_path, "w") as f:
                json.dump(summary, f, indent=2)
            LOG.info("[ibm_hw] Debug dump written to %s", debug_dump_path)
        except Exception as e:
            LOG.warning("[ibm_hw] debug dump failed: %s", e)

    # Trim returned bitstrings to logical qubit count if needed
    try:
        logical_n = getattr(ansatz, "num_qubits", None)
        detected_bits = debug_summary.get("num_bits_detected", None)
        if counts and logical_n and detected_bits and detected_bits > logical_n:
            # assume logical qubits are in the rightmost bits (heuristic)
            trimmed: Dict[str, int] = {}
            for k, v in counts.items():
                if len(k) >= logical_n:
                    newk = k[-logical_n:]
                else:
                    newk = k.rjust(logical_n, "0")
                trimmed[newk] = trimmed.get(newk, 0) + int(v)
            debug_summary["trimmed_from"] = detected_bits
            debug_summary["trim_method"] = "rightmost_slice"
            counts = trimmed
    except Exception:
        pass

    out = {
        "counts": counts,
        "backend": backend,
        "shots": shots,
        "timestamp": time.time(),
        "job_id": _job_id,
        "raw_result": result,
        "debug_summary": debug_summary,
    }

    LOG.info("[ibm_hw] Job finished: backend=%s shots=%s job_id=%s counts_len=%d",
             out["backend"], out["shots"], out["job_id"], len(counts) if counts else 0)
    return out
