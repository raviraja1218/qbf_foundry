# src/run_quantum_opt.py
"""
Run quantum optimization for one sample (or batch) and save results safely.

Usage:
    export PYTHONPATH="$PWD"
    conda activate qbf_clean
    python -m src.run_quantum_opt --n_trials 200 --device cuda
"""
import glob
import time
import logging
import inspect
from pathlib import Path
import argparse
import csv
from typing import Callable, Optional

import numpy as np
import torch

# Project imports (adjust paths if you moved files)
from src.models.classical_nn import ProteinVAE
from src.models.quantum_module import optimize_latent_with_bioscore  # optimizer entrypoint
from src.utils.scoring import score_sequence_combined  # scoring utility
from src.utils.aa_map import decode_indices_to_seq  # convert indices -> sequence string

LOG = logging.getLogger("run_quantum_opt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = "models/interim_models/protein_vae.pt"
DATA_GLOB = "data/processed/*.npz"
OUT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Loader / shape-adapt utilities (kept from previous)
# -------------------------
def _make_tensor_from(checkpoint_tensor, target_shape, device, name):
    target = torch.empty(target_shape, device=device)
    torch.nn.init.normal_(target, mean=0.0, std=0.02)
    ck = torch.as_tensor(checkpoint_tensor, dtype=target.dtype, device=device)
    overlap = tuple(min(a, b) for a, b in zip(ck.shape, target.shape))
    if any(s == 0 for s in overlap):
        LOG.info(f"[adapt] No overlap for '{name}' {ck.shape} -> {target.shape}, leaving init")
        return target
    ck_slices = tuple(slice(0, s) for s in overlap)
    tgt_slices = tuple(slice(0, s) for s in overlap)
    target[tgt_slices] = ck[ck_slices]
    LOG.info(f"[adapt] Copied overlap for '{name}' shape {tuple(ck.shape)} -> {tuple(target.shape)}")
    return target


def adapt_and_load_state_dict(model, ckpt_state_dict, device="cpu"):
    model_state = model.state_dict()
    new_state = {}
    missing = []
    unexpected = []

    for k_ck, v_ck in ckpt_state_dict.items():
        if k_ck not in model_state:
            unexpected.append(k_ck)
            LOG.warning(f"[adapt] Key in checkpoint not in model: {k_ck} (skipping)")
            continue

        v_model = model_state[k_ck]
        if isinstance(v_ck, np.ndarray):
            ck_tensor = v_ck
        elif torch.is_tensor(v_ck):
            ck_tensor = v_ck.cpu().numpy()
        else:
            try:
                ck_tensor = np.asarray(v_ck)
            except Exception:
                ck_tensor = None

        if ck_tensor is None:
            LOG.warning(f"[adapt] Unable to interpret checkpoint tensor for key {k_ck}; skipping")
            missing.append(k_ck)
            continue

        if ck_tensor.shape == tuple(v_model.shape):
            new_state[k_ck] = torch.as_tensor(ck_tensor, device=device, dtype=v_model.dtype)
            continue

        if k_ck.endswith("embedding.weight") or k_ck.endswith("embed.weight") or k_ck == "embedding.weight":
            new_state[k_ck] = _make_tensor_from(ck_tensor, tuple(v_model.shape), device, k_ck)
            continue

        if k_ck.startswith("fc_dec") and len(v_model.shape) == 2 and len(ck_tensor.shape) == 2:
            new_state[k_ck] = _make_tensor_from(ck_tensor, tuple(v_model.shape), device, k_ck)
            continue

        if k_ck.startswith("fc_dec") and len(v_model.shape) == 1 and len(ck_tensor.shape) == 1:
            new_state[k_ck] = _make_tensor_from(ck_tensor, tuple(v_model.shape), device, k_ck)
            continue

        try:
            new_state[k_ck] = _make_tensor_from(ck_tensor, tuple(v_model.shape), device, k_ck)
            LOG.warning(f"[adapt] Generic fallback adaptation used for {k_ck}")
        except Exception as e:
            LOG.error(f"[adapt] Could not adapt {k_ck}: {e}")
            missing.append(k_ck)

    for k_model in model_state:
        if k_model in new_state:
            continue
        if k_model not in ckpt_state_dict:
            missing.append(k_model)

    model_state.update(new_state)
    model.load_state_dict(model_state)
    return missing, unexpected


def load_model(device="cpu", ckpt_path=MODEL_PATH):
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    LOG.info(f"[load_model] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    ckpt_meta = {}
    if isinstance(ckpt, dict):
        for k in ("vocab", "vocab_size", "embed_dim", "seq_len", "latent_dim"):
            if k in ckpt:
                ckpt_meta[k] = ckpt[k]

    kwargs = {}
    try:
        if "vocab" in ckpt_meta:
            kwargs["vocab_size"] = int(ckpt_meta["vocab"])
        if "vocab_size" in ckpt_meta:
            kwargs["vocab_size"] = int(ckpt_meta["vocab_size"])
        if "embed_dim" in ckpt_meta:
            kwargs["embed_dim"] = int(ckpt_meta["embed_dim"])
        if "seq_len" in ckpt_meta:
            kwargs["seq_len"] = int(ckpt_meta["seq_len"])
        if "latent_dim" in ckpt_meta:
            kwargs["latent_dim"] = int(ckpt_meta["latent_dim"])
    except Exception:
        kwargs = {}

    if kwargs:
        LOG.info(f"[load_model] Instantiating ProteinVAE with args: {kwargs}")
        try:
            model = ProteinVAE(**kwargs)
        except Exception:
            LOG.warning("[load_model] Failed to instantiate with inferred kwargs; using defaults.")
            model = ProteinVAE()
    else:
        model = ProteinVAE(vocab_size=21, seq_len=300, embed_dim=64, latent_dim=64)
        LOG.info(f"[load_model] Instantiated ProteinVAE with defaults.")

    try:
        model.load_state_dict(state_dict, strict=True)
        LOG.info("[load_model] loaded checkpoint with strict=True (exact match).")
    except Exception as e_strict:
        LOG.warning(f"[load_model] strict load failed: {e_strict}")
        LOG.info("[load_model] attempting shape-adaptation fallback.")
        try:
            missing, unexpected = adapt_and_load_state_dict(model, state_dict, device=device)
            LOG.info("[load_model] Loaded checkpoint with shape-adaptation fallback.")
            if missing:
                LOG.warning(f"[load_model] Missing model keys after adaptation: {len(missing)}")
            if unexpected:
                LOG.info(f"[load_model] Unexpected checkpoint keys skipped: {len(unexpected)}")
        except Exception as e2:
            LOG.error(f"[load_model] failed to load checkpoint with shape-adaptation: {e2}")
            raise e2

    model.to(device)
    model.eval()
    return model


# -------------------------
# Scoring wrapper
# -------------------------
def make_scoring_fn_wrapper(scoring_fn: Callable[[str], float]) -> Callable:
    """
    Returns a scoring function that accepts either:
      - a FASTA-like string (e.g. 'ACDE...'), or
      - an array/list of indices (numpy/tensor) which we will decode to a string first.
    The returned function always returns a float score.
    """
    def scoring_wrapper(seq_or_indices):
        # indices (torch.Tensor or np.ndarray or list)
        if isinstance(seq_or_indices, torch.Tensor):
            arr = seq_or_indices.detach().cpu().numpy()
        elif isinstance(seq_or_indices, np.ndarray):
            arr = seq_or_indices
        elif isinstance(seq_or_indices, (list, tuple)):
            arr = np.asarray(seq_or_indices)
        else:
            # assume it's a string already
            try:
                return float(scoring_fn(seq_or_indices))
            except Exception:
                raise

        # decode indices -> sequence (AA letters)
        try:
            seq_str = decode_indices_to_seq(arr.tolist())
        except Exception:
            # fallback: attempt to join integer tokens (not ideal)
            seq_str = "".join([str(int(x)) for x in arr.flatten().tolist()[:300]])
        return float(scoring_fn(seq_str))
    return scoring_wrapper


# -------------------------
# Safe optimizer-call wrapper (explicit, no blind positional attempts)
# -------------------------
def call_optimizer_explicit(optimizer_fn, *,
                            model,
                            seq_t: torch.Tensor,
                            phys_t: Optional[torch.Tensor],
                            scoring_fn: Callable,
                            device: str,
                            n_trials: int,
                            q_kwargs: dict):
    """
    Call optimizer_fn with the known signature:
      (model, seq_t, phys_t, scoring_fn, n_qubits=..., n_layers=..., ...)
    We pass explicit keywords only to avoid positional confusion.
    """
    # Inspect signature for debugging
    try:
        sig = inspect.signature(optimizer_fn)
        LOG.info(f"[optimizer_sig] {optimizer_fn.__name__} signature: {sig}")
    except Exception:
        LOG.info(f"[optimizer_sig] could not inspect signature.")

    # Build explicit kwargs
    call_kwargs = dict(
        model=model,
        seq_t=seq_t,
        phys_t=phys_t,
        scoring_fn=scoring_fn,
        device=device,
    )
    # Provider n_trials via n_restarts / n_steps mapping? Keep n_trials out -- pass q_kwargs as-is.
    # merge q_kwargs last so it can override defaults
    call_kwargs.update(q_kwargs)

    LOG.info(f"[optimizer_call] Calling optimizer with keys: {list(call_kwargs.keys())}")
    return optimizer_fn(**call_kwargs)


# -------------------------
# Data / run helpers
# -------------------------
def load_sample_file(file_path, max_len=300):
    d = np.load(file_path, allow_pickle=True)
    seq = d.get("seq_enc")
    if seq is None:
        raise KeyError(f"seq_enc missing in {file_path}")
    seq = np.asarray(seq)[:max_len]
    phys = d.get("phys_props")
    if phys is None:
        phys = np.zeros((max_len, 5), dtype=np.float32)
    else:
        phys = np.asarray(phys)[:max_len]
    return seq, phys


def run_one(file_path, model, device="cpu", n_trials=100, q_args=None):
    LOG.info(f"[run_one] Processing: {file_path}")
    seq, phys = load_sample_file(file_path, max_len=300)

    seq_t = torch.tensor(seq[None, :], dtype=torch.long, device=device)
    phys_t = torch.tensor(phys[None, :, :], dtype=torch.float32, device=device)

    # classical decoding (compute baseline) and get mu if available
    with torch.no_grad():
        logits, mu, logvar = model(seq_t, phys_t)
        classical_logits = logits
        classical_seq_idx = classical_logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        mu_vec = mu.squeeze(0) if mu is not None else None

    classical_score = np.nan
    try:
        # score_sequence_combined expects a sequence string usually; wrap below too
        classical_score = float(score_sequence_combined(decode_indices_to_seq(classical_seq_idx.tolist())))
    except Exception:
        try:
            classical_score = float(score_sequence_combined(classical_seq_idx))
        except Exception:
            classical_score = float(np.nan)

    q_args = q_args or {}

    # create scoring wrapper for quantum optimizer
    scoring_wrapper = make_scoring_fn_wrapper(score_sequence_combined)

    # Try calling the optimizer explicitly with the required scoring_fn param
    start = time.time()
    try:
        res = call_optimizer_explicit(
            optimize_latent_with_bioscore,
            model=model,
            seq_t=seq_t,
            phys_t=phys_t,
            scoring_fn=scoring_wrapper,
            device=device,
            n_trials=n_trials,
            q_kwargs={"n_qubits": q_args.get("n_qubits", 4),
                      "n_layers": q_args.get("n_layers", 2),
                      "n_steps": q_args.get("n_steps", 200),
                      "n_restarts": q_args.get("n_restarts", 8),
                      "init_scale": q_args.get("init_scale", 1.0),
                      "seed": q_args.get("seed", None),
                      }
        )
    except Exception as e:
        LOG.error(f"[run_one] Optimizer call failed for {file_path}: {e}")
        elapsed = time.time() - start
        return {
            "file": Path(file_path).name,
            "classical_seq_idx": np.asarray(classical_seq_idx),
            "classical_score": classical_score,
            "best_seq_idx": np.array([], dtype=object),
            "best_score": None,
            "best_params": None,
            "best_zq": None,
            "trials": [],
            "elapsed_s": elapsed,
        }

    elapsed = time.time() - start
    LOG.info(f"[run_one] Quantum optimization elapsed s: {elapsed:.3f}")

    out = {
        "file": Path(file_path).name,
        "classical_seq_idx": np.asarray(classical_seq_idx),
        "classical_score": classical_score,
        "best_seq_idx": np.asarray(res.get("best_quantum_seq", [])) if res.get("best_quantum_seq", None) is not None else np.array([], dtype=object),
        "best_score": res.get("best_quantum_score", None) or res.get("best_score", None),
        "best_params": res.get("best_params", None),
        "best_zq": np.asarray(res.get("best_zq")) if res.get("best_zq", None) is not None else None,
        "trials": res.get("trials", []),
        "elapsed_s": elapsed,
    }
    return out


def safe_save_result(result_dict, out_path=OUT_DIR / "quantum_opt_result.npz"):
    savable = {}
    for k, v in result_dict.items():
        if isinstance(v, np.ndarray):
            savable[k] = v
        elif torch.is_tensor(v):
            savable[k] = v.cpu().numpy()
        elif isinstance(v, (list, tuple)):
            savable[k] = np.array(v, dtype=object)
        else:
            try:
                savable[k] = np.asarray(v)
            except Exception:
                savable[k] = np.array(v, dtype=object)
    np.savez(out_path, **savable)
    LOG.info(f"[safe_save_result] Saved: {out_path}")


# -------------------------
# Main
# -------------------------
def main(device=None, n_trials=200, limit_files=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info(f"[main] Using device: {device}")

    model = load_model(device=device)

    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError("No processed files found in data/processed/")

    if limit_files is not None:
        files = files[:limit_files]

    results = []
    for i, f in enumerate(files):
        LOG.info(f"\n[main] ({i+1}/{len(files)}) Running quantum opt for: {f}")
        try:
            r = run_one(f, model, device=device, n_trials=n_trials, q_args={})
            results.append(r)
        except Exception as e:
            LOG.error(f"[main] Error processing {f}: {e}", exc_info=False)
            continue

    # Build a compact summary for saving and CSV
    summary_rows = []
    for r in results:
        classical_score = r.get("classical_score", np.nan)
        best_score = r.get("best_score", np.nan)
        try:
            delta = float(best_score) - float(classical_score)
        except Exception:
            delta = np.nan

        summary_rows.append({
            "file": r["file"],
            "classical_score": float(classical_score) if (classical_score is not None and not np.isnan(classical_score)) else np.nan,
            "best_score": float(best_score) if (best_score is not None and not np.isnan(best_score)) else np.nan,
            "delta": delta
        })

    safe_save_result({"results": results, "summary": summary_rows}, OUT_DIR / "quantum_opt_batch_raw.npz")

    csv_path = OUT_DIR / "quantum_opt_batch_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "classical_score", "best_score", "delta"])
        writer.writeheader()
        for row in summary_rows:
            try:
                d = row["best_score"] - row["classical_score"]
            except Exception:
                d = ""
            writer.writerow({"file": row["file"], "classical_score": row["classical_score"], "best_score": row["best_score"], "delta": d})
    LOG.info(f"[main] Saved CSV summary: {csv_path}")
    LOG.info("[main] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_quantum_opt", description="Run quantum optimisation across processed files.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu). If None, auto-select.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for the quantum optimizer per file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files processed (for quick tests).")
    args = parser.parse_args()

    main(device=args.device, n_trials=args.n_trials, limit_files=args.limit)
