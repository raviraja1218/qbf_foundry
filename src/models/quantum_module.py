"""
Quantum module for latent optimization with biological scoring.

This is intentionally conservative: it uses PennyLane's default.qubit simulator
and a simple random-search / hill-climb so it works without gradient plumbing
or hardware access. It expects your project to provide:
 - ProteinVAE.decode(z: torch.Tensor) -> logits (B, seq_len, vocab)
 - scoring function: score_sequence_combined(sequence_str) -> float

Usage:
    from src.models.quantum_module import optimize_latent_with_bioscore
    optimize_latent_with_bioscore(model, seq_t, phys_t, scoring_fn, ...)
"""

from typing import Callable, Optional, Tuple
import os
import math
import time
import numpy as np
import torch

# PennyLane (make sure it's in your env)
import pennylane as qml

# for decoding index->AA mapping (optional)
try:
    from src.utils.aa_map import decode_indices_to_seq
except Exception:
    def decode_indices_to_seq(idx_list):
        # minimal fallback: convert ints to string of numbers separated by space
        return " ".join([str(int(x)) for x in idx_list])

# scoring function expected signature: func(sequence_str) -> float
# we'll import scoring utilities at runtime in runner to avoid circular imports


def _build_qnode(n_qubits: int, n_layers: int, out_dim: int):
    """
    Build a qnode that maps parameters -> out_dim expectation values in [-1,1].

    params shape: (n_layers, n_qubits, 2) or flattened
    We'll accept flattened params: length = n_layers * n_qubits * 2
    """

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(flat_params):
        # flat_params is a 1D numpy array
        params = flat_params.reshape(n_layers, n_qubits, 2)
        # simple hardware-efficient ansatz
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(params[layer, q, 0], wires=q)
                qml.RZ(params[layer, q, 1], wires=q)
            # entangle ring
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        # We'll return expectation values of PauliZ on first K wires (maybe repeat) to
        # produce as many outputs as needed (out_dim). We'll tile/pack them.
        exps = [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
        return exps

    return qnode


def _qnode_to_latent(qnode, params: np.ndarray, latent_dim: int) -> np.ndarray:
    """
    Run qnode (returns list of expvals length n_qubits) and map into latent_dim floats.
    Strategy: tile the expvals and apply simple linear transform (shift+scale) to match typical latent ranges.
    """
    exps = np.array(qnode(params)).astype(float)  # shape (n_qubits,)
    # tile / repeat to reach latent_dim
    repeats = math.ceil(latent_dim / exps.size)
    arr = np.tile(exps, repeats)[:latent_dim]
    # scale from [-1,1] to something like N(0,1) range: multiply by 1.5
    return arr * 1.5


def _decode_logits_from_model(model, z_tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a model and latent z (torch tensor [B, latent_dim]), return logits [B, seq_len, vocab]
    Tries model.decode(z) first; if missing, tries model.decode_latent or model.decoder
    As a final fallback, it uses a tiny linear layer simulated here (not ideal).
    """
    if hasattr(model, "decode"):
        return model.decode(z_tensor)
    # try common alternate
    if hasattr(model, "decode_latent"):
        return model.decode_latent(z_tensor)
    # fallback: if model has a small linear mapping to logits stored, try using it
    # last-resort: simple linear mapping implemented here
    B = z_tensor.shape[0]
    latent_dim = z_tensor.shape[-1]
    seq_len = getattr(model, "seq_len", 300)
    vocab = getattr(model, "vocab_size", getattr(model, "vocab", 21))
    # Very rough: map latent -> (B, seq_len, vocab) via a Linear -> repeat
    lin = torch.nn.Linear(latent_dim, seq_len * vocab)
    with torch.no_grad():
        out = lin(z_tensor).reshape(B, seq_len, vocab)
    return out


def _logits_to_pred_indices(logits: torch.Tensor) -> np.ndarray:
    """
    Convert logits [B, seq_len, vocab] -> predicted indices (B, seq_len) numpy int
    """
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    return preds.cpu().numpy()


def optimize_latent_with_bioscore(
    model: torch.nn.Module,
    seq_t: torch.Tensor,
    phys_t: Optional[torch.Tensor],
    scoring_fn: Callable[[str], float],
    n_qubits: int = 4,
    n_layers: int = 2,
    n_steps: int = 200,
    n_restarts: int = 8,
    init_scale: float = 1.0,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Optimize a single sample's latent vector z to maximize scoring_fn via a small quantum ansatz.

    Returns a dict with keys:
      - best_classical_seq (str), best_classical_score (float)
      - best_quantum_seq (str), best_quantum_score (float)
      - best_params (np.ndarray), best_zq (np.ndarray)
      - trials (list of tuples score, params)
    """

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ensure model on CPU for decode safety (we will call decode on CPU to avoid device mismatches)
    orig_device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    # 1) get classical latent by running model encode path
    with torch.no_grad():
        # most VAEs accept (seq, phys) and return (logits, mu, logvar)
        try:
            logits, mu, logvar = model(seq_t.to(device), phys_t.to(device) if phys_t is not None else None)
        except Exception:
            # try alternate API
            mu = getattr(model, "encode_latent", lambda x, y=None: torch.zeros((1, getattr(model, "latent_dim", 64))))(seq_t.to(device), phys_t.to(device) if phys_t is not None else None)
            logits = _decode_logits_from_model(model, mu)
            logvar = torch.zeros_like(mu)

        z = mu.detach().cpu().numpy()[0]  # (latent_dim,)
        latent_dim = z.shape[-1]

        # classical reconstructed sequence
        logits_cpu = logits.detach().cpu()
        classical_idx = logits_cpu.argmax(dim=-1).squeeze(0).cpu().numpy()
        classical_seq = decode_indices_to_seq(classical_idx)
        classical_score = scoring_fn(classical_seq)

    # Build qnode
    qnode = _build_qnode(n_qubits=n_qubits, n_layers=n_layers, out_dim=latent_dim)

    # random-search / hill-climb loop over restarts
    best_overall = {
        "score": classical_score,
        "params": None,
        "zq": z.copy(),
        "pred_idx": classical_idx,
        "seq": classical_seq,
        "why": "classical baseline"
    }
    trials = []

    param_len = n_layers * n_qubits * 2
    t0 = time.time()

    for restart in range(n_restarts):
        # start from small random init near zero
        params = np.random.normal(scale=init_scale, size=(param_len,)).astype(float)
        # evaluate initial
        zq = _qnode_to_latent(qnode, params, latent_dim)
        # convert zq to torch, decode
        with torch.no_grad():
            zq_t = torch.tensor(zq[None, :], dtype=torch.float32).to(device)
            logits_q = _decode_logits_from_model(model, zq_t)
            pred_idx = logits_q.argmax(dim=-1).squeeze(0).cpu().numpy()
            seq_q = decode_indices_to_seq(pred_idx)
            score_q = float(scoring_fn(seq_q))

        if score_q > best_overall["score"]:
            best_overall.update({"score": score_q, "params": params.copy(), "zq": zq.copy(), "pred_idx": pred_idx.copy(), "seq": seq_q, "why": f"init_restart_{restart}"})

        trials.append((score_q, params.copy()))

        # local hill-climb: gaussian proposals
        cur_params = params.copy()
        cur_score = score_q
        sigma = init_scale * 0.5
        for step in range(n_steps):
            cand = cur_params + np.random.normal(scale=sigma, size=cur_params.shape)
            zq_c = _qnode_to_latent(qnode, cand, latent_dim)
            with torch.no_grad():
                zq_t = torch.tensor(zq_c[None, :], dtype=torch.float32).to(device)
                logits_q = _decode_logits_from_model(model, zq_t)
                pred_idx_c = logits_q.argmax(dim=-1).squeeze(0).cpu().numpy()
                seq_c = decode_indices_to_seq(pred_idx_c)
                score_c = float(scoring_fn(seq_c))

            # accept if better (hill-climb)
            if score_c > cur_score:
                cur_score = score_c
                cur_params = cand
                # decrease sigma a bit for finer search
                sigma = max(1e-4, sigma * 0.98)

                # update global best
                if score_c > best_overall["score"]:
                    best_overall.update({"score": score_c, "params": cand.copy(), "zq": zq_c.copy(), "pred_idx": pred_idx_c.copy(), "seq": seq_c, "why": f"restart_{restart}_step_{step}"})

            trials.append((score_c, cand.copy()))

    elapsed = time.time() - t0

    # Package results
    result = {
        "classical_seq": classical_seq,
        "classical_score": classical_score,
        "best_seq": best_overall["seq"],
        "best_score": best_overall["score"],
        "best_params": best_overall["params"],
        "best_zq": best_overall["zq"],
        "trials": trials,
        "elapsed_s": elapsed,
        "latent_dim": latent_dim,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
    }

    # move model back to original device if necessary
    try:
        model.to(orig_device)
    except Exception:
        pass

    return result
