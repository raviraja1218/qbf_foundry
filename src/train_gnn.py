# src/train_gnn.py
"""
Train a small GNN baseline using processed data in data/processed/*.npz.

Usage:
    export PYTHONPATH="$PWD"
    conda activate qbf_clean
    python -m src.train_gnn  # or `python src/train_gnn.py`
"""
import glob
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import random_split, Dataset as TorchDataset

# torch_geometric imports
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.gnn import SimpleGNN

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)
Path("models/interim_models").mkdir(parents=True, exist_ok=True)


def safe_npz_to_pygdata(path, verbose=True):
    """
    Robust loader for .npz -> torch_geometric.data.Data.

    - Trims arrays so that seq_enc, coords, adj agree on node count (uses min).
    - Builds edge_index from adj (or chain if missing).
    - Drops edges that reference out-of-range nodes.
    - Attaches basic metadata (raw_seq, path).
    """
    d = np.load(path, allow_pickle=True)
    seq_enc = d.get("seq_enc", None)
    coords = d.get("coords", None)
    adj = d.get("adj", None)
    raw_seq = d.get("raw_seq", "")
    phys_props = d.get("phys_props", None)

    if seq_enc is None:
        raise ValueError(f"{path} missing required 'seq_enc'")

    seq_enc = np.asarray(seq_enc).reshape(-1).astype(int)
    n_seq = seq_enc.shape[0]

    n_coords = None
    if coords is not None:
        coords = np.asarray(coords)
        if coords.ndim >= 1:
            n_coords = coords.shape[0]

    n_adj = None
    if adj is not None:
        adj = np.asarray(adj)
        if adj.ndim == 2:
            n_adj = adj.shape[0]

    # choose consistent node count (use smallest to avoid OOB indices)
    candidates = [x for x in (n_seq, n_coords, n_adj) if x is not None]
    if not candidates:
        raise ValueError(f"{path} contains no usable arrays (seq/coords/adj)")

    num_nodes = int(min(candidates))

    if verbose and (n_seq != num_nodes or n_coords != num_nodes or n_adj != num_nodes):
        print(
            f"[safe_loader] {os.path.basename(path)} shapes -> seq:{n_seq} coords:{n_coords} adj:{n_adj} -> using {num_nodes} nodes"
        )

    seq_enc = seq_enc[:num_nodes]
    if phys_props is not None:
        phys_props = np.asarray(phys_props)[:num_nodes]
    if coords is not None:
        coords = coords[:num_nodes]
    if adj is not None:
        adj = adj[:num_nodes, :num_nodes]

    # Build node features: prefer phys_props, else fallback to token indices as single feature
    if phys_props is not None:
        x = torch.tensor(np.asarray(phys_props, dtype=np.float32))
        # also append token id as numeric column (optional)
        seq_col = torch.tensor(seq_enc, dtype=torch.float32).unsqueeze(-1)
        x = torch.cat([seq_col, x], dim=-1)
    else:
        x = torch.tensor(seq_enc, dtype=torch.float32).unsqueeze(-1)

    # Build edges
    if adj is None or adj.size == 0:
        # fallback: chain edges between consecutive residues
        if num_nodes <= 1:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            src = np.arange(num_nodes - 1, dtype=np.int64)
            dst = src + 1
            ei = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
            edge_index = torch.tensor(ei, dtype=torch.long)
        edge_attr = None
    else:
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"{path} adj must be square, got {adj.shape}")
        rows, cols = np.nonzero(adj)
        # remove self-loops
        mask = rows != cols
        rows = rows[mask].astype(np.int64)
        cols = cols[mask].astype(np.int64)
        ei = np.vstack([rows, cols])
        # drop edges referencing out-of-range indices (defensive)
        valid_mask = (ei[0, :] < num_nodes) & (ei[1, :] < num_nodes)
        dropped = (~valid_mask).sum()
        if verbose and dropped:
            print(f"[safe_loader] Dropping {int(dropped)} out-of-range edges in {os.path.basename(path)}")
        ei = ei[:, valid_mask]
        edge_index = torch.tensor(ei, dtype=torch.long)
        # capture weights if adj contains non-binary weights
        if not np.all((adj == 0) | (adj == 1)):
            edge_attr = torch.tensor(adj[rows, cols][valid_mask].astype(np.float32)).unsqueeze(-1)
        else:
            edge_attr = None

    data = Data(x=x, edge_index=edge_index)
    if edge_attr is not None:
        data.edge_attr = edge_attr
    if coords is not None:
        data.pos = torch.tensor(coords, dtype=torch.float32)
    data.y = torch.tensor([float(np.nanmean(phys_props[:, 0]))], dtype=torch.float32) if phys_props is not None else torch.tensor([0.0], dtype=torch.float32)
    data.raw_seq = raw_seq if raw_seq is not None else ""
    data.path = path
    data.num_nodes = num_nodes
    return data


class ProcessedDataset(TorchDataset):
    """
    Simple dataset wrapper that loads .npz files on demand using safe_npz_to_pygdata.
    """

    def __init__(self, files, transform=None):
        if isinstance(files, str):
            files = sorted(glob.glob(files))
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = safe_npz_to_pygdata(path, verbose=True)
        if self.transform is not None:
            data = self.transform(data)
        return data


def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = torch.nn.functional.mse_loss(pred, batch.y.squeeze(-1))
        loss.backward()
        opt.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    import numpy as _np

    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds.append(out.cpu().numpy())
            labels.append(batch.y.squeeze(-1).cpu().numpy())
    if len(preds) == 0:
        return {"mse": float("nan"), "mae": float("nan"), "corr": 0.0, "n": 0}
    preds = _np.concatenate(preds)
    labels = _np.concatenate(labels)
    mse = float(((preds - labels) ** 2).mean())
    mae = float(_np.abs(preds - labels).mean())
    # pearson correlation
    if preds.std() == 0 or labels.std() == 0:
        corr = 0.0
    else:
        corr = float(_np.corrcoef(preds, labels)[0, 1])
    return {"mse": mse, "mae": mae, "corr": corr, "n": len(preds)}


def main(
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=40,
    batch_size=8,
    lr=1e-3,
):
    print("Device:", device)
    files = sorted(glob.glob("data/processed/*.npz"))
    if len(files) == 0:
        raise FileNotFoundError("No processed files found in data/processed. Run data preprocessing first.")
    dataset = ProcessedDataset(files=files)
    print(f"Loaded dataset: {len(dataset)} graphs")

    # small train/test split
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = n - n_train
    # ensure deterministic split
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # infer input channels from first example
    example = dataset[0]
    in_channels = example.x.shape[1]

    model = SimpleGNN(in_channels=in_channels, hidden_channels=64, num_layers=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = {"mse": 1e9}
    history = []
    for ep in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": ep, "train_loss": train_loss, **val_metrics})
        print(
            f"Epoch {ep}/{epochs} — train_loss: {train_loss:.4f} — val_mse: {val_metrics['mse']:.6f} — corr: {val_metrics['corr']:.3f}"
        )
        if val_metrics["mse"] < best["mse"]:
            best = val_metrics
            torch.save(model.state_dict(), "models/interim_models/gnn_baseline.pt")

    # Save a simple CSV of metrics
    import csv

    out_csv = REPORT_DIR / "gnn_metrics.csv"
    keys = list(history[0].keys()) if history else ["epoch", "train_loss", "mse", "mae", "corr", "n"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    print("Saved metrics to", out_csv)
    print("Best val:", best)


if __name__ == "__main__":
    main()
