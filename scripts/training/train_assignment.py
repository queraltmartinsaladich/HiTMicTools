#!/usr/bin/env python3
"""
Train the assignment scorer MLP (Model A).

Reads pair_features / pair_labels from the HDF5 files produced by
build_dataset.py and trains a small MLP to score (cell_i @ t, cell_j @ t+1)
assignment candidates. Output replaces the Euclidean centroid distance in
the Hungarian tracker.

Usage:
    python train_assignment.py --dataset_dir /path/to/dataset \
                               --output_dir  /path/to/weights \
                               --epochs 50 --batch_size 2048
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from HiTMicTools.tracking.feature_extraction import PAIR_FEATURE_NAMES


# ── model ─────────────────────────────────────────────────────────────────────

class AssignmentMLP(nn.Module):
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── focal loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ── data loading ──────────────────────────────────────────────────────────────

def load_split(path: str):
    with h5py.File(path, "r") as f:
        feats = torch.tensor(f["pair_features"][:], dtype=torch.float32)
        labels = torch.tensor(f["pair_labels"][:], dtype=torch.float32)
    return feats, labels


# ── threshold tuning ──────────────────────────────────────────────────────────

def best_threshold(model: nn.Module, feats: torch.Tensor,
                   labels: torch.Tensor, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(feats.to(device))).cpu().numpy()
    y = labels.numpy().astype(int)
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, feats: torch.Tensor, labels: torch.Tensor,
             threshold: float, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(feats.to(device))).cpu().numpy()
    y = labels.numpy().astype(int)
    preds = (probs >= threshold).astype(int)
    return {
        "f1":        f1_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "recall":    recall_score(y, preds, zero_division=0),
        "n_pos":     int(y.sum()),
        "n_neg":     int((y == 0).sum()),
    }


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_feats, train_labels = load_split(
        os.path.join(args.dataset_dir, "train_assignments.h5"))
    val_feats, val_labels = load_split(
        os.path.join(args.dataset_dir, "val_assignments.h5"))

    print(f"Train: {len(train_labels)} pairs  "
          f"({int(train_labels.sum())} pos / {int((train_labels==0).sum())} neg)")
    print(f"Val:   {len(val_labels)} pairs  "
          f"({int(val_labels.sum())} pos / {int((val_labels==0).sum())} neg)")

    if len(train_labels) == 0:
        print("ERROR: empty training set — regenerate dataset first.")
        sys.exit(1)

    # Weighted sampler to balance classes in each batch
    n_pos = int(train_labels.sum())
    n_neg = int((train_labels == 0).sum())
    weights = torch.where(train_labels == 1,
                          torch.tensor(1.0 / n_pos),
                          torch.tensor(1.0 / n_neg))
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = TensorDataset(train_feats, train_labels)
    val_ds = TensorDataset(val_feats, val_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    model = AssignmentMLP(input_dim=train_feats.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val F1':>8}  {'Val P':>8}  {'Val R':>8}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Quick val F1 at threshold=0.5 for monitoring
        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_probs.append(torch.sigmoid(model(xb.to(device))).cpu())
                val_true.append(yb)
        val_probs = torch.cat(val_probs).numpy()
        val_true = torch.cat(val_true).numpy().astype(int)
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        val_p = precision_score(val_true, val_preds, zero_division=0)
        val_r = recall_score(val_true, val_preds, zero_division=0)
        avg_loss = total_loss / len(train_loader)

        print(f"{epoch:5d}  {avg_loss:10.4f}  {val_f1:8.4f}  {val_p:8.4f}  {val_r:8.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Load best weights and tune threshold on val set
    model.load_state_dict(best_state)
    threshold = best_threshold(model, val_feats, val_labels, device)
    metrics = evaluate(model, val_feats, val_labels, threshold, device)

    print(f"\nBest val F1 (threshold={threshold:.2f}): "
          f"F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "assignment_scorer.pt")
    torch.save({
        "state_dict":    best_state,
        "input_dim":     train_feats.shape[1],
        "feature_names": PAIR_FEATURE_NAMES,
        "threshold":     threshold,
        "val_metrics":   metrics,
        "architecture":  "AssignmentMLP[10→64→32→1]",
    }, out_path)
    print(f"Saved → {out_path}")

    with open(os.path.join(args.output_dir, "assignment_scorer_metrics.json"), "w") as f:
        json.dump({"threshold": threshold, **metrics}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
