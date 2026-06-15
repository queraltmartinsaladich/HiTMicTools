#!/usr/bin/env python3
"""
Train the division classifier MLP (Model B).

Reads triplet_features / triplet_labels from the HDF5 files produced by
build_dataset.py and trains a small MLP to score (parent @ t, d1 @ t+1,
d2 @ t+1) division candidates. Output replaces reconcile_lineage().

Usage:
    python train_division.py --dataset_dir /path/to/dataset \
                             --output_dir  /path/to/weights \
                             --epochs 50 --batch_size 512
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
from HiTMicTools.tracking.feature_extraction import TRIPLET_FEATURE_NAMES


# ── model ─────────────────────────────────────────────────────────────────────

class DivisionMLP(nn.Module):
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── focal loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        # alpha=0.75 upweights the rare positive (division) class
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha),
                              torch.tensor(1.0 - self.alpha))
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


# ── data loading ──────────────────────────────────────────────────────────────

def load_split(path: str):
    with h5py.File(path, "r") as f:
        feats = torch.tensor(f["triplet_features"][:], dtype=torch.float32)
        labels = torch.tensor(f["triplet_labels"][:], dtype=torch.float32)
    return feats, labels


# ── threshold tuning ──────────────────────────────────────────────────────────

def best_threshold_recall_priority(model: nn.Module, feats: torch.Tensor,
                                   labels: torch.Tensor,
                                   device: torch.device,
                                   min_precision: float = 0.5) -> float:
    """
    Find the threshold that maximises recall while keeping precision >= min_precision.
    For division detection, missed divisions (broken lineages) are worse than
    false positives (extra reconcile checks), so recall is prioritised.
    """
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(feats.to(device))).cpu().numpy()
    y = labels.numpy().astype(int)
    best_t, best_recall = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= t).astype(int)
        p = precision_score(y, preds, zero_division=0)
        r = recall_score(y, preds, zero_division=0)
        if p >= min_precision and r > best_recall:
            best_recall, best_t = r, float(t)
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

    print(f"Train: {len(train_labels)} triplets  "
          f"({int(train_labels.sum())} pos / {int((train_labels==0).sum())} neg)")
    print(f"Val:   {len(val_labels)} triplets  "
          f"({int(val_labels.sum())} pos / {int((val_labels==0).sum())} neg)")

    if len(train_labels) == 0 or int(train_labels.sum()) == 0:
        print("ERROR: no positive division examples — regenerate with denser movies.")
        sys.exit(1)

    # Weighted sampler — oversample the rare positive (division) class
    n_pos = max(1, int(train_labels.sum()))
    n_neg = max(1, int((train_labels == 0).sum()))
    weights = torch.where(train_labels == 1,
                          torch.tensor(1.0 / n_pos),
                          torch.tensor(1.0 / n_neg))
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = TensorDataset(train_feats, train_labels)
    val_ds = TensorDataset(val_feats, val_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    model = DivisionMLP(input_dim=train_feats.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    best_val_recall = 0.0
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

        # Track best recall (division recall matters more than precision)
        if val_r > best_val_recall:
            best_val_recall = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    model.load_state_dict(best_state)
    threshold = best_threshold_recall_priority(
        model, val_feats, val_labels, device, min_precision=args.min_precision)
    metrics = evaluate(model, val_feats, val_labels, threshold, device)

    print(f"\nBest val (threshold={threshold:.2f}, min_precision={args.min_precision}): "
          f"F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "division_classifier.pt")
    torch.save({
        "state_dict":    best_state,
        "input_dim":     train_feats.shape[1],
        "feature_names": TRIPLET_FEATURE_NAMES,
        "threshold":     threshold,
        "val_metrics":   metrics,
        "architecture":  "DivisionMLP[8→32→16→1]",
    }, out_path)
    print(f"Saved → {out_path}")

    with open(os.path.join(args.output_dir, "division_classifier_metrics.json"), "w") as f:
        json.dump({"threshold": threshold, **metrics}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_precision", type=float, default=0.5,
                        help="Minimum precision when tuning threshold (recall-priority)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
