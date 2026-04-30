"""
training/train.py — Boucle d'entraînement CRONOS JEPA

Usage :
    python -m training.train --data data/processed --epochs 200 --batch_size 32
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from models.jepa import JEPA


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class JEPADataset(Dataset):
    """
    Charge les paires (X_ctx, X_tgt) depuis les fichiers .npy.
    """

    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)

        # Charge les tenseurs
        X_ctx = np.load(data_dir / "X_ctx.npy")
        X_tgt = np.load(data_dir / "X_tgt.npy")

        self.X_ctx = torch.tensor(X_ctx, dtype=torch.float32)
        self.X_tgt = torch.tensor(X_tgt, dtype=torch.float32)

        print(f"Dataset chargé : {len(self)} paires")
        print(f"  X_ctx : {self.X_ctx.shape}")
        print(f"  X_tgt : {self.X_tgt.shape}")

    def __len__(self):
        return len(self.X_ctx)

    def __getitem__(self, idx):
        return self.X_ctx[idx], self.X_tgt[idx]


# ─────────────────────────────────────────────────────────────
# Learning rate scheduler avec warmup
# ─────────────────────────────────────────────────────────────

def get_lr(step: int, total_steps: int, lr_max: float, lr_min: float, warmup_steps: int) -> float:
    """
    Warmup linéaire puis cosine decay.

    Args:
        step         : step courant
        total_steps  : nombre total de steps
        lr_max       : learning rate maximum
        lr_min       : learning rate minimum
        warmup_steps : nombre de steps de warmup
    """
    if step < warmup_steps:
        # Warmup linéaire : 0 → lr_max
        return lr_max * step / warmup_steps
    else:
        # Cosine decay : lr_max → lr_min
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return lr_min + (lr_max - lr_min) * cosine


# ─────────────────────────────────────────────────────────────
# Boucle d'entraînement
# ─────────────────────────────────────────────────────────────

def train(
    data_dir: str = "data/processed",
    epochs: int = 200,
    batch_size: int = 32,
    lr_max: float = 3e-4,
    lr_min: float = 1e-5,
    warmup_epochs: int = 10,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    d_model: int = 64,
    ema_tau: float = 0.996,
    save_dir: str = "checkpoints",
    device: str = "auto",
):
    # ── Device ──
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # ── Dataset ──
    dataset = JEPADataset(data_dir)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    print(f"Train : {n_train} | Val : {n_val}")

    # ── Modèle ──
    model = JEPA(d_model=d_model, ema_tau=ema_tau).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params entraînables : {trainable:,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_max,
        weight_decay=weight_decay,
    )

    # ── Scheduler ──
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    # ── Save dir ──
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    step = 0

    # ─────────────────────────────────────────────────────────
    # Boucle principale
    # ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):

        # ── Train ──
        model.train()
        train_losses = []

        for x_ctx, x_tgt in train_loader:
            x_ctx = x_ctx.to(device)
            x_tgt = x_tgt.to(device)

            # Learning rate dynamique
            lr = get_lr(step, total_steps, lr_max, lr_min, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward
            _, _, loss = model(x_ctx, x_tgt)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — évite les explosions de gradient
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # EMA update de l'encodeur target
            model.update_target_encoder()

            train_losses.append(loss.item())
            step += 1

        # ── Validation ──
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x_ctx, x_tgt in val_loader:
                x_ctx = x_ctx.to(device)
                x_tgt = x_tgt.to(device)
                _, _, loss = model(x_ctx, x_tgt)
                val_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss   = sum(val_losses)   / len(val_losses)

        # ── Log ──
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train {train_loss:.4f} | "
                f"Val {val_loss:.4f} | "
                f"LR {lr:.2e}"
            )

        # ── Sauvegarde le meilleur modèle ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": {
                        "d_model": d_model,
                        "ema_tau": ema_tau,
                    },
                },
                f"{save_dir}/best_model.pt",
            )

    print(f"\nEntraînement terminé — meilleure val loss : {best_val_loss:.4f}")
    print(f"Modèle sauvegardé dans {save_dir}/best_model.pt")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        type=str,   default="data/processed")
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr_max",      type=float, default=3e-4)
    parser.add_argument("--lr_min",      type=float, default=1e-5)
    parser.add_argument("--warmup",      type=int,   default=10)
    parser.add_argument("--d_model",     type=int,   default=64)
    parser.add_argument("--ema_tau",     type=float, default=0.996)
    parser.add_argument("--save_dir",    type=str,   default="checkpoints")
    parser.add_argument("--device",      type=str,   default="auto")
    args = parser.parse_args()

    train(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        warmup_epochs=args.warmup,
        d_model=args.d_model,
        ema_tau=args.ema_tau,
        save_dir=args.save_dir,
        device=args.device,
    )