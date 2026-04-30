"""
training/train.py — Boucle d'entraînement CRONOS JEPA v2

Usage :
    python -m training.train --data data/processed --epochs 300
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
    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        X_ctx = np.load(data_dir / "X_ctx.npy")
        X_tgt = np.load(data_dir / "X_tgt.npy")
        self.X_ctx = torch.tensor(X_ctx, dtype=torch.float32)
        self.X_tgt = torch.tensor(X_tgt, dtype=torch.float32)
        print(f"Dataset : {len(self)} paires | X_ctx {self.X_ctx.shape} | X_tgt {self.X_tgt.shape}")

    def __len__(self):
        return len(self.X_ctx)

    def __getitem__(self, idx):
        return self.X_ctx[idx], self.X_tgt[idx]


# ─────────────────────────────────────────────────────────────
# Scheduler : warmup + cosine decay
# ─────────────────────────────────────────────────────────────

def get_lr(step, total_steps, lr_max, lr_min, warmup_steps):
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))


def get_ema_tau(step, total_steps, tau_base=0.996, tau_final=0.9999):
    """
    EMA tau augmente progressivement pendant l'entraînement.
    Commence avec tau faible (mise à jour rapide) puis ralentit (cible stable).
    """
    progress = step / max(total_steps, 1)
    return tau_final - (tau_final - tau_base) * (math.cos(math.pi * progress) + 1) / 2


# ─────────────────────────────────────────────────────────────
# Boucle d'entraînement
# ─────────────────────────────────────────────────────────────

def train(
    data_dir: str = "data/processed",
    epochs: int = 300,
    batch_size: int = 32,
    lr_max: float = 3e-4,
    lr_min: float = 1e-5,
    warmup_epochs: int = 20,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    d_model: int = 64,
    n_layers: int = 3,
    drop_path_rate: float = 0.1,
    ema_tau: float = 0.996,
    mask_ratio: float = 0.25,
    save_dir: str = "checkpoints",
    device: str = "auto",
    log_every: int = 10,
):
    # ── Device ──
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # ── Dataset ──
    dataset = JEPADataset(data_dir)
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    print(f"Train : {n_train} | Val : {n_val}")

    # ── Modèle ──
    model = JEPA(
        d_model=d_model,
        n_layers=n_layers,
        drop_path_rate=drop_path_rate,
        ema_tau=ema_tau,
        mask_ratio=mask_ratio,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Params : {trainable:,} entraînables / {total:,} total")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_max,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # ── Steps ──
    total_steps   = epochs * len(train_loader)
    warmup_steps  = warmup_epochs * len(train_loader)

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    step = 0

    # ─────────────────────────────────────────────────────────
    print(f"\nDébut entraînement — {epochs} epochs\n")

    for epoch in range(1, epochs + 1):

        # ── Train ──
        model.train()
        train_losses = []
        train_inv, train_var, train_cov = [], [], []

        for x_ctx, x_tgt in train_loader:
            x_ctx = x_ctx.to(device)
            x_tgt = x_tgt.to(device)

            # LR dynamique
            lr = get_lr(step, total_steps, lr_max, lr_min, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward
            _, _, loss, losses = model(x_ctx, x_tgt, use_masking=True)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # EMA update avec tau progressif
            tau = get_ema_tau(step, total_steps, ema_tau, 0.9999)
            model.update_target_encoder(tau=tau)

            train_losses.append(losses["total"])
            train_inv.append(losses["inv_loss"])
            train_var.append(losses["var_loss"])
            train_cov.append(losses["cov_loss"])
            step += 1

        # ── Validation ──
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x_ctx, x_tgt in val_loader:
                x_ctx = x_ctx.to(device)
                x_tgt = x_tgt.to(device)
                _, _, loss, losses = model(x_ctx, x_tgt, use_masking=False)
                val_losses.append(losses["total"])

        t_loss = sum(train_losses) / len(train_losses)
        v_loss = sum(val_losses)   / len(val_losses)

        # ── Log ──
        if epoch % log_every == 0 or epoch == 1:
            inv = sum(train_inv) / len(train_inv)
            var = sum(train_var) / len(train_var)
            cov = sum(train_cov) / len(train_cov)
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train {t_loss:.4f} (inv={inv:.3f} var={var:.3f} cov={cov:.3f}) | "
                f"Val {v_loss:.4f} | "
                f"LR {lr:.2e} | τ {tau:.4f}"
            )

        # ── Sauvegarde ──
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": v_loss,
                "config": {
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "ema_tau": ema_tau,
                    "mask_ratio": mask_ratio,
                },
            }, f"{save_dir}/best_model.pt")

    print(f"\n✓ Entraînement terminé")
    print(f"  Meilleure val loss : {best_val_loss:.4f}")
    print(f"  Modèle sauvegardé  : {save_dir}/best_model.pt")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",           type=str,   default="data/processed")
    parser.add_argument("--epochs",         type=int,   default=300)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr_max",         type=float, default=3e-4)
    parser.add_argument("--lr_min",         type=float, default=1e-5)
    parser.add_argument("--warmup",         type=int,   default=20)
    parser.add_argument("--d_model",        type=int,   default=64)
    parser.add_argument("--n_layers",       type=int,   default=3)
    parser.add_argument("--drop_path",      type=float, default=0.1)
    parser.add_argument("--ema_tau",        type=float, default=0.996)
    parser.add_argument("--mask_ratio",     type=float, default=0.25)
    parser.add_argument("--save_dir",       type=str,   default="checkpoints")
    parser.add_argument("--device",         type=str,   default="auto")
    parser.add_argument("--log_every",      type=int,   default=10)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        warmup_epochs=args.warmup,
        d_model=args.d_model,
        n_layers=args.n_layers,
        drop_path_rate=args.drop_path,
        ema_tau=args.ema_tau,
        mask_ratio=args.mask_ratio,
        save_dir=args.save_dir,
        device=args.device,
        log_every=args.log_every,
    )