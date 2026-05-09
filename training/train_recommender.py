"""
training/train_recommender.py — Entraînement du modèle de recommandation CRONOS.

Labels automatiques depuis signaux de récupération objectifs :
- HRV J+1 vs HRV moyenne utilisateur
- Sleep score J+1
- Body battery chargé J+1

Usage :
    python -m training.train_recommender \
        --data data/processed \
        --jepa checkpoints/best_model.pt \
        --epochs 100
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from models.jepa import JEPA
from recommendation.recommender import CRONOSRecommender
from recommendation.encoders import encode_athlete_profile, encode_race_context
from recommendation.session_types import SESSION_CATALOGUE, N_SESSIONS


# ─────────────────────────────────────────────────────────────
# Labels automatiques depuis signaux de récupération
# ─────────────────────────────────────────────────────────────

def build_labels_from_recovery(
    data_dir: str,
    meta: pd.DataFrame,
) -> torch.Tensor:
    """
    Construit les labels depuis les signaux de récupération objectifs.

    Pour chaque fenêtre :
    1. Calcule un score de récupération composite depuis HRV J+1, sleep score J+1, body battery J+1
    2. Mappe ce score aux 20 types de séances selon leur intensité et coût de récupération
    """
    N = len(meta)
    labels = torch.full((N, N_SESSIONS), 0.5)

    daily_path = Path(data_dir).parent / "raw" / "daily_metrics.csv"
    if not daily_path.exists():
        print(f"  ⚠ {daily_path} introuvable — labels par défaut (0.5)")
        return labels

    daily = pd.read_csv(daily_path, parse_dates=["date"])
    user_col = "user_name" if "user_name" in daily.columns else "user"

    # Pré-calcule la HRV médiane par user
    hrv_medians = {}
    for u in daily[user_col].unique():
        hrv_medians[u] = daily[daily[user_col] == u]["hrv_last_night"].median()

    n_labeled = 0

    for idx, row in meta.iterrows():
        user = row.get("user")
        if not user:
            continue

        window_end = pd.Timestamp(row["window_end"])
        next_day   = window_end + pd.Timedelta(days=1)

        next_data = daily[
            (daily[user_col] == user) &
            (daily["date"] == next_day)
        ]

        if next_data.empty:
            continue

        nd = next_data.iloc[0]
        hrv_mean = hrv_medians.get(user, np.nan)

        # ── Score de récupération composite 0-1 ──
        recovery = 0.5
        weight_total = 0.0

        # HRV J+1 (poids 50%)
        hrv_next = nd.get("hrv_last_night")
        if pd.notna(hrv_next) and pd.notna(hrv_mean) and hrv_mean > 0:
            hrv_ratio = float(hrv_next) / float(hrv_mean)
            hrv_score = min(hrv_ratio, 1.5) / 1.5  # 0-1
            recovery += (hrv_score - 0.5) * 0.5
            weight_total += 0.5

        # Sleep score J+1 (poids 30%)
        sleep_next = nd.get("sleep_score")
        if pd.notna(sleep_next) and float(sleep_next) > 0:
            sleep_score = float(sleep_next) / 100
            recovery += (sleep_score - 0.5) * 0.3
            weight_total += 0.3

        # Body battery chargé J+1 (poids 20%)
        bb_next = nd.get("body_battery_charged")
        if pd.notna(bb_next) and float(bb_next) > 0:
            bb_score = float(bb_next) / 100
            recovery += (bb_score - 0.5) * 0.2
            weight_total += 0.2

        if weight_total == 0:
            continue

        recovery = max(0.05, min(0.95, recovery))

        # ── Mappe recovery → scores par type de séance ──
        for s_idx, session in enumerate(SESSION_CATALOGUE):
            if recovery >= 0.7:
                # Bien récupéré → séances intenses recommandées
                target_intensity = 0.75
                score = 1.0 - abs(session.intensity - target_intensity) * 0.6
            elif recovery >= 0.5:
                # Récupération correcte → séances modérées
                target_intensity = 0.5
                score = 1.0 - abs(session.intensity - target_intensity) * 0.8
            elif recovery >= 0.35:
                # Fatigue légère → séances légères
                target_intensity = 0.3
                score = 1.0 - abs(session.intensity - target_intensity) * 1.0
            else:
                # Fatigue importante → récupération active uniquement
                target_intensity = 0.15
                score = 1.0 - abs(session.intensity - target_intensity) * 1.2

            # Pénalité si séance à haut coût de récupération et athlète fatigué
            if recovery < 0.4 and session.recovery_cost > 0.6:
                score *= 0.4

            # Bonus séances de récupération si fatigué
            if recovery < 0.4 and session.category == "recuperation":
                score = min(0.95, score * 1.3)

            labels[idx, s_idx] = float(max(0.05, min(0.95, score)))

        n_labeled += 1

    print(f"  → {n_labeled}/{N} fenêtres labellisées depuis signaux de récupération")
    return labels


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class RecommendationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        jepa_model: JEPA,
        profiles: list[dict],
        races: list[dict],
        device: str = "cpu",
    ):
        data_dir_path = Path(data_dir)
        X_ctx = np.load(data_dir_path / "X_ctx.npy")
        meta  = pd.read_csv(data_dir_path / "meta.csv")

        self.X_ctx = torch.tensor(X_ctx, dtype=torch.float32)

        # Extrait les vecteurs latents JEPA
        print("[CRONOS] Extraction des vecteurs latents JEPA...")
        jepa_model.eval()
        with torch.no_grad():
            self.Z = jepa_model.encode(self.X_ctx.to(device)).cpu()
        print(f"  → {self.Z.shape[0]} vecteurs z_jepa extraits")

        # Labels automatiques depuis récupération
        print("[CRONOS] Construction des labels de récupération...")
        self.labels = build_labels_from_recovery(data_dir, meta)

        # Encode profil et course (premier disponible, TODO : mapper par user)
        self.profile = encode_athlete_profile(profiles[0] if profiles else {})
        self.race    = encode_race_context(races[0] if races else None)

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, idx):
        return (
            self.Z[idx],
            self.profile.squeeze(0),
            self.race.squeeze(0),
            self.labels[idx],
        )


# ─────────────────────────────────────────────────────────────
# Loss de ranking (ListNet + MSE)
# ─────────────────────────────────────────────────────────────

def combined_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    lambda_rank: float = 0.7,
    lambda_mse:  float = 0.3,
) -> torch.Tensor:
    p_labels = F.softmax(labels, dim=-1)
    p_scores = F.log_softmax(scores, dim=-1)
    rank_loss = -torch.sum(p_labels * p_scores, dim=-1).mean()
    mse_loss  = F.mse_loss(scores, labels)
    return lambda_rank * rank_loss + lambda_mse * mse_loss


# ─────────────────────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────────────────────

def train(
    data_dir:        str   = "data/processed",
    jepa_checkpoint: str   = "checkpoints/best_model.pt",
    profiles_path:   str   = "data/profiles.json",
    races_path:      str   = "data/races.json",
    epochs:          int   = 100,
    batch_size:      int   = 32,
    lr:              float = 1e-3,
    val_split:       float = 0.2,
    save_dir:        str   = "checkpoints",
    device:          str   = "auto",
    log_every:       int   = 10,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # ── JEPA ──
    print(f"[CRONOS] Chargement JEPA depuis {jepa_checkpoint}...")
    checkpoint = torch.load(jepa_checkpoint, map_location=device)
    jepa = JEPA()
    jepa.load_state_dict(checkpoint["model_state"])
    jepa.to(device)
    jepa.eval()
    print(f"  → val_loss JEPA : {checkpoint.get('val_loss', 'N/A'):.4f}")

    # ── Profils et courses ──
    profiles = []
    if Path(profiles_path).exists():
        with open(profiles_path) as f:
            profiles = json.load(f)
        print(f"  → {len(profiles)} profils chargés")
    else:
        print(f"  ⚠ Pas de profils — profil par défaut")

    races = []
    if Path(races_path).exists():
        with open(races_path) as f:
            races = json.load(f)
        print(f"  → {len(races)} courses chargées")
    else:
        print(f"  ⚠ Pas de courses planifiées")

    # ── Dataset ──
    dataset = RecommendationDataset(data_dir, jepa, profiles, races, device=device)
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train : {n_train} | Val : {n_val}")

    # ── Modèle ──
    model = CRONOSRecommender().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params recommender : {params:,}")

    # ── Boucle ──
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nDébut entraînement — {epochs} epochs\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for z_jepa, x_profile, x_race, labels in train_loader:
            z_jepa    = z_jepa.to(device)
            x_profile = x_profile.to(device)
            x_race    = x_race.to(device)
            labels    = labels.to(device)

            scores = model(z_jepa, x_profile, x_race)
            loss   = combined_loss(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for z_jepa, x_profile, x_race, labels in val_loader:
                scores = model(
                    z_jepa.to(device),
                    x_profile.to(device),
                    x_race.to(device)
                )
                val_losses.append(combined_loss(scores, labels.to(device)).item())

        t_loss = sum(train_losses) / len(train_losses)
        v_loss = sum(val_losses)   / len(val_losses)

        if epoch % log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Train {t_loss:.4f} | Val {v_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": v_loss,
            }, f"{save_dir}/best_recommender.pt")

    print(f"\n✓ Entraînement terminé — val loss : {best_val_loss:.4f}")
    print(f"  Modèle : {save_dir}/best_recommender.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      type=str,   default="data/processed")
    parser.add_argument("--jepa",      type=str,   default="checkpoints/best_model.pt")
    parser.add_argument("--profiles",  type=str,   default="data/profiles.json")
    parser.add_argument("--races",     type=str,   default="data/races.json")
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--batch_size",type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--save_dir",  type=str,   default="checkpoints")
    parser.add_argument("--device",    type=str,   default="auto")
    parser.add_argument("--log_every", type=int,   default=10)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        jepa_checkpoint=args.jepa,
        profiles_path=args.profiles,
        races_path=args.races,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device,
        log_every=args.log_every,
    )