"""
training/train_recommender.py — Entraînement du modèle de recommandation CRONOS.

Pipeline :
  1. Charge le modèle JEPA pré-entraîné → extrait z_jepa pour chaque fenêtre
  2. Charge les profils athlètes et courses planifiées
  3. Construit les labels depuis les RPE enregistrés
  4. Entraîne le SessionScorer avec une loss de ranking

Usage :
    python -m training.train_recommender \
        --data data/processed \
        --jepa checkpoints/best_model.pt \
        --epochs 100
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from models.jepa import JEPA
from recommendation.recommender import CRONOSRecommender
from recommendation.encoders import encode_athlete_profile, encode_race_context
from recommendation.session_types import SESSION_CATALOGUE, N_SESSIONS


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class RecommendationDataset(Dataset):
    """
    Dataset pour l'entraînement du recommender.

    Chaque sample contient :
    - z_jepa    : vecteur latent JEPA (état physiologique)
    - x_profile : features profil athlète
    - x_race    : features contexte course
    - labels    : vecteur (20,) avec les scores RPE normalisés par séance
                  (0 = pas de données, valeur normalisée sinon)
    """

    def __init__(
        self,
        data_dir: str,
        jepa_model: JEPA,
        profiles: list[dict],
        races: list[dict],
        rpe_data: list[dict],
        device: str = "cpu",
    ):
        data_dir = Path(data_dir)
        X_ctx = np.load(data_dir / "X_ctx.npy")
        self.X_ctx = torch.tensor(X_ctx, dtype=torch.float32)

        # Extrait les vecteurs latents JEPA
        print("[CRONOS] Extraction des vecteurs latents JEPA...")
        jepa_model.eval()
        with torch.no_grad():
            self.Z = jepa_model.encode(self.X_ctx.to(device)).cpu()
        print(f"  → {self.Z.shape[0]} vecteurs z_jepa extraits")

        # Encode les profils (on prend le premier profil disponible pour l'instant)
        # TODO: quand on aura plusieurs athlètes avec profils, mapper par user_id
        self.profile = encode_athlete_profile(profiles[0] if profiles else {})
        self.race    = encode_race_context(races[0] if races else None)

        # Construit les labels depuis les RPE
        self.labels = self._build_labels(rpe_data)

    def _build_labels(self, rpe_data: list[dict]) -> torch.Tensor:
        """
        Construit un vecteur de labels (N, 20) depuis les données RPE.

        Pour chaque fenêtre de 14 jours, on calcule pour chaque type de séance
        le RPE moyen observé (inversé : RPE faible = séance bien tolérée = bon label).
        """
        N = len(self.Z)
        labels = torch.full((N, N_SESSIONS), 0.5)  # 0.5 par défaut = incertain

        if not rpe_data:
            print("  ⚠ Pas de données RPE — labels par défaut (0.5)")
            return labels

        # Groupe les RPE par type de séance
        rpe_by_type: dict[str, list[float]] = {}
        for entry in rpe_data:
            activity_type = entry.get("activity_type", "unknown")
            rpe = entry.get("rpe")
            if rpe and 1 <= rpe <= 10:
                if activity_type not in rpe_by_type:
                    rpe_by_type[activity_type] = []
                rpe_by_type[activity_type].append(rpe)

        # Mappe les types d'activités Garmin vers les types de séances CRONOS
        TYPE_MAP = {
            "running":           [1, 9],   # Sortie longue Z2, Endurance fondamentale
            "trail_running":     [2, 13],  # Sortie longue trail, Trail technique
            "treadmill_running": [4, 5],   # Fractionné court, Fractionné long
            "track_running":     [4, 18],  # Fractionné court, VO2max
            "cycling":           [19],     # Cross-training
            "swimming":          [19],     # Cross-training
            "strength_training": [15],     # Gainage/renforcement
            "yoga":              [16],     # Mobilité
        }

        # Calcule les labels pour chaque fenêtre
        for session_idx, session in enumerate(SESSION_CATALOGUE):
            # Trouve les types Garmin correspondants
            garmin_types = []
            for garmin_type, session_ids in TYPE_MAP.items():
                if session.id in session_ids:
                    garmin_types.append(garmin_type)

            # Calcule le score moyen
            rpe_values = []
            for gt in garmin_types:
                rpe_values.extend(rpe_by_type.get(gt, []))

            if rpe_values:
                avg_rpe = sum(rpe_values) / len(rpe_values)
                # Inverse le RPE : RPE faible → séance bien tolérée → score élevé
                # RPE 1 → score 0.9, RPE 5 → score 0.5, RPE 10 → score 0.05
                score = max(0.05, 1.0 - (avg_rpe - 1) / 9 * 0.95)
                labels[:, session_idx] = score

        print(f"  → Labels construits depuis {len(rpe_data)} entrées RPE")
        return labels

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
# Loss de ranking (ListNet)
# ─────────────────────────────────────────────────────────────

def listnet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Loss ListNet pour le ranking.
    Minimise la cross-entropy entre la distribution des scores prédits
    et la distribution des labels cibles.
    """
    # Softmax sur les labels et les scores
    p_labels = F.softmax(labels, dim=-1)
    p_scores = F.log_softmax(scores, dim=-1)
    return -torch.sum(p_labels * p_scores, dim=-1).mean()


def combined_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    lambda_rank: float = 0.7,
    lambda_mse: float  = 0.3,
) -> torch.Tensor:
    """Loss combinée : ranking + MSE point-à-point."""
    rank_loss = listnet_loss(scores, labels)
    mse_loss  = F.mse_loss(scores, labels)
    return lambda_rank * rank_loss + lambda_mse * mse_loss


# ─────────────────────────────────────────────────────────────
# Boucle d'entraînement
# ─────────────────────────────────────────────────────────────

def train(
    data_dir: str = "data/processed",
    jepa_checkpoint: str = "checkpoints/best_model.pt",
    rpe_path: str = "data/rpe.json",
    profiles_path: str = "data/profiles.json",
    races_path: str = "data/races.json",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    save_dir: str = "checkpoints",
    device: str = "auto",
    log_every: int = 10,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # ── Charge le modèle JEPA ──
    print(f"[CRONOS] Chargement JEPA depuis {jepa_checkpoint}...")
    checkpoint = torch.load(jepa_checkpoint, map_location=device)
    jepa = JEPA()
    jepa.load_state_dict(checkpoint["model_state"])
    jepa.to(device)
    jepa.eval()
    print(f"  → JEPA chargé (val_loss: {checkpoint.get('val_loss', 'N/A'):.4f})")

    # ── Charge les données annexes ──
    profiles = []
    if Path(profiles_path).exists():
        with open(profiles_path) as f:
            profiles = json.load(f)
        print(f"  → {len(profiles)} profils athlètes chargés")
    else:
        print(f"  ⚠ {profiles_path} introuvable — profil par défaut utilisé")

    races = []
    if Path(races_path).exists():
        with open(races_path) as f:
            races = json.load(f)
        print(f"  → {len(races)} courses planifiées chargées")
    else:
        print(f"  ⚠ {races_path} introuvable — pas de contexte course")

    rpe_data = []
    if Path(rpe_path).exists():
        with open(rpe_path) as f:
            rpe_data = json.load(f)
        print(f"  → {len(rpe_data)} entrées RPE chargées")
    else:
        print(f"  ⚠ {rpe_path} introuvable — labels par défaut (0.5)")

    # ── Dataset ──
    dataset = RecommendationDataset(
        data_dir, jepa, profiles, races, rpe_data, device=device
    )
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

    # ── Entraînement ──
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nDébut entraînement — {epochs} epochs\n")

    for epoch in range(1, epochs + 1):
        # Train
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

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for z_jepa, x_profile, x_race, labels in val_loader:
                z_jepa    = z_jepa.to(device)
                x_profile = x_profile.to(device)
                x_race    = x_race.to(device)
                labels    = labels.to(device)
                scores    = model(z_jepa, x_profile, x_race)
                val_losses.append(combined_loss(scores, labels).item())

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

    print(f"\n✓ Entraînement terminé")
    print(f"  Meilleure val loss : {best_val_loss:.4f}")
    print(f"  Modèle sauvegardé  : {save_dir}/best_recommender.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     type=str, default="data/processed")
    parser.add_argument("--jepa",     type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--rpe",      type=str, default="data/rpe.json")
    parser.add_argument("--profiles", type=str, default="data/profiles.json")
    parser.add_argument("--races",    type=str, default="data/races.json")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device",   type=str, default="auto")
    parser.add_argument("--log_every",type=int, default=10)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        jepa_checkpoint=args.jepa,
        rpe_path=args.rpe,
        profiles_path=args.profiles,
        races_path=args.races,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device,
        log_every=args.log_every,
    )
