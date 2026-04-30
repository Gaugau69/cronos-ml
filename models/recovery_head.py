"""
models/recovery_head.py — Tête de prédiction supervisée

Utilisée en finetuning après le pré-entraînement JEPA.
Prédit un score de récupération (0-100) depuis le vecteur latent.

Usage :
    # Pré-entraînement JEPA (auto-supervisé)
    jepa = JEPA()
    train_jepa(jepa, X_ctx, X_tgt)

    # Finetuning supervisé
    head = RecoveryHead(d_model=64)
    z = jepa.encode(x)          # extrait les représentations
    score = head(z)             # prédit le score 0-100
    loss = mse_loss(score, y)   # y = labels de récupération
"""

import torch
import torch.nn as nn


class RecoveryHead(nn.Module):
    """
    Tête MLP pour prédire le score de récupération (0-100).

    Architecture :
        z (d_model,)
            ↓
        Linear (d_model → d_model // 2)
            ↓
        GELU + Dropout
            ↓
        Linear (d_model // 2 → 1)
            ↓
        Sigmoid × 100
            ↓
        score (0-100)
    """

    def __init__(self, d_model: int = 64, dropout: float = 0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (batch, d_model) — vecteur latent JEPA
        Returns:
            score : (batch,) — score de récupération entre 0 et 100
        """
        return self.net(z).squeeze(-1) * 100


class RecoveryModel(nn.Module):
    """
    Modèle complet pour l'inférence : encodeur JEPA + tête supervisée.
    Utilisé après le finetuning.
    """

    def __init__(self, encoder: nn.Module, head: RecoveryHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, n_features)
        Returns:
            score : (batch,) — score de récupération 0-100
        """
        z = self.encoder(x)
        return self.head(z)


if __name__ == "__main__":
    from models.encoder import Encoder

    encoder = Encoder()
    head = RecoveryHead()
    model = RecoveryModel(encoder, head)

    x = torch.randn(32, 14, 12)
    score = model(x)
    print(f"Input  : {x.shape}")
    print(f"Score  : {score.shape}")   # (32,)
    print(f"Range  : [{score.min():.1f}, {score.max():.1f}]")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")