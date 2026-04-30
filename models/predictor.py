"""
models/predictor.py — Prédicteur pour CRONOS JEPA

Prend le vecteur latent du contexte z_ctx et prédit
le vecteur latent de la cible z_tgt.

Architecture :
    z_ctx (batch, d_model)
        ↓
    Linear (d_model → d_ff)
        ↓
    GELU
        ↓
    Dropout
        ↓
    Linear (d_ff → d_model)
        ↓
    LayerNorm
        ↓
    z_pred (batch, d_model)
"""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    MLP qui prédit z_tgt depuis z_ctx.
    Simple mais efficace — le vrai travail est fait par l'encodeur.
    """

    def __init__(
        self,
        d_model: int = 64,    # dimension latente (doit matcher l'encodeur)
        d_ff: int = 128,      # dimension intermédiaire
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # Couche 1 : élargissement
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),

            # Couche 2 : rétrécissement
            nn.Linear(d_ff, d_model),
        )

        # Normalisation finale
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_ctx : (batch, d_model) — vecteur latent du contexte
        Returns:
            z_pred : (batch, d_model) — prédiction du vecteur latent cible
        """
        z_pred = self.net(z_ctx)
        z_pred = self.norm(z_pred)
        return z_pred


if __name__ == "__main__":
    # Test rapide
    predictor = Predictor()
    z_ctx = torch.randn(32, 64)    # batch de 32 vecteurs latents
    z_pred = predictor(z_ctx)
    print(f"Input  : {z_ctx.shape}")    # (32, 64)
    print(f"Output : {z_pred.shape}")   # (32, 64)
    print(f"Params : {sum(p.numel() for p in predictor.parameters()):,}")