"""
models/jepa.py — Architecture JEPA complète pour CRONOS

Flow :
    X_ctx ──► Encodeur contexte ──► z_ctx ──► Prédicteur ──► z_pred
                                                                  │
    X_tgt ──► Encodeur target ──► z_tgt ◄──────────────── Loss(z_pred, z_tgt)

L'encodeur target est une copie EMA de l'encodeur contexte.
Il ne s'entraîne pas par backprop — il suit lentement l'encodeur contexte.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.predictor import Predictor


class JEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture pour la prédiction de récupération.

    Args:
        n_features  : nombre de features par jour (12)
        d_model     : dimension latente (64)
        n_heads     : têtes d'attention dans le Transformer (4)
        n_layers    : couches Transformer (2)
        d_ff_enc    : dimension feed-forward de l'encodeur (256)
        d_ff_pred   : dimension feed-forward du prédicteur (128)
        dropout     : taux de dropout (0.1)
        seq_len     : longueur de la fenêtre (14 jours)
        ema_tau     : taux EMA pour l'encodeur target (0.996)
    """

    def __init__(
        self,
        n_features: int = 12,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff_enc: int = 256,
        d_ff_pred: int = 128,
        dropout: float = 0.1,
        seq_len: int = 14,
        ema_tau: float = 0.996,
    ):
        super().__init__()

        self.ema_tau = ema_tau

        # ── Encodeur contexte (s'entraîne par backprop) ──
        self.context_encoder = Encoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff_enc,
            dropout=dropout,
            seq_len=seq_len,
        )

        # ── Encodeur target (copie EMA, pas de backprop) ──
        self.target_encoder = copy.deepcopy(self.context_encoder)

        # L'encodeur target ne s'entraîne jamais par gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # ── Prédicteur ──
        self.predictor = Predictor(
            d_model=d_model,
            d_ff=d_ff_pred,
            dropout=dropout,
        )

    @torch.no_grad()
    def update_target_encoder(self):
        """
        Met à jour l'encodeur target par EMA.
        Appelé après chaque step d'entraînement.

        θ_target = τ * θ_target + (1 - τ) * θ_context
        """
        for param_ctx, param_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            param_tgt.data = (
                self.ema_tau * param_tgt.data
                + (1 - self.ema_tau) * param_ctx.data
            )

    def forward(self, x_ctx: torch.Tensor, x_tgt: torch.Tensor):
        """
        Forward pass complet.

        Args:
            x_ctx : (batch, seq_len, n_features) — fenêtre contexte
            x_tgt : (batch, seq_len, n_features) — fenêtre cible

        Returns:
            z_pred : (batch, d_model) — prédiction du vecteur latent cible
            z_tgt  : (batch, d_model) — vecteur latent cible (stop gradient)
            loss   : scalar — loss cosinus entre z_pred et z_tgt
        """
        # 1. Encode le contexte
        z_ctx = self.context_encoder(x_ctx)

        # 2. Prédit le vecteur latent cible
        z_pred = self.predictor(z_ctx)

        # 3. Encode la cible (sans gradient — stop gradient)
        with torch.no_grad():
            z_tgt = self.target_encoder(x_tgt)

        # 4. Calcule la loss
        loss = self.loss(z_pred, z_tgt)

        return z_pred, z_tgt, loss

    def loss(self, z_pred: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        """
        Loss cosinus — mesure l'angle entre z_pred et z_tgt.
        = 0 si les vecteurs sont identiques
        = 2 si les vecteurs sont opposés

        On utilise 2 - cosine_similarity pour avoir une loss positive
        qu'on veut minimiser.
        """
        # Normalise les vecteurs (norme = 1)
        z_pred = F.normalize(z_pred, dim=-1)
        z_tgt = F.normalize(z_tgt, dim=-1)

        # Similarité cosinus : produit scalaire de vecteurs normalisés
        cosine_sim = (z_pred * z_tgt).sum(dim=-1)  # (batch,)

        # Loss = 2 - similarité (entre 0 et 4, minimisé quand sim = 1)
        loss = (2 - cosine_sim).mean()

        return loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode une fenêtre avec l'encodeur contexte.
        Utilisé à l'inférence pour extraire les représentations.

        Args:
            x : (batch, seq_len, n_features)
        Returns:
            z : (batch, d_model)
        """
        return self.context_encoder(x)


if __name__ == "__main__":
    # Test rapide
    model = JEPA()

    # Simule un batch de 32 paires (contexte, cible)
    x_ctx = torch.randn(32, 14, 12)
    x_tgt = torch.randn(32, 14, 12)

    z_pred, z_tgt, loss = model(x_ctx, x_tgt)

    print(f"x_ctx  : {x_ctx.shape}")    # (32, 14, 12)
    print(f"z_pred : {z_pred.shape}")   # (32, 64)
    print(f"z_tgt  : {z_tgt.shape}")    # (32, 64)
    print(f"loss   : {loss.item():.4f}")

    # Test mise à jour EMA
    model.update_target_encoder()
    print("EMA update OK")

    # Compte les paramètres entraînables
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params entraînables : {trainable:,} / {total:,}")