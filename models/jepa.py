"""
models/jepa.py — Architecture JEPA v2 pour CRONOS

Améliorations vs v1 :
  1. Encodeur amélioré (embeddings séparés, recent bias, stochastic depth)
  2. Masquage aléatoire style MAE sur le contexte
  3. Loss combinée : cosinus + variance + covariance (VICReg)
  4. EMA avec warmup progressif du tau

Flow :
    X_ctx ──► Masquage ──► Encodeur contexte ──► z_ctx ──► Prédicteur ──► z_pred
                                                                               │
    X_tgt ──────────────► Encodeur target  ──► z_tgt ◄──────────────── Loss VICReg
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.predictor import Predictor


# ─────────────────────────────────────────────────────────────
# Masquage aléatoire style MAE
# ─────────────────────────────────────────────────────────────

def random_masking(x: torch.Tensor, mask_ratio: float = 0.25) -> torch.Tensor:
    """
    Masque aléatoirement mask_ratio des jours dans la fenêtre.
    Les jours masqués sont remplacés par des zéros.

    Args:
        x          : (batch, seq_len, n_features)
        mask_ratio : proportion de jours à masquer (0.25 = 25%)
    Returns:
        x_masked   : (batch, seq_len, n_features) avec certains jours à 0
    """
    batch, seq_len, n_features = x.shape
    n_mask = int(seq_len * mask_ratio)

    if n_mask == 0:
        return x

    x_masked = x.clone()

    for b in range(batch):
        # Choisit aléatoirement n_mask indices à masquer
        mask_indices = torch.randperm(seq_len, device=x.device)[:n_mask]
        x_masked[b, mask_indices, :] = 0.0

    return x_masked


# ─────────────────────────────────────────────────────────────
# Loss VICReg (Variance-Invariance-Covariance Regularization)
# ─────────────────────────────────────────────────────────────

def vicreg_loss(
    z_pred: torch.Tensor,
    z_tgt: torch.Tensor,
    lambda_inv: float = 25.0,
    lambda_var: float = 25.0,
    lambda_cov: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Loss VICReg — bien meilleure que la simple cosine loss.

    3 termes :
    1. Invariance  : z_pred doit être proche de z_tgt (MSE dans l'espace latent)
    2. Variance    : les représentations doivent rester variées (évite le collapse)
    3. Covariance  : les dimensions du vecteur latent doivent être décorrélées

    Args:
        z_pred, z_tgt : (batch, d_model)
    Returns:
        loss   : scalar total
        losses : dict des 3 termes pour le logging
    """
    batch, d = z_pred.shape

    # ── 1. Invariance : MSE entre prédiction et cible ──
    inv_loss = F.mse_loss(z_pred, z_tgt)

    # ── 2. Variance : std de chaque dimension ≥ 1 ──
    # Si std < 1 → les représentations s'effondrent → on pénalise
    std_pred = torch.sqrt(z_pred.var(dim=0) + 1e-4)
    std_tgt  = torch.sqrt(z_tgt.var(dim=0)  + 1e-4)
    var_loss = F.relu(1 - std_pred).mean() + F.relu(1 - std_tgt).mean()

    # ── 3. Covariance : les dimensions doivent être décorrélées ──
    # Normalise par la moyenne
    z_pred_centered = z_pred - z_pred.mean(dim=0)
    z_tgt_centered  = z_tgt  - z_tgt.mean(dim=0)

    # Matrices de covariance : (d, d)
    cov_pred = (z_pred_centered.T @ z_pred_centered) / (batch - 1)
    cov_tgt  = (z_tgt_centered.T  @ z_tgt_centered)  / (batch - 1)

    # On veut que les éléments hors-diagonale soient proches de 0
    def off_diagonal(mat):
        n = mat.shape[0]
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    cov_loss = (off_diagonal(cov_pred).pow(2).sum() +
                off_diagonal(cov_tgt).pow(2).sum()) / d

    # ── Loss totale ──
    loss = lambda_inv * inv_loss + lambda_var * var_loss + lambda_cov * cov_loss

    return loss, {
        "inv_loss": inv_loss.item(),
        "var_loss": var_loss.item(),
        "cov_loss": cov_loss.item(),
        "total":    loss.item(),
    }


# ─────────────────────────────────────────────────────────────
# JEPA v2
# ─────────────────────────────────────────────────────────────

class JEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture v2 pour CRONOS.
    """

    def __init__(
        self,
        n_features: int = 12,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff_enc: int = 256,
        d_ff_pred: int = 128,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        seq_len: int = 14,
        ema_tau: float = 0.996,
        mask_ratio: float = 0.25,
    ):
        super().__init__()

        self.ema_tau    = ema_tau
        self.mask_ratio = mask_ratio

        # ── Encodeur contexte (entraîné par backprop) ──
        self.context_encoder = Encoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff_enc,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            seq_len=seq_len,
            use_augmentation=True,
        )

        # ── Encodeur target (EMA, pas de backprop, pas d'augmentation) ──
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.augmentation = nn.Identity()  # Pas d'augmentation sur la cible
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # ── Prédicteur ──
        self.predictor = Predictor(
            d_model=d_model,
            d_ff=d_ff_pred,
            dropout=dropout,
        )

    @torch.no_grad()
    def update_target_encoder(self, tau: float = None):
        """
        Met à jour l'encodeur target par EMA.
        θ_target = τ * θ_target + (1 - τ) * θ_context
        """
        if tau is None:
            tau = self.ema_tau
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_tgt.data = tau * p_tgt.data + (1 - tau) * p_ctx.data

    def forward(
        self,
        x_ctx: torch.Tensor,
        x_tgt: torch.Tensor,
        use_masking: bool = True,
    ):
        """
        Args:
            x_ctx       : (batch, seq_len, n_features)
            x_tgt       : (batch, seq_len, n_features)
            use_masking : applique le masquage aléatoire sur x_ctx
        Returns:
            z_pred  : (batch, d_model)
            z_tgt   : (batch, d_model)
            loss    : scalar
            losses  : dict des composantes de la loss
        """
        # 1. Masquage aléatoire du contexte
        if use_masking and self.training:
            x_ctx_masked = random_masking(x_ctx, self.mask_ratio)
        else:
            x_ctx_masked = x_ctx

        # 2. Encode le contexte (avec augmentation si training)
        z_ctx = self.context_encoder(x_ctx_masked)

        # 3. Prédit le vecteur latent cible
        z_pred = self.predictor(z_ctx)

        # 4. Encode la cible (sans gradient, sans augmentation)
        with torch.no_grad():
            z_tgt = self.target_encoder(x_tgt)

        # 5. Loss VICReg
        loss, losses = vicreg_loss(z_pred, z_tgt)

        return z_pred, z_tgt, loss, losses

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode une fenêtre à l'inférence (sans augmentation ni masquage)."""
        self.eval()
        with torch.no_grad():
            return self.context_encoder(x)


if __name__ == "__main__":
    model = JEPA()

    x_ctx = torch.randn(32, 14, 12)
    x_tgt = torch.randn(32, 14, 12)

    z_pred, z_tgt, loss, losses = model(x_ctx, x_tgt)

    print(f"x_ctx  : {x_ctx.shape}")
    print(f"z_pred : {z_pred.shape}")
    print(f"z_tgt  : {z_tgt.shape}")
    print(f"Loss   : {loss.item():.4f}")
    print(f"  inv  : {losses['inv_loss']:.4f}")
    print(f"  var  : {losses['var_loss']:.4f}")
    print(f"  cov  : {losses['cov_loss']:.4f}")

    model.update_target_encoder()
    print("EMA update OK")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Params entraînables : {trainable:,} / {total:,}")