"""
models/encoder.py — Encodeur amélioré pour CRONOS JEPA v2

Améliorations vs v1 :
  1. Embeddings séparés repos / séance
  2. Positional Encoding avec recent bias (appris)
  3. Stochastic Depth (drop path) pour la régularisation
  4. Augmentation de données (bruit gaussien, jitter temporel)

Architecture :
    Input (batch, 14, 12)
        ↓
    Split features → repos (5) + séance (7)
        ↓
    Projections séparées → concat (batch, 14, d_model)
        ↓
    Positional Encoding + Recent Bias
        ↓
    Transformer Encoder avec Stochastic Depth
        ↓
    Mean Pooling pondéré par récence
        ↓
    Output (batch, d_model)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Indices des features (ordre fixe dans FEATURE_NAMES)
# ─────────────────────────────────────────────────────────────

# Features de repos : hrv_rmssd, hr_rest, sleep_duration, sleep_quality, is_rest_day
REST_INDICES   = [0, 1, 2, 3, 11]

# Features de séance : hr_mean, hr_drift, pace_mean, pace_hr_ratio, duration, elevation_gain, training_load
SEANCE_INDICES = [4, 5, 6, 7, 8, 9, 10]

N_REST   = len(REST_INDICES)    # 5
N_SEANCE = len(SEANCE_INDICES)  # 7


# ─────────────────────────────────────────────────────────────
# 1. Stochastic Depth (Drop Path)
# ─────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """
    Stochastic Depth : éteint aléatoirement des résidus entiers pendant l'entraînement.
    Chaque couche du Transformer a une probabilité drop_prob d'être ignorée.
    Améliore la généralisation sur les petits datasets.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # Crée un masque binaire de shape (batch, 1, 1) pour broadcaster
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


# ─────────────────────────────────────────────────────────────
# 2. Transformer Layer avec Stochastic Depth
# ─────────────────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    """
    Couche Transformer avec :
    - Pre-LayerNorm (plus stable)
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - Stochastic Depth sur les deux résidus
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # Self-Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-Forward
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x         : (batch, seq_len, d_model)
            attn_bias : (seq_len, seq_len) biais d'attention optionnel (recent bias)
        Returns:
            x         : (batch, seq_len, d_model)
        """
        # Self-Attention avec résidu + stochastic depth
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_bias)
        x = residual + self.drop_path(attn_out)

        # Feed-Forward avec résidu + stochastic depth
        residual = x
        x = self.norm2(x)
        x = residual + self.drop_path(self.ff(x))

        return x


# ─────────────────────────────────────────────────────────────
# 3. Recent Bias — pondération temporelle apprise
# ─────────────────────────────────────────────────────────────

class RecentBias(nn.Module):
    """
    Biais d'attention appris qui favorise les jours récents.

    Pour chaque paire (query_pos, key_pos), on apprend un biais
    qui encode la distance temporelle relative.

    Les positions récentes (proches de seq_len) auront naturellement
    plus d'influence sur la prédiction J+1.
    """

    def __init__(self, seq_len: int = 14, n_heads: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.n_heads = n_heads

        # Biais appris par tête : (n_heads, seq_len, seq_len)
        # Initialisé avec un prior de récence (les positions récentes ont un biais positif)
        self.bias = nn.Parameter(torch.zeros(n_heads, seq_len, seq_len))

        # Initialisation avec prior de récence
        with torch.no_grad():
            for i in range(seq_len):
                for j in range(seq_len):
                    # Les clés récentes (grand j) reçoivent un biais positif
                    self.bias[:, i, j] = j / seq_len * 0.5

    def forward(self) -> torch.Tensor:
        """
        Returns:
            bias : (n_heads * batch, seq_len, seq_len) pour nn.MultiheadAttention
        """
        # nn.MultiheadAttention attend (seq_len, seq_len) ou (batch*heads, seq_len, seq_len)
        # On retourne (seq_len, seq_len) moyenné sur les têtes pour simplicité
        return self.bias.mean(dim=0)  # (seq_len, seq_len)


# ─────────────────────────────────────────────────────────────
# 4. Positional Encoding sinusoïdal
# ─────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal fixe."""

    def __init__(self, d_model: int, max_len: int = 14, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────
# 5. Augmentation de données
# ─────────────────────────────────────────────────────────────

class TimeSeriesAugmentation(nn.Module):
    """
    Augmentations légères pour les séries temporelles physiologiques.

    Appliquées uniquement pendant l'entraînement :
    1. Bruit gaussien (simule variabilité du capteur)
    2. Scaling aléatoire (simule différences inter-individuelles)
    3. Dropout temporel (simule jours sans montre)
    """

    def __init__(
        self,
        noise_std: float = 0.02,
        scale_range: tuple = (0.95, 1.05),
        temporal_dropout: float = 0.05,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.temporal_dropout = temporal_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        # 1. Bruit gaussien
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise

        # 2. Scaling aléatoire par feature
        scale = torch.empty(x.shape[-1], device=x.device).uniform_(*self.scale_range)
        x = x * scale

        # 3. Dropout temporel — met certains jours à 0 (jour sans montre)
        if self.temporal_dropout > 0:
            mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > self.temporal_dropout
            x = x * mask

        return x


# ─────────────────────────────────────────────────────────────
# 6. Encodeur principal
# ─────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Encodeur amélioré CRONOS v2.

    Transforme (batch, seq_len, n_features) → (batch, d_model).
    """

    def __init__(
        self,
        n_features: int = 12,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        seq_len: int = 14,
        use_augmentation: bool = True,
    ):
        super().__init__()

        assert d_model % 2 == 0, "d_model doit être pair pour les embeddings séparés"
        d_half = d_model // 2  # 32 pour chaque groupe

        # ── Augmentation ──
        self.augmentation = TimeSeriesAugmentation() if use_augmentation else nn.Identity()

        # ── Embeddings séparés repos / séance ──
        self.rest_proj   = nn.Linear(N_REST,   d_half)
        self.seance_proj = nn.Linear(N_SEANCE, d_half)

        # Normalisation après projection
        self.proj_norm = nn.LayerNorm(d_model)

        # ── Positional Encoding ──
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # ── Recent Bias ──
        self.recent_bias = RecentBias(seq_len=seq_len, n_heads=n_heads)

        # ── Transformer avec Stochastic Depth ──
        # Drop path rate augmente progressivement avec la profondeur
        dp_rates = [drop_path_rate * i / max(n_layers - 1, 1) for i in range(n_layers)]

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                drop_path=dp_rates[i],
            )
            for i in range(n_layers)
        ])

        # ── Normalisation finale ──
        self.norm = nn.LayerNorm(d_model)

        # ── Poids de pooling appris (pondération par récence) ──
        # Les derniers jours ont plus de poids par défaut
        self.pool_weights = nn.Parameter(torch.ones(seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, n_features)
        Returns:
            z : (batch, d_model)
        """
        # 1. Augmentation (seulement en training)
        x = self.augmentation(x)

        # 2. Split features repos / séance
        x_rest   = x[:, :, REST_INDICES]    # (batch, seq_len, 5)
        x_seance = x[:, :, SEANCE_INDICES]  # (batch, seq_len, 7)

        # 3. Projections séparées puis concat
        z_rest   = self.rest_proj(x_rest)    # (batch, seq_len, 32)
        z_seance = self.seance_proj(x_seance) # (batch, seq_len, 32)
        z = torch.cat([z_rest, z_seance], dim=-1)  # (batch, seq_len, 64)
        z = self.proj_norm(z)

        # 4. Positional Encoding
        z = self.pos_encoding(z)

        # 5. Recent bias pour l'attention
        attn_bias = self.recent_bias()  # (seq_len, seq_len)

        # 6. Transformer layers
        for layer in self.layers:
            z = layer(z, attn_bias=attn_bias)

        # 7. Normalisation
        z = self.norm(z)

        # 8. Weighted Mean Pooling (poids appris + softmax)
        weights = F.softmax(self.pool_weights, dim=0)  # (seq_len,)
        z = (z * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (batch, d_model)

        return z


if __name__ == "__main__":
    encoder = Encoder()
    x = torch.randn(32, 14, 12)
    z = encoder(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {z.shape}")
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Params : {trainable:,}")