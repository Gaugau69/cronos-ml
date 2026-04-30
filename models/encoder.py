"""
models/encoder.py — Encodeur pour CRONOS JEPA

Architecture :
    Input (batch, 14, 12)
        ↓
    Linear projection (12 → d_model)
        ↓
    Positional Encoding
        ↓
    Transformer Encoder (n_layers, n_heads)
        ↓
    Mean Pooling
        ↓
    Output (batch, d_model)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Ajoute une information de position temporelle à chaque vecteur.
    Utilise des sinus/cosinus fixes (non appris).
    """

    def __init__(self, d_model: int, max_len: int = 14, dropout: float = 0.1):
        """
        Args:
            d_model  : dimension des vecteurs (64 par défaut)
            max_len  : longueur max de la séquence (14 jours)
            dropout  : taux de dropout
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Matrice de positional encoding de shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Vecteur des positions : [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

        # Facteur de division pour les sinus/cosinus
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Dimensions paires → sinus, dimensions impaires → cosinus
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Ajoute une dimension batch : (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer → sauvegardé avec le modèle mais pas entraîné
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, d_model)
        Returns:
            x : (batch, seq_len, d_model) avec position ajoutée
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    """
    Encodeur temporel pour les fenêtres de 14 jours.
    Transforme (batch, 14, 12) → (batch, d_model).
    """

    def __init__(
        self,
        n_features: int = 12,    # nombre de features par jour
        d_model: int = 64,       # dimension latente
        n_heads: int = 4,        # têtes d'attention
        n_layers: int = 2,       # couches Transformer
        d_ff: int = 256,         # dimension feed-forward interne
        dropout: float = 0.1,
        seq_len: int = 14,       # longueur de la fenêtre
    ):
        super().__init__()

        # 1. Projection linéaire : 12 features → d_model
        self.input_projection = nn.Linear(n_features, d_model)

        # 2. Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   # (batch, seq, features) au lieu de (seq, batch, features)
            norm_first=True,    # Pre-LN : plus stable à l'entraînement
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Normalisation finale
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, n_features) — fenêtre de 14 jours
        Returns:
            z : (batch, d_model) — vecteur latent
        """
        # 1. Projection : (batch, 14, 12) → (batch, 14, 64)
        x = self.input_projection(x)

        # 2. Positional Encoding : ajoute l'info de position
        x = self.pos_encoding(x)

        # 3. Transformer : (batch, 14, 64) → (batch, 14, 64)
        x = self.transformer(x)

        # 4. Mean Pooling : moyenne sur les 14 jours → (batch, 64)
        z = x.mean(dim=1)

        # 5. Normalisation
        z = self.norm(z)

        return z


if __name__ == "__main__":
    # Test rapide
    encoder = Encoder()
    x = torch.randn(32, 14, 12)   # batch de 32 fenêtres
    z = encoder(x)
    print(f"Input  : {x.shape}")   # (32, 14, 12)
    print(f"Output : {z.shape}")   # (32, 64)
    print(f"Params : {sum(p.numel() for p in encoder.parameters()):,}")