"""
recommendation/recommender.py — Modèle de recommandation de séances CRONOS.

Architecture :
    z_jepa    (64,)  ← état physiologique (JEPA)
    z_profile (32,)  ← profil athlète
    z_race    (16,)  ← contexte course
        ↓
    Concat (112,) → Transformer cross-attention → MLP → scores (20,)
        ↓
    Top-5 séances avec scores /100 et explications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recommendation.session_types import SESSION_CATALOGUE, N_SESSIONS, get_session_embeddings
from recommendation.encoders import AthleteProfileEncoder, RaceContextEncoder


class SessionScorer(nn.Module):
    """
    Prédit un score de pertinence (0-100) pour chaque type de séance
    en fonction de l'état physiologique, du profil et du contexte course.
    """

    def __init__(
        self,
        d_jepa: int = 64,
        d_profile: int = 32,
        d_race: int = 16,
        d_session: int = 7,
        d_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Encodeurs
        self.profile_encoder = AthleteProfileEncoder(dropout=dropout)
        self.race_encoder    = RaceContextEncoder(dropout=dropout)

        # Projection séances (features fixes → embedding)
        self.session_proj = nn.Linear(d_session, 32)

        # Dimension totale du contexte
        d_context = d_jepa + d_profile + d_race  # 64 + 32 + 16 = 112

        # Projection contexte
        self.context_proj = nn.Linear(d_context, d_hidden)

        # Cross-attention : chaque séance "interroge" le contexte
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            kdim=d_hidden,
            vdim=d_hidden,
            dropout=dropout,
            batch_first=True,
        )

        # MLP de scoring final
        self.scorer = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Embeddings fixes des séances (non entraînables)
        session_emb = get_session_embeddings()  # (20, 7)
        self.register_buffer("session_embeddings", session_emb)

    def forward(
        self,
        z_jepa:   torch.Tensor,  # (batch, 64)
        x_profile: torch.Tensor, # (batch, 12)
        x_race:    torch.Tensor, # (batch, 6)
    ) -> torch.Tensor:
        """
        Returns:
            scores : (batch, 20) — scores 0-1 pour chaque séance
        """
        batch = z_jepa.shape[0]

        # 1. Encode profil et course
        z_profile = self.profile_encoder(x_profile)  # (batch, 32)
        z_race    = self.race_encoder(x_race)         # (batch, 16)

        # 2. Contexte global
        context = torch.cat([z_jepa, z_profile, z_race], dim=-1)  # (batch, 112)
        context = self.context_proj(context).unsqueeze(1)          # (batch, 1, 128)

        # 3. Embeddings séances
        session_emb = self.session_proj(
            self.session_embeddings.unsqueeze(0).expand(batch, -1, -1)
        )  # (batch, 20, 32)

        # 4. Cross-attention : séances interrogent le contexte
        attended, _ = self.cross_attn(
            query=session_emb,
            key=context,
            value=context,
        )  # (batch, 20, 32)

        # 5. Score final pour chaque séance
        scores = self.scorer(attended).squeeze(-1)  # (batch, 20)

        return scores


class CRONOSRecommender(nn.Module):
    """
    Modèle complet de recommandation CRONOS.
    Wraps SessionScorer avec les encodeurs et génère les recommandations finales.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.scorer = SessionScorer(**kwargs)

    def forward(self, z_jepa, x_profile, x_race):
        return self.scorer(z_jepa, x_profile, x_race)

    @torch.no_grad()
    def recommend(
        self,
        z_jepa:    torch.Tensor,
        x_profile: torch.Tensor,
        x_race:    torch.Tensor,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Génère les top-k recommandations de séances.

        Returns:
            list de dicts avec :
                rank        : classement (1 = meilleur)
                session_id  : id dans SESSION_CATALOGUE
                name        : nom de la séance
                score       : score /100
                category    : catégorie
                description : description
                example     : exemple concret
                intensity   : intensité 0-1
                duration_min: durée typique
                distance_km : distance typique
        """
        self.eval()
        scores = self.forward(z_jepa, x_profile, x_race)  # (1, 20)
        scores = scores.squeeze(0)  # (20,)

        # Top-k indices triés par score décroissant
        top_indices = torch.argsort(scores, descending=True)[:top_k]

        recommendations = []
        for rank, idx in enumerate(top_indices):
            session = SESSION_CATALOGUE[idx.item()]
            recommendations.append({
                "rank":         rank + 1,
                "session_id":   session.id,
                "name":         session.name,
                "score":        round(scores[idx].item() * 100, 1),
                "category":     session.category,
                "description":  session.description,
                "example":      session.example,
                "intensity":    session.intensity,
                "duration_min": session.duration_min,
                "distance_km":  session.distance_km,
            })

        return recommendations


if __name__ == "__main__":
    model = CRONOSRecommender()

    # Test avec données aléatoires
    z_jepa    = torch.randn(1, 64)
    x_profile = torch.rand(1, 12)
    x_race    = torch.rand(1, 6)

    recs = model.recommend(z_jepa, x_profile, x_race, top_k=5)

    print("CRONOS — Top 5 séances recommandées :")
    print("=" * 50)
    for r in recs:
        print(f"\n#{r['rank']} — {r['name']} ({r['score']}/100)")
        print(f"   Catégorie  : {r['category']}")
        print(f"   Durée      : {r['duration_min']} min | {r['distance_km']} km")
        print(f"   Intensité  : {r['intensity']*100:.0f}%")
        print(f"   Exemple    : {r['example']}")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParams entraînables : {params:,}")
