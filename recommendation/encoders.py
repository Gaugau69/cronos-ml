"""
recommendation/encoders.py — Encodeurs du profil athlète et du contexte course.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Encodeur profil athlète
# ─────────────────────────────────────────────────────────────

class AthleteProfileEncoder(nn.Module):
    """
    Encode le profil athlète en un vecteur latent.

    Entrée : vecteur de 12 features numériques
        0  level_encoded       — niveau 0-3 (debutant/inter/avance/elite)
        1  sport_type_encoded  — type 0-4 (route/trail/ultra/tri/mixte)
        2  years_running_norm  — années de pratique / 20
        3  weekly_km_norm      — km/semaine / 150
        4  weekly_sessions_norm— séances/semaine / 10
        5  long_run_norm       — sortie longue / 50km
        6  vo2max_norm         — VO2max / 80
        7  best_5k_norm        — 5km en min / 30
        8  best_marathon_norm  — marathon en min / 300
        9  max_weekly_km_norm  — km max / 200
        10 goal_encoded        — objectif 0-3 (finir/chrono/podium/prog)
        11 target_dist_encoded — distance cible 0-4 (5k/10k/semi/mara/ultra)
    """

    INPUT_DIM = 12
    OUTPUT_DIM = 32

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.OUTPUT_DIM),
            nn.LayerNorm(self.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, 12)
        Returns:
            z : (batch, 32)
        """
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Encodeur contexte course
# ─────────────────────────────────────────────────────────────

class RaceContextEncoder(nn.Module):
    """
    Encode le contexte de la prochaine course en un vecteur latent.

    Entrée : vecteur de 6 features
        0  days_to_race_norm   — jours avant la course / 365
        1  distance_norm       — distance km / 100
        2  elevation_norm      — dénivelé / 3000
        3  goal_encoded        — objectif 0-2 (finir/chrono/podium)
        4  priority_encoded    — priorité 0-2 (C/B/A)
        5  has_race            — 1 si course planifiée, 0 sinon
    """

    INPUT_DIM = 6
    OUTPUT_DIM = 16

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, self.OUTPUT_DIM),
            nn.LayerNorm(self.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, 6)
        Returns:
            z : (batch, 16)
        """
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Helpers de préparation des features
# ─────────────────────────────────────────────────────────────

LEVEL_MAP    = {"debutant": 0, "intermediaire": 1, "avance": 2, "elite": 3}
SPORT_MAP    = {"route": 0, "trail": 1, "ultra": 2, "triathlon": 3, "mixte": 4}
GOAL_MAP     = {"plaisir": 0, "progression": 1, "chrono": 2, "competition": 3, "finir": 0, "podium": 3}
DIST_MAP     = {"5k": 0, "10k": 1, "semi": 2, "marathon": 3, "ultra": 4}
PRIORITY_MAP = {"C": 0, "B": 1, "A": 2}
RACE_GOAL_MAP= {"finir": 0, "chrono": 1, "podium": 2}


def encode_athlete_profile(profile: dict) -> torch.Tensor:
    """
    Transforme un dict profil athlète en tenseur (1, 12).
    Les valeurs manquantes sont remplacées par des valeurs par défaut.
    """
    features = [
        LEVEL_MAP.get(profile.get("level", "intermediaire"), 1) / 3,
        SPORT_MAP.get(profile.get("sport_type", "route"), 0) / 4,
        min((profile.get("years_running") or 3) / 20, 1.0),
        min((profile.get("weekly_km") or 40) / 150, 1.0),
        min((profile.get("weekly_sessions") or 4) / 10, 1.0),
        min((profile.get("long_run_km") or 15) / 50, 1.0),
        min((profile.get("vo2max_estimated") or 45) / 80, 1.0),
        min((profile.get("best_5k_min") or 25) / 30, 1.0),
        min((profile.get("best_marathon_min") or 240) / 300, 1.0),
        min((profile.get("max_weekly_km") or 60) / 200, 1.0),
        GOAL_MAP.get(profile.get("primary_goal", "finir"), 0) / 3,
        DIST_MAP.get(profile.get("target_distance", "10k"), 1) / 4,
    ]
    return torch.tensor([features], dtype=torch.float32)


def encode_race_context(race: dict | None) -> torch.Tensor:
    """
    Transforme un dict course planifiée en tenseur (1, 6).
    Si pas de course → vecteur de zéros sauf has_race=0.
    """
    if not race:
        return torch.zeros(1, 6)

    features = [
        min((race.get("days_to_race") or 90) / 365, 1.0),
        min((race.get("distance_km") or 10) / 100, 1.0),
        min((race.get("elevation_m") or 0) / 3000, 1.0),
        RACE_GOAL_MAP.get(race.get("goal_type", "finir"), 0) / 2,
        PRIORITY_MAP.get(race.get("priority", "B"), 1) / 2,
        1.0,  # has_race = True
    ]
    return torch.tensor([features], dtype=torch.float32)
