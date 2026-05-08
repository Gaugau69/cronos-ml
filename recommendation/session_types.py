"""
recommendation/session_types.py — Catalogue des types de séances.

Définit les 20 types de séances possibles que CRONOS peut recommander.
Chaque séance a des caractéristiques fixes utilisées pour l'encodage.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionType:
    id: int
    name: str
    category: str          # endurance / intensite / recuperation / force / specifique
    intensity: float       # 0-1 (Z1=0.2, Z2=0.4, Z3=0.6, Z4=0.8, Z5=1.0)
    duration_min: float    # durée typique en minutes
    distance_km: float     # distance typique en km (0 si non applicable)
    elevation_m: float     # dénivelé typique (0 si route)
    recovery_cost: float   # coût en récupération 0-1
    description: str
    example: str           # exemple concret


SESSION_CATALOGUE = [
    SessionType(0,  "Récupération active",       "recuperation", 0.2, 40,   6,   0,   0.1, "Footing très lent Z1, conversations possibles", "40min à 6:30/km"),
    SessionType(1,  "Sortie longue Z2",           "endurance",    0.4, 120,  20,  0,   0.4, "Course longue en zone aérobie, allure confort", "2h à 5:30/km"),
    SessionType(2,  "Sortie longue trail",        "endurance",    0.4, 150,  25,  800, 0.5, "Sortie longue en nature avec dénivelé", "2h30 avec 800m D+"),
    SessionType(3,  "Tempo continu",              "intensite",    0.7, 50,   10,  0,   0.6, "Course soutenue au seuil anaérobie", "50min à 4:20/km"),
    SessionType(4,  "Fractionné court",           "intensite",    0.9, 55,   10,  0,   0.7, "Répétitions courtes à haute intensité", "10x400m avec récup 90s"),
    SessionType(5,  "Fractionné long",            "intensite",    0.8, 60,   12,  0,   0.7, "Répétitions longues au seuil", "5x1000m avec récup 2min"),
    SessionType(6,  "Côtes",                      "force",        0.85, 50,  8,   300, 0.7, "Répétitions en montée pour la force", "10x200m de côte"),
    SessionType(7,  "Fartlek",                    "intensite",    0.7, 55,   10,  0,   0.5, "Variations libres d'allure", "55min avec accélérations libres"),
    SessionType(8,  "Progression",                "endurance",    0.6, 60,   12,  0,   0.5, "Course avec augmentation progressive de l'allure", "60min en finissant vite"),
    SessionType(9,  "Endurance fondamentale",     "endurance",    0.5, 75,   13,  0,   0.3, "Course à allure modérée Z2/Z3", "75min à 5:00/km"),
    SessionType(10, "Spécifique marathon",        "specifique",   0.7, 90,   18,  0,   0.6, "Allure marathon en condition de course", "90min à allure objectif"),
    SessionType(11, "Spécifique semi",            "specifique",   0.75, 70,  14,  0,   0.6, "Allure semi-marathon en condition de course", "70min à allure objectif"),
    SessionType(12, "Spécifique 10km",            "specifique",   0.8, 55,   11,  0,   0.6, "Allure 10km en condition de course", "55min à allure objectif"),
    SessionType(13, "Trail technique",            "specifique",   0.6, 90,   15,  500, 0.5, "Travail technique en terrain varié", "90min sur sentiers techniques"),
    SessionType(14, "Ultra endurance",            "endurance",    0.4, 240,  40,  1500,0.8, "Sortie très longue pour ultra-trail", "4h avec 1500m D+"),
    SessionType(15, "Gainage et renforcement",    "force",        0.4, 45,   0,   0,   0.2, "Séance de renforcement musculaire", "45min PPG/gainage"),
    SessionType(16, "Mobilité et récupération",   "recuperation", 0.1, 30,   0,   0,   0.0, "Stretching, yoga, mobilité", "30min yoga/stretching"),
    SessionType(17, "Seuil lactique",             "intensite",    0.75, 55,  11,  0,   0.65,"Course au seuil lactique", "20min échauffement + 3x10min seuil"),
    SessionType(18, "VO2max",                     "intensite",    0.95, 50,  10,  0,   0.8, "Intervalles à VO2max", "6x3min à 95% FCmax"),
    SessionType(19, "Repos actif / cross-training","recuperation", 0.2, 45,   0,   0,   0.1, "Natation, vélo, elliptique", "45min vélo ou natation"),
]

# Matrice d'encodage : (20, 7) — une ligne par séance
import torch

def get_session_embeddings() -> torch.Tensor:
    """
    Retourne un tenseur (20, 7) avec les caractéristiques normalisées
    de chaque type de séance.
    """
    features = []
    for s in SESSION_CATALOGUE:
        features.append([
            s.intensity,
            s.duration_min / 240,        # normalisé par 4h
            s.distance_km / 40,          # normalisé par 40km
            s.elevation_m / 1500,        # normalisé par 1500m
            s.recovery_cost,
            1.0 if s.category == "endurance"   else 0.0,
            1.0 if s.category == "intensite"   else 0.0,
        ])
    return torch.tensor(features, dtype=torch.float32)


N_SESSIONS = len(SESSION_CATALOGUE)  # 20
