"""
recommendation/session_types.py — Catalogue enrichi des séances CRONOS v2

45 types de séances couvrant tous les niveaux et situations.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class SessionType:
    id: int
    name: str
    category: str           # endurance / intensite / recuperation / force / specifique / repos
    intensity: float        # 0-1
    duration_min: float     # durée typique minutes
    distance_km: float      # distance typique km (0 si non applicable)
    elevation_m: float      # dénivelé+ pour trail
    recovery_cost: float    # coût récupération 0-1
    min_level: int          # 0=tous, 1=intermédiaire+, 2=avancé+
    description: str
    example_debutant: str   # allure/contenu pour débutant
    example_intermediaire: str
    example_avance: str


SESSION_CATALOGUE = [
    # ── REPOS ──
    SessionType(0, "Repos complet", "repos", 0.0, 0, 0, 0, 0.0, 0,
        "Journée sans activité physique intense — récupération maximale.",
        "Marche légère si envie, pas de course", "Repos total", "Repos actif léger"),

    SessionType(1, "Récupération passive", "repos", 0.05, 20, 0, 0, 0.0, 0,
        "Étirements, yoga, mobilité — aucun effort cardiovasculaire.",
        "20min yoga ou étirements doux", "20min mobilité ciblée", "20min yoga + 10min foam roller"),

    # ── RÉCUPÉRATION ACTIVE ──
    SessionType(2, "Footing de récupération", "recuperation", 0.2, 30, 5, 0, 0.1, 0,
        "Course très lente, on peut tenir une conversation sans s'essouffler.",
        "30min à 8:00-9:00/km", "30min à 6:30-7:30/km", "30min à 5:30-6:30/km"),

    SessionType(3, "Sortie plaisir", "recuperation", 0.25, 40, 6, 0, 0.15, 0,
        "Course sans montre ni objectif — écouter son corps, profiter du dehors.",
        "40min à l'aise, sans regarder la vitesse", "40min en nature à allure libre", "45min en terrain varié, sans pression"),

    SessionType(4, "Cross-training vélo", "recuperation", 0.25, 45, 0, 0, 0.1, 0,
        "Vélo, elliptique ou natation — entretient la forme sans impact.",
        "45min vélo ou elliptique tranquille", "45min vélo avec quelques accélérations", "50min vélo avec intervalles légers"),

    SessionType(5, "Natation récupération", "recuperation", 0.2, 40, 0, 0, 0.05, 0,
        "Natation lente — parfait pour récupérer sans impact.",
        "40min nage libre tranquille", "40min crawl à allure confort", "40min mixte crawl / dos"),

    # ── ENDURANCE DE BASE ──
    SessionType(6, "Sortie courte endurance", "endurance", 0.35, 30, 5, 0, 0.2, 0,
        "Course courte à allure facile — idéale pour les semaines chargées.",
        "30min à 8:00/km", "30min à 6:00/km", "30min à 5:00/km"),

    SessionType(7, "Endurance fondamentale", "endurance", 0.45, 60, 10, 0, 0.3, 0,
        "Course à allure modérée Z2 — la base de tout plan d'entraînement.",
        "60min à 7:30/km", "60min à 5:30/km", "60min à 4:45/km"),

    SessionType(8, "Sortie longue débutant", "endurance", 0.35, 60, 8, 0, 0.35, 0,
        "Sortie longue adaptée aux débutants — marche/course si besoin.",
        "60min marche/course alternée", "60min course continue à 7:00/km", "—"),

    SessionType(9, "Sortie longue Z2", "endurance", 0.4, 90, 15, 0, 0.4, 1,
        "Course longue en zone aérobie — construit l'endurance de base.",
        "90min à 7:00/km", "90min à 5:30/km", "90min à 4:45/km"),

    SessionType(10, "Sortie longue progressive", "endurance", 0.55, 90, 16, 0, 0.5, 1,
        "Commence lentement et finit plus vite — travail de la résistance à la fatigue.",
        "90min : 60min facile + 30min un peu plus vite", "90min avec 30min à allure semi", "90min avec 30min à allure marathon"),

    SessionType(11, "Sortie très longue", "endurance", 0.4, 150, 25, 0, 0.6, 2,
        "Sortie longue pour construire le fond — réservée aux coureurs expérimentés.",
        "—", "2h30 à 5:30/km", "2h30 à 4:45/km"),

    # ── TRAIL / NATURE ──
    SessionType(12, "Trail découverte", "endurance", 0.4, 60, 10, 200, 0.35, 0,
        "Première sortie en nature — marche dans les montées, course dans les plats.",
        "60min trail avec marche dans les côtes", "60min trail à allure nature", "60min trail technique"),

    SessionType(13, "Trail technique", "specifique", 0.55, 90, 14, 400, 0.5, 1,
        "Travail de la technique en terrain varié — descentes, sentiers étroits.",
        "—", "90min trail avec dénivelé modéré", "90min technique descentes + montées"),

    SessionType(14, "Sortie longue trail", "endurance", 0.4, 150, 22, 800, 0.55, 1,
        "Sortie longue en montagne — marche/course selon le dénivelé.",
        "—", "2h30 avec 600m D+", "2h30 avec 800m D+"),

    SessionType(15, "Ultra endurance", "endurance", 0.35, 240, 40, 1500, 0.85, 2,
        "Sortie très longue pour la préparation ultra — gestion nutrition/hydratation.",
        "—", "—", "4h avec 1500m D+, ravitaillement"),

    SessionType(16, "Power hiking", "force", 0.45, 60, 8, 500, 0.4, 0,
        "Marche rapide en côte — technique ultra, préserve les jambes.",
        "60min randonnée rapide en montée", "60min power hiking avec bâtons", "60min power hiking intensif"),

    # ── ALLURE / TEMPO ──
    SessionType(17, "Tempo court", "intensite", 0.65, 40, 8, 0, 0.5, 1,
        "Course soutenue courte au seuil — version accessible du tempo.",
        "—", "40min : 10min échauffement + 20min tempo + 10min retour", "40min : 10min éch + 20min tempo rapide"),

    SessionType(18, "Tempo continu", "intensite", 0.7, 55, 10, 0, 0.6, 1,
        "Course soutenue au seuil anaérobie — peut parler mais avec difficulté.",
        "—", "55min à allure semi-marathon confort", "55min à allure 10km confort"),

    SessionType(19, "Tempo long", "intensite", 0.68, 70, 13, 0, 0.65, 2,
        "Tempo étendu — développe la résistance à l'allure de course.",
        "—", "—", "70min : 15min éch + 45min tempo + 10min retour"),

    SessionType(20, "Progression", "endurance", 0.55, 60, 11, 0, 0.45, 1,
        "Course avec augmentation progressive de l'allure — finit fort.",
        "—", "60min : commence à 6:30/km, finit à 5:00/km", "60min : commence à 5:30/km, finit à 4:15/km"),

    # ── FRACTIONNÉ ──
    SessionType(21, "Fractionné court débutant", "intensite", 0.7, 40, 6, 0, 0.5, 0,
        "Introduction au fractionné — courtes accélérations suivies de récupération.",
        "40min : 8x200m rapide avec 2min marche entre", "40min : 8x300m avec 90s récup trot", "—"),

    SessionType(22, "Fractionné court", "intensite", 0.85, 50, 9, 0, 0.65, 1,
        "Répétitions courtes à haute intensité — développe la vitesse.",
        "—", "50min : 10x400m avec 90s récup", "50min : 12x400m avec 75s récup"),

    SessionType(23, "Fractionné long", "intensite", 0.78, 60, 11, 0, 0.65, 1,
        "Répétitions longues au seuil — développe l'endurance à allure vive.",
        "—", "60min : 5x1000m avec 2min récup", "60min : 6x1000m avec 90s récup"),

    SessionType(24, "VO2max", "intensite", 0.95, 50, 9, 0, 0.8, 2,
        "Intervalles à intensité maximale aérobie — pour coureurs confirmés seulement.",
        "—", "—", "50min : 6x3min à 95% FCmax avec 3min récup"),

    SessionType(25, "30-30", "intensite", 0.85, 40, 7, 0, 0.6, 1,
        "30 secondes vite / 30 secondes lent — excellente séance VO2max accessible.",
        "—", "40min : 20x(30s rapide + 30s trot)", "40min : 30x(30s rapide + 30s trot)"),

    SessionType(26, "Pyramide", "intensite", 0.8, 55, 10, 0, 0.65, 1,
        "Fractionné en pyramide — distances croissantes puis décroissantes.",
        "—", "200-400-600-800-600-400-200m avec récup égale", "400-800-1200-800-400m avec 2min récup"),

    SessionType(27, "Côtes courtes", "force", 0.85, 45, 7, 250, 0.65, 0,
        "Répétitions en montée courte — développe la force et la puissance.",
        "45min : 8x100m de côte raide avec descente walk", "45min : 10x150m côte avec descente trot", "45min : 12x200m côte avec descente trot"),

    SessionType(28, "Côtes longues", "force", 0.78, 55, 9, 400, 0.7, 1,
        "Répétitions en montée longue — endurance musculaire en côte.",
        "—", "55min : 6x400m côte avec descente trot", "55min : 8x400m côte avec descente trot"),

    SessionType(29, "Fartlek", "intensite", 0.65, 50, 9, 0, 0.5, 0,
        "Jeu de vitesse libre — accélérations spontanées sans chrono.",
        "50min avec 6-8 accélérations de 30s-1min quand envie", "50min avec accélérations sur repères visuels", "50min fartlek structuré"),

    # ── SPÉCIFIQUE COURSE ──
    SessionType(30, "Spécifique 5km", "specifique", 0.85, 45, 9, 0, 0.6, 1,
        "Séance ciblée préparation 5km — travail à allure et au-dessus.",
        "—", "45min : 3x1600m à allure 5km cible", "45min : 4x1600m à allure 5km cible"),

    SessionType(31, "Spécifique 10km", "specifique", 0.78, 55, 11, 0, 0.6, 1,
        "Travail à allure 10km — construit la résistance à l'allure de course.",
        "—", "55min : 3x2km à allure 10km cible", "55min : 4x2km à allure 10km cible"),

    SessionType(32, "Spécifique semi-marathon", "specifique", 0.7, 75, 14, 0, 0.6, 1,
        "Allure semi-marathon en condition de course.",
        "—", "75min : 40min à allure semi cible + 35min facile", "75min : 50min à allure semi cible"),

    SessionType(33, "Spécifique marathon", "specifique", 0.65, 100, 20, 0, 0.65, 2,
        "Allure marathon sur longue distance — la séance clé de la préparation.",
        "—", "—", "100min dont 60min à allure marathon cible"),

    SessionType(34, "Allure compétition débutant", "specifique", 0.6, 45, 7, 0, 0.4, 0,
        "Courir à l'allure envisagée pour sa première course — repères concrets.",
        "45min à l'allure de ta prochaine course (sans s'épuiser)", "45min à allure cible confortable", "—"),

    # ── FORCE & RENFORCEMENT ──
    SessionType(35, "Gainage running", "force", 0.35, 30, 0, 0, 0.2, 0,
        "Renforcement spécifique running — gainage, fessiers, ischios.",
        "30min : gainage, squats, fentes, pont fessier", "30min circuit renforcement running", "30min PPG avancée"),

    SessionType(36, "Renforcement musculaire", "force", 0.4, 45, 0, 0, 0.25, 0,
        "Séance gym ou poids de corps — prévention blessures et puissance.",
        "45min full body léger", "45min circuit fonctionnel", "45min haltères + gainage"),

    SessionType(37, "Mobilité et souplesse", "recuperation", 0.1, 30, 0, 0, 0.0, 0,
        "Étirements ciblés, foam roller, travail de mobilité articulaire.",
        "30min étirements statiques doux", "30min yoga running", "30min mobilité dynamique + foam roller"),

    SessionType(38, "ABC de course", "force", 0.4, 30, 4, 0, 0.2, 0,
        "Exercices techniques de course — améliore la foulée et l'économie de course.",
        "30min : échauffement + ABC + 15min facile", "30min ABC + 15min allure tempo", "30min ABC avancés + drills"),

    # ── SEUIL LACTIQUE ──
    SessionType(39, "Seuil lactique court", "intensite", 0.72, 45, 8, 0, 0.55, 1,
        "Travail au seuil lactique — juste au-dessus de la zone de confort.",
        "—", "45min : 2x15min seuil avec 3min récup", "45min : 3x12min seuil avec 2min récup"),

    SessionType(40, "Seuil lactique long", "intensite", 0.75, 65, 12, 0, 0.65, 2,
        "Effort prolongé au seuil — développe la capacité à maintenir l'allure.",
        "—", "—", "65min : 3x15min seuil avec 2min récup"),

    # ── SPÉCIAL / SITUATIONS ──
    SessionType(41, "Pré-compétition activation", "specifique", 0.45, 30, 5, 0, 0.2, 0,
        "Sortie d'activation J-2 ou J-1 — réveille les jambes sans les fatiguer.",
        "30min très facile avec 4x20s d'accélérations légères", "30min facile + 6x20s strides", "30min facile + 8x20s strides"),

    SessionType(42, "Récupération post-compétition", "recuperation", 0.15, 25, 3, 0, 0.0, 0,
        "Footing très lent le lendemain d'une course — évacue les toxines.",
        "25min marche/trot très lent", "25min footing très facile", "25min footing récupération"),

    SessionType(43, "Séance sur piste", "intensite", 0.82, 50, 10, 0, 0.65, 1,
        "Entraînement sur piste d'athlétisme — allures précises, terrain plat.",
        "—", "50min : 8x400m avec 90s récup", "50min : 10x400m + 2x800m"),

    SessionType(44, "Course en nature", "endurance", 0.4, 60, 10, 100, 0.3, 0,
        "Sortie trail décontractée — profiter de la nature sans objectif de performance.",
        "60min en forêt à allure plaisir", "60min nature avec quelques accélérations", "60min nature avec variations terrain"),
]

N_SESSIONS = len(SESSION_CATALOGUE)  # 45


def get_session_embeddings() -> torch.Tensor:
    features = []
    for s in SESSION_CATALOGUE:
        features.append([
            s.intensity,
            s.duration_min / 240,
            s.distance_km / 40,
            s.elevation_m / 1500,
            s.recovery_cost,
            float(s.min_level == 0),  # accessible à tous
            1.0 if s.category == "endurance"   else 0.0,
            1.0 if s.category == "intensite"   else 0.0,
            1.0 if s.category == "recuperation" else 0.0,
            1.0 if s.category == "repos"        else 0.0,
        ])
    return torch.tensor(features, dtype=torch.float32)


if __name__ == "__main__":
    print(f"Catalogue CRONOS v2 : {N_SESSIONS} séances")
    cats = {}
    for s in SESSION_CATALOGUE:
        cats[s.category] = cats.get(s.category, 0) + 1
    for cat, n in sorted(cats.items()):
        print(f"  {cat:15s} : {n} séances")