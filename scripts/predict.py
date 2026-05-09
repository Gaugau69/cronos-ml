"""
scripts/predict.py — Génère les recommandations de séances pour un athlète.

Usage :
    python -m scripts.predict --name Laurent --db_url postgresql://...
    python -m scripts.predict --name Emmanuel --top_k 5
"""

import argparse
import json
import os
from datetime import date, timedelta

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor


# ─────────────────────────────────────────────────────────────
# Catalogue des séances (copie de session_types.py)
# ─────────────────────────────────────────────────────────────

SESSIONS = [
    {"id": 0,  "name": "Récupération active",    "category": "recuperation", "intensity": 0.2,  "duration_min": 40,  "distance_km": 6,  "recovery_cost": 0.1,  "description": "Footing très lent Z1", "example": "40min à 6:30/km"},
    {"id": 1,  "name": "Sortie longue Z2",        "category": "endurance",    "intensity": 0.4,  "duration_min": 120, "distance_km": 20, "recovery_cost": 0.4,  "description": "Course longue en zone aérobie", "example": "2h à 5:30/km"},
    {"id": 2,  "name": "Sortie longue trail",     "category": "endurance",    "intensity": 0.4,  "duration_min": 150, "distance_km": 25, "recovery_cost": 0.5,  "description": "Sortie longue avec dénivelé", "example": "2h30 avec 800m D+"},
    {"id": 3,  "name": "Tempo continu",           "category": "intensite",    "intensity": 0.7,  "duration_min": 50,  "distance_km": 10, "recovery_cost": 0.6,  "description": "Course au seuil anaérobie", "example": "50min à 4:20/km"},
    {"id": 4,  "name": "Fractionné court",        "category": "intensite",    "intensity": 0.9,  "duration_min": 55,  "distance_km": 10, "recovery_cost": 0.7,  "description": "Répétitions courtes haute intensité", "example": "10x400m avec récup 90s"},
    {"id": 5,  "name": "Fractionné long",         "category": "intensite",    "intensity": 0.8,  "duration_min": 60,  "distance_km": 12, "recovery_cost": 0.7,  "description": "Répétitions longues au seuil", "example": "5x1000m avec récup 2min"},
    {"id": 6,  "name": "Côtes",                   "category": "force",        "intensity": 0.85, "duration_min": 50,  "distance_km": 8,  "recovery_cost": 0.7,  "description": "Répétitions en montée", "example": "10x200m de côte"},
    {"id": 7,  "name": "Fartlek",                 "category": "intensite",    "intensity": 0.7,  "duration_min": 55,  "distance_km": 10, "recovery_cost": 0.5,  "description": "Variations libres d'allure", "example": "55min avec accélérations"},
    {"id": 8,  "name": "Progression",             "category": "endurance",    "intensity": 0.6,  "duration_min": 60,  "distance_km": 12, "recovery_cost": 0.5,  "description": "Allure progressive", "example": "60min en finissant vite"},
    {"id": 9,  "name": "Endurance fondamentale",  "category": "endurance",    "intensity": 0.5,  "duration_min": 75,  "distance_km": 13, "recovery_cost": 0.3,  "description": "Course à allure modérée Z2/Z3", "example": "75min à 5:00/km"},
    {"id": 10, "name": "Spécifique marathon",     "category": "specifique",   "intensity": 0.7,  "duration_min": 90,  "distance_km": 18, "recovery_cost": 0.6,  "description": "Allure marathon", "example": "90min à allure objectif"},
    {"id": 11, "name": "Spécifique semi",         "category": "specifique",   "intensity": 0.75, "duration_min": 70,  "distance_km": 14, "recovery_cost": 0.6,  "description": "Allure semi-marathon", "example": "70min à allure objectif"},
    {"id": 12, "name": "Spécifique 10km",         "category": "specifique",   "intensity": 0.8,  "duration_min": 55,  "distance_km": 11, "recovery_cost": 0.6,  "description": "Allure 10km", "example": "55min à allure objectif"},
    {"id": 13, "name": "Trail technique",         "category": "specifique",   "intensity": 0.6,  "duration_min": 90,  "distance_km": 15, "recovery_cost": 0.5,  "description": "Terrain varié technique", "example": "90min sur sentiers"},
    {"id": 14, "name": "Ultra endurance",         "category": "endurance",    "intensity": 0.4,  "duration_min": 240, "distance_km": 40, "recovery_cost": 0.8,  "description": "Sortie très longue ultra", "example": "4h avec 1500m D+"},
    {"id": 15, "name": "Gainage et renforcement", "category": "force",        "intensity": 0.4,  "duration_min": 45,  "distance_km": 0,  "recovery_cost": 0.2,  "description": "Renforcement musculaire", "example": "45min PPG/gainage"},
    {"id": 16, "name": "Mobilité et récupération","category": "recuperation", "intensity": 0.1,  "duration_min": 30,  "distance_km": 0,  "recovery_cost": 0.0,  "description": "Stretching, yoga", "example": "30min yoga/stretching"},
    {"id": 17, "name": "Seuil lactique",          "category": "intensite",    "intensity": 0.75, "duration_min": 55,  "distance_km": 11, "recovery_cost": 0.65, "description": "Course au seuil lactique", "example": "3x10min seuil"},
    {"id": 18, "name": "VO2max",                  "category": "intensite",    "intensity": 0.95, "duration_min": 50,  "distance_km": 10, "recovery_cost": 0.8,  "description": "Intervalles à VO2max", "example": "6x3min à 95% FCmax"},
    {"id": 19, "name": "Cross-training",          "category": "recuperation", "intensity": 0.2,  "duration_min": 45,  "distance_km": 0,  "recovery_cost": 0.1,  "description": "Natation, vélo, elliptique", "example": "45min vélo ou natation"},
]


# ─────────────────────────────────────────────────────────────
# Récupération des données depuis Supabase
# ─────────────────────────────────────────────────────────────

def fetch_metrics(conn, name: str, days: int = 15) -> list[dict]:
    since = (date.today() - timedelta(days=days)).isoformat()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT dm.*
            FROM daily_metrics dm
            JOIN users u ON u.id = dm.user_id
            WHERE u.name = %s AND dm.date >= %s
            ORDER BY dm.date DESC
        """, (name, since))
        return [dict(r) for r in cur.fetchall()]


def fetch_user(conn, name: str) -> dict | None:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM users WHERE name = %s", (name,))
        row = cur.fetchone()
        return dict(row) if row else None


# ─────────────────────────────────────────────────────────────
# Calcul du score de récupération
# ─────────────────────────────────────────────────────────────

def compute_recovery(metrics: list[dict]) -> tuple[float, dict]:
    if not metrics:
        return 0.5, {}

    latest = metrics[0]
    hrv_values = [float(m["hrv_last_night"]) for m in metrics if m.get("hrv_last_night")]
    hrv_mean   = float(np.median(hrv_values)) if hrv_values else None
    hrv_today  = float(latest["hrv_last_night"]) if latest.get("hrv_last_night") else None
    sleep_today= float(latest["sleep_score"])    if latest.get("sleep_score")    else None
    bb_today   = float(latest["body_battery_charged"]) if latest.get("body_battery_charged") else None

    recovery = 0.5
    n_signals = 0

    if hrv_today and hrv_mean and hrv_mean > 0:
        hrv_score = min(hrv_today / hrv_mean, 1.5) / 1.5
        recovery += (hrv_score - 0.5) * 0.5
        n_signals += 1

    if sleep_today:
        recovery += (sleep_today / 100 - 0.5) * 0.3
        n_signals += 1

    if bb_today:
        recovery += (bb_today / 100 - 0.5) * 0.2
        n_signals += 1

    recovery = max(0.05, min(0.95, recovery))

    return recovery, {
        "hrv_today":    round(hrv_today, 1) if hrv_today else None,
        "hrv_mean":     round(hrv_mean, 1)  if hrv_mean  else None,
        "sleep_score":  int(sleep_today)    if sleep_today else None,
        "body_battery": int(bb_today)       if bb_today  else None,
        "n_signals":    n_signals,
    }


# ─────────────────────────────────────────────────────────────
# Ranking des séances
# ─────────────────────────────────────────────────────────────

def rank_sessions(recovery: float, top_k: int = 5) -> list[dict]:
    scored = []
    for s in SESSIONS:
        intensity = s["intensity"]
        cost = s["recovery_cost"]

        if recovery >= 0.7:
            score = 1.0 - abs(intensity - 0.75) * 0.6
        elif recovery >= 0.5:
            score = 1.0 - abs(intensity - 0.5) * 0.8
        elif recovery >= 0.35:
            score = 1.0 - abs(intensity - 0.3) * 1.0
        else:
            score = 1.0 - abs(intensity - 0.15) * 1.2

        if recovery < 0.4 and cost > 0.6:
            score *= 0.4
        if recovery < 0.4 and s["category"] == "recuperation":
            score = min(0.95, score * 1.3)

        scored.append((max(0.05, min(0.95, score)), s))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "rank":         i + 1,
            "name":         s["name"],
            "score":        round(sc * 100, 1),
            "category":     s["category"],
            "description":  s["description"],
            "example":      s["example"],
            "duration_min": s["duration_min"],
            "distance_km":  s["distance_km"],
        }
        for i, (sc, s) in enumerate(scored[:top_k])
    ]


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def predict(name: str, top_k: int = 5, database_url: str = None, output: str = None):
    db_url = database_url or os.environ.get("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")

    if not db_url:
        print("❌ DATABASE_URL non défini")
        return

    conn = psycopg2.connect(db_url)

    user = fetch_user(conn, name)
    if not user:
        print(f"❌ Utilisateur '{name}' introuvable")
        return

    metrics = fetch_metrics(conn, name, days=15)
    conn.close()

    recovery, signals = compute_recovery(metrics)
    recommendations   = rank_sessions(recovery, top_k)

    result = {
        "user":  name,
        "date":  date.today().isoformat(),
        "recovery": {
            "score": round(recovery * 100, 1),
            "level": (
                "Excellente" if recovery >= 0.75 else
                "Bonne"      if recovery >= 0.55 else
                "Moyenne"    if recovery >= 0.4  else
                "Faible"
            ),
            **signals,
        },
        "recommendations": recommendations,
    }

    # Affichage
    print(f"\n{'='*55}")
    print(f"  CRONOS — Recommandations pour {name}")
    print(f"  {date.today().isoformat()}")
    print(f"{'='*55}")
    print(f"\n  Récupération : {result['recovery']['score']}/100 ({result['recovery']['level']})")
    if signals.get("hrv_today"):
        print(f"  HRV          : {signals['hrv_today']} ms (moy. {signals['hrv_mean']} ms)")
    if signals.get("sleep_score"):
        print(f"  Sommeil      : {signals['sleep_score']}/100")
    if signals.get("body_battery"):
        print(f"  Body Battery : {signals['body_battery']}/100")
    print(f"  Signaux      : {signals.get('n_signals', 0)}/3")

    print(f"\n  Top {top_k} séances recommandées :")
    print(f"  {'─'*50}")
    for r in recommendations:
        bar = "█" * int(r["score"] / 10) + "░" * (10 - int(r["score"] / 10))
        print(f"\n  #{r['rank']} {r['name']} ({r['score']}/100)")
        print(f"     {bar}")
        print(f"     {r['category'].upper()} | {r['duration_min']}min | {r['distance_km']}km")
        print(f"     {r['example']}")

    print(f"\n{'='*55}\n")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  → Sauvegardé dans {output}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRONOS — Recommandation de séances")
    parser.add_argument("--name",    required=True, help="Nom de l'athlète")
    parser.add_argument("--top_k",  type=int, default=5, help="Nombre de séances")
    parser.add_argument("--db_url", type=str, default=None, help="DATABASE_URL PostgreSQL")
    parser.add_argument("--output", type=str, default=None, help="Fichier JSON de sortie")
    args = parser.parse_args()

    predict(
        name=args.name,
        top_k=args.top_k,
        database_url=args.db_url,
        output=args.output,
    )
