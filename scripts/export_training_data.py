"""
scripts/export_training_data.py — Export des données Supabase pour l'entraînement CRONOS.

Génère :
    data/rpe.json       — données RPE par activité
    data/profiles.json  — profils athlètes
    data/races.json     — courses planifiées

Usage :
    python -m scripts.export_training_data
    python -m scripts.export_training_data --output data/
"""

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor


def get_connection(database_url: str):
    """Connexion PostgreSQL via DATABASE_URL."""
    return psycopg2.connect(database_url)


def export_rpe(conn, output_dir: Path) -> int:
    """Exporte les données RPE depuis la table activities."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                a.activity_id,
                a.activity_name,
                a.activity_type,
                a.date::text AS date,
                a.duration_min,
                a.distance_km,
                a.avg_hr,
                a.max_hr,
                a.elevation_gain_m,
                a.training_effect,
                a.rpe,
                u.name AS user_name
            FROM activities a
            JOIN users u ON u.id = a.user_id
            WHERE a.rpe IS NOT NULL
            ORDER BY a.date DESC
        """)
        rows = [dict(r) for r in cur.fetchall()]

    path = output_dir / "rpe.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"  → {len(rows)} entrées RPE exportées → {path}")
    return len(rows)


def export_profiles(conn, output_dir: Path) -> int:
    """Exporte les profils athlètes depuis athlete_profiles."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Vérifie si la table existe
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'athlete_profiles'
            )
        """)
        if not cur.fetchone()["exists"]:
            print("  ⚠ Table athlete_profiles inexistante — fichier vide généré")
            path = output_dir / "profiles.json"
            with open(path, "w") as f:
                json.dump([], f)
            return 0

        cur.execute("""
            SELECT
                ap.*,
                u.name AS user_name,
                u.email AS user_email
            FROM athlete_profiles ap
            JOIN users u ON u.id = ap.user_id
            ORDER BY ap.user_id
        """)
        rows = [dict(r) for r in cur.fetchall()]

    path = output_dir / "profiles.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"  → {len(rows)} profils exportés → {path}")
    return len(rows)


def export_races(conn, output_dir: Path) -> int:
    """Exporte les courses planifiées depuis planned_races."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'planned_races'
            )
        """)
        if not cur.fetchone()["exists"]:
            print("  ⚠ Table planned_races inexistante — fichier vide généré")
            path = output_dir / "races.json"
            with open(path, "w") as f:
                json.dump([], f)
            return 0

        today = date.today().isoformat()
        cur.execute("""
            SELECT
                pr.*,
                u.name AS user_name,
                (pr.race_date - CURRENT_DATE) AS days_to_race
            FROM planned_races pr
            JOIN users u ON u.id = pr.user_id
            WHERE pr.is_completed = FALSE
              AND pr.race_date >= %s
            ORDER BY pr.race_date ASC
        """, (today,))
        rows = [dict(r) for r in cur.fetchall()]

    path = output_dir / "races.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"  → {len(rows)} courses exportées → {path}")
    return len(rows)


def export_daily_metrics(conn, output_dir: Path) -> int:
    """Exporte les métriques journalières pour analyse."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                dm.*,
                u.name AS user_name
            FROM daily_metrics dm
            JOIN users u ON u.id = dm.user_id
            ORDER BY dm.date DESC
            LIMIT 10000
        """)
        rows = [dict(r) for r in cur.fetchall()]

    path = output_dir / "daily_metrics.csv"

    # Export CSV
    if rows:
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  → {len(rows)} métriques journalières exportées → {path}")
    
    return len(rows)


def export_activities(conn, output_dir: Path) -> int:
    """Exporte toutes les activités pour le pipeline features."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                a.*,
                u.name AS user_name
            FROM activities a
            JOIN users u ON u.id = a.user_id
            ORDER BY a.date DESC
        """)
        rows = [dict(r) for r in cur.fetchall()]

    path = output_dir / "activities.csv"

    if rows:
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  → {len(rows)} activités exportées → {path}")

    return len(rows)


def main(output_dir: str = "data", database_url: str = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # DATABASE_URL depuis les variables d'environnement ou argument
    db_url = database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL non défini — ajoutez-le dans .env ou passez --db_url")
        return

    print(f"[CRONOS] Export depuis Supabase → {output_dir}/")
    print("=" * 50)

    try:
        conn = get_connection(db_url)
        
        n_rpe      = export_rpe(conn, output_dir)
        n_profiles = export_profiles(conn, output_dir)
        n_races    = export_races(conn, output_dir)
        n_daily    = export_daily_metrics(conn, output_dir)
        n_acts     = export_activities(conn, output_dir)

        conn.close()

        print("\n" + "=" * 50)
        print("✓ Export terminé !")
        print(f"  RPE        : {n_rpe} entrées")
        print(f"  Profils    : {n_profiles} athlètes")
        print(f"  Courses    : {n_races} planifiées")
        print(f"  Métriques  : {n_daily} jours")
        print(f"  Activités  : {n_acts} séances")

        if n_rpe == 0:
            print("\n⚠ Aucun RPE — le recommender utilisera des labels par défaut.")
            print("  Encourage tes athlètes à noter leurs séances sur le site !")

        if n_profiles == 0:
            print("\n⚠ Aucun profil athlète — encourage tes athlètes à remplir leur profil.")

    except Exception as e:
        print(f"❌ Erreur connexion DB : {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  type=str, default="data/raw", help="Dossier de sortie")
    parser.add_argument("--db_url",  type=str, default=None, help="DATABASE_URL PostgreSQL")
    args = parser.parse_args()

    main(output_dir=args.output, database_url=args.db_url)
