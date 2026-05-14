"""
scripts/retrain.py — Réentraînement automatique hebdomadaire de CRONOS.

Lance l'export Supabase → pipeline features → réentraînement JEPA → réentraînement recommender.

Usage :
    python -m scripts.retrain --db_url postgresql://...
    python -m scripts.retrain  # utilise DATABASE_URL depuis .env
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(cmd: str, description: str) -> bool:
    print(f"\n{'─'*55}")
    print(f"  {description}")
    print(f"{'─'*55}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  ❌ Échec : {cmd}")
        return False
    print(f"  ✅ OK")
    return True


def retrain(
    db_url: str = None,
    epochs_jepa: int = 300,
    epochs_rec:  int = 100,
    skip_jepa:   bool = False,
    skip_rec:    bool = False,
):
    start = datetime.now()
    print(f"\n{'='*55}")
    print(f"  CRONOS — Réentraînement automatique")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    db_url = db_url or os.environ.get("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    if not db_url:
        print("❌ DATABASE_URL non défini")
        sys.exit(1)

    python = f'"{sys.executable}"'

    # ── 1. Export Supabase ──
    ok = run(
        f'{python} -m scripts.export_training_data --output data/raw --db_url "{db_url}"',
        "1/4 Export depuis Supabase"
    )
    if not ok:
        sys.exit(1)

    # ── 2. Pipeline features ──
    ok = run(
        f'{python} -m features.pipeline --daily data/raw/daily_metrics.csv --activities data/raw/activities.csv --window 14 --save data/processed',
        "2/4 Pipeline features"
    )
    if not ok:
        sys.exit(1)

    # ── 3. Réentraînement JEPA ──
    if not skip_jepa:
        ok = run(
            f'{python} -m training.train --data data/processed --epochs {epochs_jepa} --log_every 50',
            f"3/4 Réentraînement JEPA ({epochs_jepa} epochs)"
        )
        if not ok:
            sys.exit(1)
    else:
        print("\n  ⏭ JEPA ignoré (--skip_jepa)")

    # ── 4. Réentraînement Recommender ──
    if not skip_rec:
        ok = run(
            f'{python} -m training.train_recommender --epochs {epochs_rec} --log_every 20 --profiles data/raw/profiles.json --races data/raw/races.json',
            f"4/4 Réentraînement Recommender ({epochs_rec} epochs)"
        )
        if not ok:
            sys.exit(1)
    else:
        print("\n  ⏭ Recommender ignoré (--skip_rec)")

    elapsed = (datetime.now() - start).seconds // 60
    print(f"\n{'='*55}")
    print(f"  ✅ Réentraînement terminé en {elapsed} min")
    print(f"  Modèles : checkpoints/best_model.pt + best_recommender.pt")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRONOS — Réentraînement automatique")
    parser.add_argument("--db_url",      type=str,  default=None)
    parser.add_argument("--epochs_jepa", type=int,  default=300)
    parser.add_argument("--epochs_rec",  type=int,  default=100)
    parser.add_argument("--skip_jepa",   action="store_true")
    parser.add_argument("--skip_rec",    action="store_true")
    args = parser.parse_args()

    retrain(
        db_url=args.db_url,
        epochs_jepa=args.epochs_jepa,
        epochs_rec=args.epochs_rec,
        skip_jepa=args.skip_jepa,
        skip_rec=args.skip_rec,
    )
