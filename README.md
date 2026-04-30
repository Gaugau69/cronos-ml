# CRONOS — ML

Pipeline de feature engineering et modèle JEPA pour la prédiction de récupération athlétique J+1.

## Structure

```
cronos-ml/
├── data/                  ← ignoré par git (CSV + tenseurs)
│   ├── raw/               ← CSV exportés depuis Supabase
│   └── processed/         ← X.npy, meta.csv, norm_stats.json
├── features/              ← pipeline feature engineering
├── models/                ← architecture JEPA
├── training/              ← boucle d'entraînement
├── evaluation/            ← métriques
└── notebooks/             ← exploration
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Feature engineering

```bash
python features/pipeline.py \
  --daily data/raw/daily_metrics.csv \
  --activities data/raw/activities.csv \
  --window 14 \
  --save data/processed/
```

## Dataset

- **Shape** : `(N, 14, 12)` — N fenêtres de 14 jours × 12 features
- **Paires JEPA** : `X_ctx` et `X_tgt` avec horizon = 1 jour
- **Features** : hr_rest, sleep_duration, stress, body_battery, hr_mean, hr_drift, pace_mean, pace_hr_ratio, duration, elevation_gain, training_load, is_rest_day

## Modèle

Architecture JEPA (Joint Embedding Predictive Architecture) :
- **Encodeur contexte** : encode une fenêtre de 14 jours
- **Encodeur target** : encode la fenêtre cible (EMA du contexte)
- **Prédicteur** : prédit la représentation latente de la cible depuis le contexte
- **Loss** : similarité cosinus dans l'espace latent

## Auteurs

Gauthier ARGENTIERI & Antoine BOUBBÉE — CentraleSupélec 2026
