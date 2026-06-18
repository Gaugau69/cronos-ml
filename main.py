"""
cronos-ml — API FastAPI minimale pour déclencher le réentraînement à distance.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ML_SECRET = os.environ.get("CRONOS_ML_SECRET", "")
_retrain_lock = asyncio.Lock()
_retrain_status: dict = {"running": False, "last_run": None, "last_result": None}


async def _run_retrain():
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        log.error("[RETRAIN] DATABASE_URL non défini")
        return {"success": False, "error": "DATABASE_URL manquant"}

    from scripts.retrain import retrain
    try:
        log.info("[RETRAIN] Démarrage du réentraînement")
        retrain(
            db_url=db_url,
            epochs_jepa=int(os.environ.get("RETRAIN_EPOCHS_JEPA", "300")),
            epochs_rec=int(os.environ.get("RETRAIN_EPOCHS_REC", "100")),
        )
        log.info("[RETRAIN] Terminé avec succès")
        return {"success": True}
    except Exception as e:
        log.error(f"[RETRAIN] Erreur: {e}")
        return {"success": False, "error": str(e)}


app = FastAPI(title="cronos-ml", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "retrain_running": _retrain_status["running"]}


@app.post("/retrain")
async def trigger_retrain(x_secret: str = Header(default="")):
    if ML_SECRET and x_secret != ML_SECRET:
        raise HTTPException(status_code=401, detail="Secret invalide")

    if _retrain_status["running"]:
        return JSONResponse({"status": "already_running", "message": "Réentraînement déjà en cours"})

    async def _job():
        async with _retrain_lock:
            _retrain_status["running"] = True
            from datetime import datetime
            _retrain_status["last_run"] = datetime.utcnow().isoformat()
            result = await asyncio.to_thread(_run_retrain_sync)
            _retrain_status["last_result"] = result
            _retrain_status["running"] = False

    asyncio.create_task(_job())
    return {"status": "started", "message": "Réentraînement lancé en arrière-plan"}


def _run_retrain_sync():
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return {"success": False, "error": "DATABASE_URL manquant"}
    from scripts.retrain import retrain
    try:
        retrain(
            db_url=db_url,
            epochs_jepa=int(os.environ.get("RETRAIN_EPOCHS_JEPA", "300")),
            epochs_rec=int(os.environ.get("RETRAIN_EPOCHS_REC", "100")),
        )
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/retrain/status")
def retrain_status(x_secret: str = Header(default="")):
    if ML_SECRET and x_secret != ML_SECRET:
        raise HTTPException(status_code=401, detail="Secret invalide")
    return _retrain_status


# ── Inférence ML via HTTP ────────────────────────────────────────────────────

from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    feature_rows: list[list[float]]   # (14, 22) — 14 jours × 22 features
    profile:      Optional[dict] = None
    races:        Optional[list] = None
    user_name:    Optional[str]  = None
    top_k:        int = 5


@app.post("/predict")
def predict(payload: PredictRequest, x_secret: str = Header(default="")):
    """
    Inférence ML : prend les features sur 14 jours, retourne les top_k séances.
    Appelé par cronos-backend — les checkpoints restent sur ce service.
    """
    if ML_SECRET and x_secret != ML_SECRET:
        raise HTTPException(status_code=401, detail="Secret invalide")

    try:
        import sys, os, json
        import numpy as np
        import torch
        from pathlib import Path

        from models.encoder import Encoder
        from recommendation.recommender import SessionScorer
        from recommendation.encoders import encode_athlete_profile, encode_race_context
        from recommendation.session_types_v2 import SESSION_CATALOGUE

        ckpt_dir = Path("checkpoints")
        enc_path  = ckpt_dir / "best_model.pt"
        rec_path  = ckpt_dir / "best_recommender.pt"

        if not enc_path.exists() or not rec_path.exists():
            return {"error": "checkpoints_not_found", "recommendations": None}

        enc = Encoder(n_features=22)
        enc.load_state_dict(torch.load(enc_path, map_location="cpu", weights_only=True), strict=False)
        enc.eval()

        scorer = SessionScorer()
        scorer.load_state_dict(torch.load(rec_path, map_location="cpu", weights_only=True), strict=False)
        scorer.eval()

        x = torch.tensor([payload.feature_rows], dtype=torch.float32)  # (1, 14, 22)
        with torch.no_grad():
            z_jepa = enc(x)

        x_profile = encode_athlete_profile(payload.profile or {})
        x_race    = encode_race_context(payload.races[0] if payload.races else None)

        with torch.no_grad():
            scores = scorer(
                z_jepa.unsqueeze(0) if z_jepa.dim() == 1 else z_jepa,
                x_profile.unsqueeze(0),
                x_race.unsqueeze(0),
            ).squeeze(0).tolist()

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for s_idx, score in indexed[:payload.top_k]:
            s = SESSION_CATALOGUE[s_idx]
            results.append({
                "id": s.id, "name": s.name, "category": s.category,
                "score": round(score * 100, 1),
                "intensity": s.intensity, "duration_min": s.duration_min,
                "distance_km": s.distance_km, "recovery_cost": s.recovery_cost,
                "description": s.description,
            })

        return {"recommendations": results, "computed_with_ml": True}

    except Exception as e:
        log.error(f"[PREDICT] Erreur: {e}")
        return {"error": str(e), "recommendations": None}
