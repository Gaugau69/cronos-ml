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


# Cache en mémoire des modèles — rechargé automatiquement si les checkpoints changent
_model_cache: dict = {"enc": None, "scorer": None, "mtime_enc": 0.0, "mtime_rec": 0.0}


def _get_models():
    """Charge ou recharge les modèles si les checkpoints ont changé depuis le dernier load."""
    import torch
    from pathlib import Path
    from models.encoder import Encoder
    from recommendation.recommender import SessionScorer

    ckpt_dir = Path("checkpoints")
    enc_path  = ckpt_dir / "best_model.pt"
    rec_path  = ckpt_dir / "best_recommender.pt"

    if not enc_path.exists() or not rec_path.exists():
        return None, None

    mtime_enc = enc_path.stat().st_mtime
    mtime_rec = rec_path.stat().st_mtime

    if (mtime_enc != _model_cache["mtime_enc"] or
            mtime_rec != _model_cache["mtime_rec"] or
            _model_cache["enc"] is None):
        log.info("[MODELS] Chargement des checkpoints (nouveaux ou premier load)")

        from models.jepa import JEPA
        from recommendation.recommender import CRONOSRecommender

        # best_model.pt → dict {"model_state": JEPA.state_dict(), ...}
        jepa = JEPA()
        ckpt_jepa = torch.load(enc_path, map_location="cpu", weights_only=False)
        jepa.load_state_dict(ckpt_jepa["model_state"], strict=True)
        jepa.eval()
        # On utilise l'encodeur de contexte pour l'inférence
        enc = jepa.context_encoder
        enc.eval()

        # best_recommender.pt → dict {"model_state": CRONOSRecommender.state_dict(), ...}
        scorer = CRONOSRecommender()
        ckpt_rec = torch.load(rec_path, map_location="cpu", weights_only=False)
        scorer.load_state_dict(ckpt_rec["model_state"], strict=True)
        scorer.eval()

        _model_cache.update({"enc": enc, "scorer": scorer, "mtime_enc": mtime_enc, "mtime_rec": mtime_rec})
        log.info("[MODELS] Checkpoints chargés en cache")

    return _model_cache["enc"], _model_cache["scorer"]


@app.post("/predict")
def predict(payload: PredictRequest, x_secret: str = Header(default="")):
    """
    Inférence ML : prend les features sur 14 jours, retourne les top_k séances.
    Appelé par cronos-backend — les checkpoints restent sur ce service.
    Modèles mis en cache, rechargés automatiquement après chaque réentraînement.
    """
    if ML_SECRET and x_secret != ML_SECRET:
        raise HTTPException(status_code=401, detail="Secret invalide")

    try:
        import torch
        from recommendation.encoders import encode_athlete_profile, encode_race_context
        from recommendation.session_types_v2 import SESSION_CATALOGUE

        enc, scorer = _get_models()
        if enc is None:
            return {"error": "checkpoints_not_found", "recommendations": None}

        x = torch.tensor([payload.feature_rows], dtype=torch.float32)  # (1, 14, 22)
        with torch.no_grad():
            z_jepa = enc(x)  # (1, 64)
            if z_jepa.dim() == 1:
                z_jepa = z_jepa.unsqueeze(0)

        x_profile = encode_athlete_profile(payload.profile or {})
        x_race    = encode_race_context(payload.races[0] if payload.races else None)

        # Assure batch dim = 1
        if x_profile.dim() == 1:
            x_profile = x_profile.unsqueeze(0)
        if x_race.dim() == 1:
            x_race = x_race.unsqueeze(0)

        with torch.no_grad():
            recs = scorer.recommend(z_jepa, x_profile, x_race, top_k=payload.top_k)

        results = []
        for rec in recs:
            results.append({
                "id":           rec["session_id"],
                "name":         rec["name"],
                "category":     rec["category"],
                "score":        rec["score"],
                "intensity":    rec["intensity"],
                "duration_min": rec["duration_min"],
                "distance_km":  rec["distance_km"],
                "recovery_cost": 0.0,
                "description":  rec["description"],
            })

        return {"recommendations": results, "computed_with_ml": True}

    except Exception as e:
        log.error(f"[PREDICT] Erreur: {e}")
        return {"error": str(e), "recommendations": None}
