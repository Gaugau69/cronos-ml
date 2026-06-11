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
