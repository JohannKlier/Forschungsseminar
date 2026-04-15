from __future__ import annotations

from fastapi import FastAPI, HTTPException

from trainer_service.json_utils import to_jsonable
from trainer_service.model_store import list_model_names, load_model_payload, normalize_stored_model_payload
from trainer_service.schemas import RefitRequest, SaveModelRequest, TrainRequest
from trainer_service.storage import (
    get_saved_model_payload,
    list_saved_model_names,
    save_saved_model_payload,
)
from trainer_service.training import build_train_response


app = FastAPI()


@app.post("/train")
def train(request: TrainRequest):
    response = build_train_response(request)
    return to_jsonable(response)


@app.post("/refit")
def refit(request: RefitRequest):
    response = build_train_response(
        request,
        edited_partials=request.partials,
        locked_features=request.locked_features,
        feature_modes=request.feature_modes or None,
    )
    return to_jsonable(response)


@app.get("/models")
def list_models():
    return {"models": list_model_names()}


@app.get("/models/{name}")
def get_model(name: str):
    return load_model_payload(name)


@app.get("/saved-models")
def list_saved_models():
    try:
        return {"models": list_saved_model_names()}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/saved-models/{name}")
def get_saved_model(name: str):
    try:
        payload = get_saved_model_payload(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if payload is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    return normalize_stored_model_payload(payload)


@app.post("/saved-models")
def save_model(request: SaveModelRequest):
    try:
        safe_name = save_saved_model_payload(request.name, request.payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"saved": safe_name}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
