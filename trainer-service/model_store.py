from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

from fastapi import HTTPException

from paths import MODELS_DIR


def list_model_names() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted([path.stem for path in MODELS_DIR.glob("*.json") if path.is_file()])


def normalize_stored_model_payload(payload: Dict) -> Dict:
    if all(key in payload for key in ("model", "data", "version")):
        return payload

    partials = payload.get("partials") or []
    shapes = []
    train_x = {}
    categories = {}
    feature_labels = {}
    for partial in partials:
        key = str(partial.get("key", ""))
        if not key:
            continue
        label = str(partial.get("label") or key)
        partial_categories = [str(cat) for cat in (partial.get("categories") or [])]
        editable_x = partial.get("editableX")
        editable_y = partial.get("editableY")
        scatter_x = partial.get("scatterX") or []
        train_x[key] = scatter_x
        feature_labels[key] = label
        if partial_categories:
            categories[key] = partial_categories
        shapes.append(
            {
                "key": key,
                "label": label,
                "editableX": editable_x,
                "editableY": editable_y,
                "categories": partial_categories or None,
            }
        )

    model_type = payload.get("model_type") or payload.get("source") or "igann_interactive"
    task = payload.get("task") or "regression"
    points = int(payload.get("points") or 250)
    train_metrics = payload.get("trainMetrics") or {"count": len(payload.get("y") or [])}
    test_metrics = payload.get("testMetrics") or {"count": len(payload.get("testY") or [])}
    timestamp = int(time.time() * 1000)

    return {
        "model": {
            "dataset": payload.get("dataset") or "unknown",
            "model_type": model_type if model_type in {"igann", "igann_interactive"} else "igann_interactive",
            "task": task if task in {"regression", "classification"} else "regression",
            "selected_features": list(feature_labels.keys()),
            "selected_interactions": [shape.get("key") for shape in shapes if shape.get("editableZ")],
            "selected_operations": [
                {
                    "kind": "interaction",
                    "operator": "product",
                    "sources": shape.get("key", "").split("__")[:2],
                    "key": shape.get("key"),
                    "label": shape.get("label"),
                }
                for shape in shapes
                if shape.get("editableZ")
            ],
            "seed": int(payload.get("seed") or 3),
            "n_estimators": int(payload.get("n_estimators") or 100),
            "boost_rate": float(payload.get("boost_rate") or 0.1),
            "init_reg": float(payload.get("init_reg") or 1),
            "elm_alpha": float(payload.get("elm_alpha") or 1),
            "early_stopping": int(payload.get("early_stopping") or 50),
            "scale_y": bool(payload.get("scale_y", True)),
            "points": points,
        },
        "data": {
            "trainX": train_x,
            "trainY": payload.get("y") or [],
            "testY": payload.get("testY") or [],
            "categories": categories,
            "featureLabels": feature_labels,
        },
        "version": {
            "versionId": str(timestamp),
            "timestamp": timestamp,
            "source": "train",
            "center_shapes": bool(payload.get("center_shapes", False)),
            "intercept": float(payload.get("intercept") or 0.0),
            "trainMetrics": train_metrics,
            "testMetrics": test_metrics,
            "shapes": shapes,
        },
    }


def load_model_payload(name: str) -> Dict:
    safe_name = Path(name).name
    if not safe_name.endswith(".json"):
        safe_name = f"{safe_name}.json"
    path = MODELS_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found.")
    with path.open("r", encoding="utf-8") as file:
        return normalize_stored_model_payload(json.load(file))
