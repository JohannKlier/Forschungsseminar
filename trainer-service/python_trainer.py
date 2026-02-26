"""Lightweight Python trainer service for the bike_hourly dataset."""

from typing import Dict, List
from collections.abc import Mapping, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from igann import IGANN, IGANN_interactive
from sklearn.model_selection import train_test_split

from adult_preprocessing import preprocess_adult_income
from breast_preprocessing import preprocess_breast_cancer
from bike_preprocessing import preprocess_bike_hourly
import json
from pathlib import Path


MODELS_DIR = Path(__file__).parent / "models"
SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"


class SaveModelRequest(BaseModel):
    name: str
    payload: Dict

# FastAPI app setup with permissive CORS so the frontend can call it directly.
app = FastAPI()


class TrainRequest(BaseModel):
    dataset: str
    model_type: str = "igann"
    center_shapes: bool = False
    seed: int = 3
    points: int | None = 250
    n_estimators: int = 100
    boost_rate: float = 0.1
    init_reg: float = 1
    elm_alpha: float = 1
    early_stopping: int = 50
    scale_y: bool = True


class RefitRequest(TrainRequest):
    partials: List[Dict]
    locked_features: List[str] = []
    refit_estimators: int = 50
    refit_early_stopping: int | None = None


def _to_jsonable(obj):
    """Recursively convert numpy/pandas objects to plain Python types for FastAPI responses."""
    if obj is None:
        return None
    if isinstance(obj, (float, int, str, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, Mapping):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_to_jsonable(v) for v in obj]
    return obj


def interpolate_numeric(values: List[float], xs: List[float], ys: List[float]) -> List[float]:
    """Linear interpolation helper for numeric shape functions."""
    if xs is None or ys is None:
        return [0.0 for _ in values]
    if len(xs) == 0 or len(ys) == 0 or len(xs) != len(ys):
        return [0.0 for _ in values]
    pairs = sorted(zip(list(xs), list(ys)), key=lambda p: p[0])
    sx, sy = zip(*pairs)
    result: List[float] = []
    for v in values:
        if v <= sx[0]:
            result.append(sy[0])
            continue
        if v >= sx[-1]:
            result.append(sy[-1])
            continue
        # find interval
        hi = next(i for i, x in enumerate(sx) if x >= v)
        lo = hi - 1
        t = (v - sx[lo]) / max(1e-9, sx[hi] - sx[lo])
        result.append(sy[lo] * (1 - t) + sy[hi] * t)
    return result


def evaluate_contribs(
    shape_fn: Dict,
    feat_values: List[float],
    categories: List[str] | None = None,
) -> List[float]:
    """Compute per-row contributions using shape function info."""
    if not shape_fn:
        return [0.0 for _ in feat_values]
    if shape_fn.get("datatype") == "categorical":
        cats = shape_fn.get("x", [])
        vals = shape_fn.get("y", [])
        mapping = {c: vals[i] if i < len(vals) else 0.0 for i, c in enumerate(cats)}
        contribs: List[float] = []
        for v in feat_values:
            if isinstance(v, str):
                cname = v
            else:
                idx = int(round(v))
                cname = categories[idx] if categories and idx < len(categories) else None
            contribs.append(mapping.get(cname, 0.0))
        return contribs
    # numeric
    xs = shape_fn.get("x", [])
    ys = shape_fn.get("y", [])
    return interpolate_numeric(feat_values, xs, ys)


def normalize_numeric_shape_points(shape_functions: Dict, feature_keys: List[str], cat_info: Dict, num_points: int) -> Dict:
    """
    Ensure numeric shape functions use exactly ``num_points`` knots for stable frontend behavior.
    Categorical shape functions are kept unchanged.
    """
    normalized: Dict = {}
    for key in feature_keys:
        shape_fn = shape_functions.get(key, {})
        if not shape_fn:
            normalized[key] = shape_fn
            continue
        if key in cat_info or shape_fn.get("datatype") == "categorical":
            normalized[key] = shape_fn
            continue

        xs_raw = [float(v) for v in (shape_fn.get("x") or [])]
        ys_raw = [float(v) for v in (shape_fn.get("y") or [])]
        pair_count = min(len(xs_raw), len(ys_raw))
        if pair_count == 0:
            normalized[key] = shape_fn
            continue

        pairs = sorted(zip(xs_raw[:pair_count], ys_raw[:pair_count]), key=lambda p: p[0])
        xs_sorted = [p[0] for p in pairs]
        ys_sorted = [p[1] for p in pairs]
        min_x, max_x = xs_sorted[0], xs_sorted[-1]

        if num_points <= 1:
            target_x = [min_x]
        else:
            target_x = np.linspace(min_x, max_x, num_points).tolist()
        target_y = interpolate_numeric(target_x, xs_sorted, ys_sorted)

        normalized[key] = {
            **shape_fn,
            "datatype": "numerical",
            "x": target_x,
            "y": target_y,
        }
    return normalized


@app.post("/train")
def train(request: TrainRequest):
    # Entrypoint used by the frontend for training and returning editable partials.
    response = build_train_response(request)
    return _to_jsonable(response)


def _apply_edited_partials_to_interactive_model(igann_model, edited_partials: List[Dict], cat_info: Dict):
    """Update IGANN_interactive GAM feature_dict from frontend-edited partials."""
    if getattr(igann_model, "GAM", None) is None:
        raise HTTPException(status_code=400, detail="Refit requires an interactive model with GAM wrapper.")
    if not edited_partials:
        return

    updates = {}
    for partial in edited_partials:
        key = partial.get("key")
        if not key:
            continue

        if key in cat_info:
            categories = [str(v) for v in (partial.get("categories") or cat_info.get(key) or [])]
            y_vals = [float(v) for v in (partial.get("editableY") or [])]
            if categories and len(y_vals) != len(categories):
                # Keep alignment deterministic for partially malformed payloads.
                y_vals = (y_vals + [0.0] * len(categories))[: len(categories)]
            updates[key] = {
                "datatype": "categorical",
                "x": categories,
                "y": y_vals,
            }
        else:
            x_vals = [float(v) for v in (partial.get("editableX") or [])]
            y_vals = [float(v) for v in (partial.get("editableY") or [])]
            if not x_vals or not y_vals:
                continue
            pair_count = min(len(x_vals), len(y_vals))
            pairs = sorted(zip(x_vals[:pair_count], y_vals[:pair_count]), key=lambda p: p[0])
            updates[key] = {
                "datatype": "numerical",
                "x": [p[0] for p in pairs],
                "y": [p[1] for p in pairs],
            }

    if updates:
        igann_model.GAM.update_feature_dict(updates)


def build_train_response(
    request: TrainRequest,
    edited_partials: List[Dict] | None = None,
    locked_features: List[str] | None = None,
    refit_estimators: int = 0,
    refit_early_stopping: int | None = None,
):
    # Validate the dataset early to keep error responses simple and explicit.
    if request.dataset not in {"bike_hourly", "adult_income", "breast_cancer"}:
        raise HTTPException(status_code=400, detail="Only bike_hourly, adult_income, and breast_cancer are supported.")
    model_type = request.model_type if request.model_type in {"igann", "igann_interactive"} else "igann"
    center_shapes = bool(getattr(request, "center_shapes", False))
    # Clamp point count and hyperparameters to keep UI responsiveness predictable.
    num_points = max(2, min(250, request.points or 250))
    n_estimators = max(10, min(500, request.n_estimators))
    boost_rate = max(0.01, min(1.0, request.boost_rate))
    init_reg = max(0.01, min(10.0, request.init_reg))
    elm_alpha = max(0.01, min(10.0, request.elm_alpha))
    early_stopping = max(5, min(200, request.early_stopping))
    task_type = "classification" if request.dataset in {"adult_income", "breast_cancer"} else "regression"
    igann_task = "regression" if request.dataset in {"adult_income", "breast_cancer"} else task_type
    # Dataset-specific preprocessing returns: features, target, categorical metadata, labels.
    if request.dataset == "adult_income":
        X_proc, y_full, cat_info, labels = preprocess_adult_income(request.seed)
    elif request.dataset == "breast_cancer":
        X_proc, y_full, cat_info, labels = preprocess_breast_cancer()
    else:
        X_proc, y_full, cat_info, labels = preprocess_bike_hourly(request.seed)
    # Train/test split for metrics and visual validation.
    feature_keys = list(X_proc.columns)
    X_train_df, X_test_df, y_train_arr, y_test_arr = train_test_split(
        X_proc, y_full, test_size=0.2, random_state=request.seed
    )
    y_train = np.array(y_train_arr).astype(float).flatten()
    y_test = np.array(y_test_arr).astype(float).flatten()
    use_scale_y = bool(request.scale_y) if task_type == "regression" else False

    # Train either base IGANN or IGANN_interactive based on frontend selection.
    model_cls = IGANN_interactive if model_type == "igann_interactive" else IGANN
    igann = model_cls(
        task=igann_task,
        n_estimators=n_estimators,
        boost_rate=boost_rate,
        init_reg=init_reg,
        elm_alpha=elm_alpha,
        early_stopping=early_stopping,
        device="cpu",
        random_state=request.seed,
        verbose=0,
        scale_y=use_scale_y,
        **({"GAMwrapper": True, "GAM_detail": num_points} if model_type == "igann_interactive" else {}),
    )
    igann.fit(X_train_df, y_train)

    if edited_partials:
        if model_type != "igann_interactive":
            raise HTTPException(
                status_code=400,
                detail="Refit from edited shape functions currently requires model_type='igann_interactive'.",
            )
        _apply_edited_partials_to_interactive_model(igann, edited_partials, cat_info)
        if locked_features:
            igann.locked_feature_names = [str(f) for f in locked_features]

        can_continue = hasattr(igann, "continue_fit") and refit_estimators > 0
        if can_continue:
            igann.n_estimators = max(1, min(500, int(refit_estimators)))
            if refit_early_stopping is not None:
                igann.early_stopping = max(1, min(200, int(refit_early_stopping)))
            igann.continue_fit(X_train_df, y_train)

    if center_shapes:
        if model_type != "igann_interactive" or not hasattr(igann, "center_shape_functions"):
            raise HTTPException(
                status_code=400,
                detail="Centering shape functions currently requires model_type='igann_interactive'.",
            )
        igann.center_shape_functions(X_train_df, update_intercept=True)

    def calc_metrics(y_true, y_pred):
        # Minimal metrics for quick UI summaries.
        if len(y_true) == 0:
            return {"rmse": None, "r2": None, "acc": None, "count": 0}
        if task_type == "classification":
            y_bin = (y_true >= 0.5).astype(float)
            p_bin = (y_pred >= 0.5).astype(float)
            acc_val = float(np.mean(y_bin == p_bin))
            return {"acc": acc_val, "count": int(len(y_true))}
        rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mean_y = float(np.mean(y_true))
        r2_den = float(np.sum((y_true - mean_y) ** 2))
        r2_val = float(1 - np.sum((y_true - y_pred) ** 2) / r2_den) if r2_den != 0 else 0.0
        return {"rmse": rmse_val, "r2": r2_val, "count": int(len(y_true))}

    # Map feature keys to human-readable labels for display.
    label_map = labels
    features_train = {k: X_train_df[k].tolist() for k in feature_keys}
    features_test = {k: X_test_df[k].tolist() for k in feature_keys} if len(X_test_df) else {}
    # Normalize categorical values to strings to match IGANN shape keys
    for cat_key in cat_info.keys():
        if cat_key in features_train:
            features_train[cat_key] = [str(v) for v in features_train[cat_key]]
        if cat_key in features_test:
            features_test[cat_key] = [str(v) for v in features_test[cat_key]]
    test_len = len(X_test_df)

    # Shape functions come from the interactive GAM wrapper or from IGANN directly.
    shape_functions = (
        igann.GAM.get_feature_dict()
        if getattr(igann, "GAM", None) is not None
        else igann.get_shape_functions_as_dict()
    )
    if not shape_functions:
        raise HTTPException(status_code=500, detail="Model did not produce shape functions.")
    shape_functions = normalize_numeric_shape_points(shape_functions, feature_keys, cat_info, num_points)

    def get_shape(key: str) -> Dict:
        return shape_functions.get(key, {})

    # Build per-row contributions for each feature and aggregate totals.
    contribs_train: List[np.ndarray] = []
    contribs_test: List[np.ndarray] = []
    for key in feature_keys:
        shape_fn = get_shape(key)
        contribs = np.array(evaluate_contribs(shape_fn, features_train[key], cat_info.get(key)))
        contribs_train.append(contribs)
        if test_len:
            contribs_t = np.array(evaluate_contribs(shape_fn, features_test[key], cat_info.get(key)))
            contribs_test.append(contribs_t)

    total_train = np.sum(np.stack(contribs_train, axis=0), axis=0) if contribs_train else np.zeros_like(y_train)
    total_test = np.sum(np.stack(contribs_test, axis=0), axis=0) if contribs_test else np.zeros_like(y_test)
    if task_type == "classification":
        # Calibrate a global intercept so predicted probabilities match the target mean.
        target_mean = float(np.mean(y_train)) if len(y_train) else 0.0
        target_mean = min(max(target_mean, 1e-4), 1 - 1e-4)
        low, high = -12.0, 12.0
        for _ in range(40):
            mid = (low + high) / 2
            probs = 1 / (1 + np.exp(-(total_train + mid)))
            if float(np.mean(probs)) < target_mean:
                low = mid
            else:
                high = mid
        intercept_val = (low + high) / 2
        preds_train = 1 / (1 + np.exp(-(total_train + intercept_val)))
        preds_test = 1 / (1 + np.exp(-(total_test + intercept_val))) if len(total_test) else np.array([])
    else:
        intercept_val = float(np.mean(y_train - total_train)) if len(y_train) else 0.0
        preds_train = total_train + intercept_val
        preds_test = total_test + intercept_val if len(total_test) else np.array([])

    train_metrics = calc_metrics(y_train, preds_train)
    test_metrics = calc_metrics(y_test, preds_test)

    y_display = y_train.tolist()

    # Build partials: each feature gets editable x/y plus scatter data for UI rendering.
    partials = []
    for term_idx, key in enumerate(feature_keys):
        shape_fn = get_shape(key)
        if key in cat_info:
            # Categorical features: use category indices as editable positions.
            categories = cat_info[key]
            mapping = {c: 0.0 for c in categories}
            x_vals = shape_fn.get("x", [])
            y_vals = shape_fn.get("y", [])
            for i, cat in enumerate(x_vals):
                if cat in mapping:
                    mapping[cat] = y_vals[i] if i < len(y_vals) else 0.0
            contribs = [mapping.get(cat, 0.0) for cat in categories]
            partials.append(
                {
                    "key": key,
                    "label": label_map.get(key, key),
                    "categories": categories,
                    "scatterX": features_train[key],
                    "trueSignal": None,
                    "editableX": list(range(len(categories))),
                    "editableY": contribs,
                }
            )
        else:
            # Continuous features: use wrapper-provided points directly.
            discrete = {"x": list(shape_fn.get("x", [])), "y": list(shape_fn.get("y", []))}
            partials.append(
                {
                    "key": key,
                    "label": label_map.get(key, key),
                    "gridX": [],
                    "curve": [],
                    "scatterX": features_train[key],
                    "trueSignal": None,
                    "editableX": discrete["x"],
                    "editableY": discrete["y"],
                }
            )

    return {
        "dataset": request.dataset,
        "model_type": model_type,
        "center_shapes": center_shapes,
        "seed": request.seed,
        "n_estimators": n_estimators,
        "boost_rate": boost_rate,
        "init_reg": init_reg,
        "elm_alpha": elm_alpha,
        "early_stopping": early_stopping,
        "scale_y": use_scale_y,
        "points": num_points,
        "intercept": intercept_val,
        "partials": partials,
        "predictions": preds_train.tolist(),
        "y": y_display,
        "task": task_type,
        "source": model_type,
        "trainMetrics": train_metrics,
        "testMetrics": test_metrics,
        "testPreds": preds_test.tolist(),
        "testY": y_test.tolist(),
        "point_counts": {key: len((shape_functions.get(key) or {}).get("x", [])) for key in feature_keys},
    }


@app.post("/refit")
def refit(request: RefitRequest):
    """Refit from frontend-edited shape functions and return refreshed partials/predictions."""
    response = build_train_response(
        request,
        edited_partials=request.partials,
        locked_features=request.locked_features,
        refit_estimators=request.refit_estimators,
        refit_early_stopping=request.refit_early_stopping,
    )
    response["refit_from_edits"] = True
    response["locked_features"] = [str(f) for f in request.locked_features]
    response["center_shapes"] = bool(request.center_shapes)
    return _to_jsonable(response)


@app.get("/models")
def list_models():
    # Return names without extensions for display and selection.
    if not MODELS_DIR.exists():
        return {"models": []}
    names = sorted([p.stem for p in MODELS_DIR.glob("*.json") if p.is_file()])
    return {"models": names}


@app.get("/models/{name}")
def get_model(name: str):
    # Resolve a single stored model by name.
    safe_name = Path(name).name
    if not safe_name.endswith(".json"):
        safe_name = f"{safe_name}.json"
    path = MODELS_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found.")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/saved-models")
def list_saved_models():
    # User-edited models saved from the UI.
    if not SAVED_MODELS_DIR.exists():
        return {"models": []}
    names = sorted([p.stem for p in SAVED_MODELS_DIR.glob("*.json") if p.is_file()])
    return {"models": names}


@app.get("/saved-models/{name}")
def get_saved_model(name: str):
    # Resolve a single saved edit by name.
    safe_name = Path(name).name
    if not safe_name.endswith(".json"):
        safe_name = f"{safe_name}.json"
    path = SAVED_MODELS_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found.")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/saved-models")
def save_model(request: SaveModelRequest):
    # Persist edited model payloads on disk.
    safe_name = Path(request.name).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Missing model name.")
    if not safe_name.endswith(".json"):
        safe_name = f"{safe_name}.json"
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVED_MODELS_DIR / safe_name
    with path.open("w", encoding="utf-8") as f:
        json.dump(request.payload, f, indent=2)
    return {"saved": safe_name}


@app.get("/healthz")
def healthz():
    # Lightweight liveness check.
    return {"status": "ok"}
