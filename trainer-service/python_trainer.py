"""Lightweight Python trainer service for the bike_hourly dataset."""

from typing import Dict, List
from collections.abc import Mapping, Sequence
import inspect

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
import time
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


def _coerce_numeric_points(values) -> List[float]:
    """Normalize shape point containers without relying on ambiguous array truthiness."""
    if values is None:
        return []
    return [float(v) for v in values]


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

        xs_raw = _coerce_numeric_points(shape_fn.get("x"))
        ys_raw = _coerce_numeric_points(shape_fn.get("y"))
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


def build_feature_dict_from_partials(edited_partials: List[Dict], cat_info: Dict) -> Dict:
    """Convert frontend partials into the IGANN interactive feature_dict format."""
    if not edited_partials:
        return {}
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
    return updates


def merge_learned_shapes_preserve_base_grid(base_shapes: Dict, learned_shapes: Dict, feature_keys: List[str]) -> Dict:
    """
    Keep the edited base grid/category ordering and project learned shapes onto it.
    This preserves frontend detail while still reflecting refit updates.
    """
    merged: Dict = {**base_shapes}
    for key in feature_keys:
        learned = learned_shapes.get(key)
        if not learned:
            continue
        base = base_shapes.get(key)
        if not base:
            merged[key] = learned
            continue

        if learned.get("datatype") == "categorical" or base.get("datatype") == "categorical":
            base_x_values = base.get("x")
            learned_x_values = learned.get("x")
            learned_y_values = learned.get("y")
            base_cats = [str(v) for v in ([] if base_x_values is None else base_x_values)]
            learned_x = [str(v) for v in ([] if learned_x_values is None else learned_x_values)]
            learned_y = [float(v) for v in ([] if learned_y_values is None else learned_y_values)]
            learned_map = {cat: learned_y[i] for i, cat in enumerate(learned_x) if i < len(learned_y)}
            base_y_values = base.get("y")
            base_y = [float(v) for v in ([] if base_y_values is None else base_y_values)]
            merged[key] = {
                **base,
                "datatype": "categorical",
                "x": base_cats,
                "y": [learned_map.get(cat, base_y[i] if i < len(base_y) else 0.0) for i, cat in enumerate(base_cats)],
            }
            continue

        base_x = _coerce_numeric_points(base.get("x"))
        if len(base_x) == 0:
            merged[key] = learned
            continue
        learned_x = _coerce_numeric_points(learned.get("x"))
        learned_y = _coerce_numeric_points(learned.get("y"))
        projected = interpolate_numeric(base_x, learned_x, learned_y)
        merged[key] = {
            **base,
            "datatype": "numerical",
            "x": base_x,
            "y": projected,
        }
    return merged


def center_shape_functions_for_data(
    shape_functions: Dict,
    feature_keys: List[str],
    features_train: Dict[str, List],
    cat_info: Dict,
) -> tuple[Dict, float]:
    """Center each feature contribution on empirical training data and return intercept shift."""
    centered = {k: {**v} for k, v in shape_functions.items()}
    intercept_shift = 0.0
    for key in feature_keys:
        shape_fn = centered.get(key)
        if not shape_fn:
            continue
        contribs = evaluate_contribs(shape_fn, features_train.get(key, []), cat_info.get(key))
        if not contribs:
            continue
        mu = float(np.mean(np.asarray(contribs, dtype=float)))
        ys_raw = shape_fn.get("y")
        ys = _coerce_numeric_points(ys_raw)
        if len(ys) == 0:
            continue
        shape_fn["y"] = [y - mu for y in ys]
        intercept_shift += mu
    return centered, intercept_shift

def build_train_response(
    request: TrainRequest,
    edited_partials: List[Dict] | None = None,
    locked_features: List[str] | None = None,
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
    model_kwargs = dict(
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
    )
    
    
    
    if model_type == "igann_interactive":
        model_kwargs["GAMwrapper"] = True
        model_kwargs["GAM_detail"] = num_points
    igann = model_cls(**model_kwargs)

    if edited_partials:
        if model_type != "igann_interactive":
            raise HTTPException(
                status_code=400,
                detail="Refit from edited shape functions currently requires model_type='igann_interactive'.",
            )
        feature_dict = build_feature_dict_from_partials(edited_partials, cat_info)
        locked_set = {str(f) for f in (locked_features or [])}
        fit_cols = [col for col in feature_keys if str(col) not in locked_set]
        if not fit_cols:
            raise HTTPException(
                status_code=400,
                detail="All features are locked; no features left for refit.",
            )
        igann.fit_from_shape_functions(
            X_train_df[fit_cols],
            y_train,
            feature_dict,
        )
    else:
        igann.fit(X_train_df, y_train)

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
            return {"rmse": None, "mae": None, "r2": None, "acc": None, "count": 0}
        if task_type == "classification":
            y_bin = (y_true >= 0.5).astype(float)
            p_bin = (y_pred >= 0.5).astype(float)
            acc_val = float(np.mean(y_bin == p_bin))
            return {"acc": acc_val, "count": int(len(y_true))}
        rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae_val = float(np.mean(np.abs(y_true - y_pred)))
        mean_y = float(np.mean(y_true))
        r2_den = float(np.sum((y_true - mean_y) ** 2))
        r2_val = float(1 - np.sum((y_true - y_pred) ** 2) / r2_den) if r2_den != 0 else 0.0
        return {"rmse": rmse_val, "mae": mae_val, "r2": r2_val, "count": int(len(y_true))}

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

    # Shape functions:
    # - default: interactive GAM wrapper dict or base IGANN dict
    # - refit-from-edits: merge edited base dict with freshly derived shapes from current model
    if edited_partials and model_type == "igann_interactive":
        base_shapes = build_feature_dict_from_partials(edited_partials, cat_info)
        learned_shapes = igann.get_shape_functions_as_dict()
        shape_functions = merge_learned_shapes_preserve_base_grid(base_shapes, learned_shapes, feature_keys)
    else:
        shape_functions = (
            igann.get_gam_feature_dict()
            if getattr(igann, "GAM", None) is not None
            else igann.get_shape_functions_as_dict()
        )
    if not shape_functions:
        raise HTTPException(status_code=500, detail="Model did not produce shape functions.")
    shape_functions = normalize_numeric_shape_points(shape_functions, feature_keys, cat_info, num_points)
    if center_shapes and edited_partials and model_type == "igann_interactive":
        shape_functions, _ = center_shape_functions_for_data(
            shape_functions, feature_keys, features_train, cat_info
        )

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

    # Strip scatterX out of partials — raw data lives in the data block now.
    shapes = []
    for partial in partials:
        shape: Dict = {"key": partial["key"], "label": partial["label"]}
        if "categories" in partial and partial["categories"]:
            shape["categories"] = partial["categories"]
        if "editableX" in partial:
            shape["editableX"] = partial["editableX"]
        if "editableY" in partial:
            shape["editableY"] = partial["editableY"]
        shapes.append(shape)

    version_id = str(int(time.time() * 1000))
    is_refit = bool(edited_partials)

    return {
        "model": {
            "dataset": request.dataset,
            "model_type": model_type,
            "task": task_type,
            "seed": request.seed,
            "n_estimators": n_estimators,
            "boost_rate": boost_rate,
            "init_reg": init_reg,
            "elm_alpha": elm_alpha,
            "early_stopping": early_stopping,
            "scale_y": use_scale_y,
            "points": num_points,
        },
        "data": {
            "trainX": features_train,
            "trainY": y_display,
            "testY": y_test.tolist(),
            "categories": cat_info,
            "featureLabels": label_map,
        },
        "version": {
            "versionId": version_id,
            "timestamp": int(time.time() * 1000),
            "source": "refit" if is_refit else "train",
            "center_shapes": center_shapes,
            "locked_features": [str(f) for f in (locked_features or [])],
            "refit_from_edits": is_refit,
            "intercept": intercept_val,
            "trainMetrics": train_metrics,
            "testMetrics": test_metrics,
            "shapes": shapes,
        },
    }


@app.post("/refit")
def refit(request: RefitRequest):
    """Refit from frontend-edited shape functions and return refreshed shapes/data."""
    response = build_train_response(
        request,
        edited_partials=request.partials,
        locked_features=request.locked_features,
    )
    return _to_jsonable(response)


@app.get("/models")
def list_models():
    # Return names without extensions for display and selection.
    if not MODELS_DIR.exists():
        return {"models": []}
    names = sorted([p.stem for p in MODELS_DIR.glob("*.json") if p.is_file()])
    return {"models": names}


def normalize_stored_model_payload(payload: Dict) -> Dict:
    # Support legacy preset JSONs by mapping them into the current TrainResponse shape.
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

    model_type = payload.get("model_type") or payload.get("source") or "igann"
    task = payload.get("task") or "regression"
    points = int(payload.get("points") or 250)
    train_metrics = payload.get("trainMetrics") or {"count": len(payload.get("y") or [])}
    test_metrics = payload.get("testMetrics") or {"count": len(payload.get("testY") or [])}
    timestamp = int(time.time() * 1000)

    return {
        "model": {
            "dataset": payload.get("dataset") or "unknown",
            "model_type": model_type if model_type in {"igann", "igann_interactive"} else "igann",
            "task": task if task in {"regression", "classification"} else "regression",
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
            "locked_features": [str(f) for f in (payload.get("locked_features") or [])],
            "refit_from_edits": False,
            "intercept": float(payload.get("intercept") or 0.0),
            "trainMetrics": train_metrics,
            "testMetrics": test_metrics,
            "shapes": shapes,
        },
    }


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
        return normalize_stored_model_payload(json.load(f))


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
        return normalize_stored_model_payload(json.load(f))


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
