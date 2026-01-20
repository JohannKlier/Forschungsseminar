"""Lightweight Python trainer service for the bike_hourly dataset."""

from typing import Dict, List
from collections.abc import Mapping, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from igann import IGANN
from sklearn.model_selection import train_test_split

from adult_preprocessing import preprocess_adult_income
from bike_preprocessing import preprocess_bike_hourly
import json
from pathlib import Path

# Default grid density used when IGANN does not provide a curve for a feature.
DEFAULT_GRID_POINTS = 120
MODELS_DIR = Path(__file__).parent / "models"
SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"


class SaveModelRequest(BaseModel):
    name: str
    payload: Dict

# FastAPI app setup with permissive CORS so the frontend can call it directly.
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    dataset: str
    bandwidth: float
    seed: int
    points: int | None = 10


def discretize_curve(grid: List[float], curve: List[float], num_points: int = 20) -> Dict[str, List[float]]:
    """
    Sample a smooth curve at a fixed number of points so the frontend can edit/send
    plain x/y lists instead of dense floats.
    """
    if not grid or not curve:
        return {"x": [], "y": []}

    xs: List[float] = []
    ys: List[float] = []
    for i in range(num_points):
        target_x = grid[0] + (grid[-1] - grid[0]) * i / max(1, num_points - 1)
        for j in range(len(grid) - 1):
            if grid[j] <= target_x <= grid[j + 1]:
                t = (target_x - grid[j]) / max(1e-9, (grid[j + 1] - grid[j]))
                value = curve[j] * (1 - t) + curve[j + 1] * t
                xs.append(target_x)
                ys.append(value)
                break
        else:
            xs.append(target_x)
            ys.append(curve[min(len(curve) - 1, len(xs) - 1)])

    return {"x": xs, "y": ys}


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


@app.post("/train")
def train(request: TrainRequest):
    # Entrypoint used by the frontend for training and returning editable partials.
    response = build_train_response(request)
    return _to_jsonable(response)


def build_train_response(request: TrainRequest):
    # Validate the dataset early to keep error responses simple and explicit.
    if request.dataset not in {"bike_hourly", "adult_income"}:
        raise HTTPException(status_code=400, detail="Only bike_hourly and adult_income are supported.")
    # Clamp point count to keep UI responsiveness predictable.
    num_points = max(2, min(200, request.points or 10))
    task_type = "classification" if request.dataset == "adult_income" else "regression"
    igann_task = "regression" if request.dataset == "adult_income" else task_type
    # Dataset-specific preprocessing returns: features, target, categorical metadata, labels.
    if request.dataset == "adult_income":
        X_proc, y_full, cat_info, labels = preprocess_adult_income(request.seed)
    else:
        X_proc, y_full, cat_info, labels = preprocess_bike_hourly(request.seed)
    # Train/test split for metrics and visual validation.
    feature_keys = list(X_proc.columns)
    X_train_df, X_test_df, y_train_arr, y_test_arr = train_test_split(
        X_proc, y_full, test_size=0.2, random_state=request.seed
    )
    y_train = np.array(y_train_arr).flatten()
    y_test = np.array(y_test_arr).flatten()

    # IGANN model chosen for interpretable shape functions.
    igann = IGANN(
        task=igann_task,
        n_estimators=100,
        boost_rate=0.1,
        init_reg=1,
        elm_alpha=1,
        early_stopping=50,
        device="cpu",
        random_state=request.seed,
        verbose=0,
    )
    igann.fit(X_train_df, y_train)

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
    feature_matrix = X_train_df.to_numpy()
    test_len = len(X_test_df)

    # Extract learned shape functions from IGANN for visualization.
    shape_functions = igann.get_shape_functions_as_dict()

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

    grid_points = max(num_points * 4, DEFAULT_GRID_POINTS)
    y_display = y_train.tolist()

    # Build partials: each feature gets editable x/y plus scatter data for UI rendering.
    partials = []
    for term_idx, key in enumerate(feature_keys):
        shape_fn = get_shape(key)
        contrib_train = contribs_train[term_idx] if term_idx < len(contribs_train) else np.zeros_like(y_train)
        feat_vals = feature_matrix[:, term_idx]
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
            # Continuous features: discretize the learned curve into editable points.
            xs = shape_fn.get("x", [])
            ys = shape_fn.get("y", [])
            if len(xs) and len(ys):
                pairs = sorted(zip(list(xs), list(ys)), key=lambda p: p[0])
                sx, sy = zip(*pairs)
                discrete = discretize_curve(list(sx), list(sy), num_points=num_points)
            else:
                fmin = float(np.min(feat_vals))
                fmax = float(np.max(feat_vals))
                if fmin == fmax:
                    fmin -= 1.0
                    fmax += 1.0
                grid = np.linspace(fmin, fmax, grid_points)
                discrete = {"x": grid.tolist(), "y": [0.0 for _ in grid]}
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
        "bandwidth": request.bandwidth,
        "points": num_points,
        "intercept": intercept_val,
        "partials": partials,
        "predictions": preds_train.tolist(),
        "y": y_display,
        "task": task_type,
        "source": "igann",
        "trainMetrics": train_metrics,
        "testMetrics": test_metrics,
        "testPreds": preds_test.tolist(),
        "testY": y_test.tolist(),
    }


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
