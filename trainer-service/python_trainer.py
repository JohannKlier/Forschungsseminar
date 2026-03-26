"""Lightweight Python trainer service for the bike_hourly dataset."""

from typing import Dict, List, Set
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
    selected_features: List[str] = []
    selected_interactions: List[str] | None = None
    selected_operations: List[Dict] = []
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
    feature_modes: Dict[str, str] = {}


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


def _get_numeric_col(key: str, X_df, cat_info: Dict) -> np.ndarray:
    """Return numerical values for a feature column, label-encoding categoricals."""
    if key in cat_info:
        cats = cat_info[key]
        cat_to_idx = {str(c): float(i) for i, c in enumerate(cats)}
        return np.array([cat_to_idx.get(str(v), 0.0) for v in X_df[key]], dtype=float)
    return X_df[key].values.astype(float)


def _operation_symbol(operator: str) -> str:
    symbols = {
        "product": "×",
        "sum": "+",
        "difference": "−",
        "ratio": "/",
        "absolute_difference": "|Δ|",
    }
    return symbols.get(operator, operator)


def _operation_key(left: str, right: str, operator: str) -> str:
    if operator == "product":
        return f"{left}__{right}"
    return f"{left}__{operator}__{right}"


def _operation_label(left: str, right: str, operator: str, label_map: Dict | None = None) -> str:
    left_label = label_map.get(left, left) if label_map else left
    right_label = label_map.get(right, right) if label_map else right
    return f"{left_label} {_operation_symbol(operator)} {right_label}"


def _operation_scalar_values(left_vals: np.ndarray, right_vals: np.ndarray, operator: str) -> np.ndarray:
    if operator == "product":
        return left_vals * right_vals
    if operator == "sum":
        return left_vals + right_vals
    if operator == "difference":
        return left_vals - right_vals
    if operator == "ratio":
        safe_den = np.where(np.abs(right_vals) < 1e-9, 1e-9, right_vals)
        return left_vals / safe_den
    if operator == "absolute_difference":
        return np.abs(left_vals - right_vals)
    raise HTTPException(status_code=400, detail=f"Unsupported operation operator: {operator}")


def _normalize_operation_specs(
    feature_keys: List[str],
    requested_interactions: List[str] | None,
    selected_operations: List[Dict] | None,
    label_map: Dict | None = None,
) -> List[Dict]:
    feature_key_set = set(feature_keys)
    if selected_operations:
        specs = []
        for raw in selected_operations:
            kind = str(raw.get("kind") or "interaction")
            if kind != "interaction":
                raise HTTPException(status_code=400, detail=f"Unsupported operation kind: {kind}")
            operator = str(raw.get("operator") or "product")
            if operator not in {"product", "sum", "difference", "ratio", "absolute_difference"}:
                raise HTTPException(status_code=400, detail=f"Unsupported operation operator: {operator}")
            sources = [str(source) for source in (raw.get("sources") or [])]
            if len(sources) != 2:
                raise HTTPException(status_code=400, detail="Each selected_operation must have exactly two sources.")
            left, right = sources
            unknown_sources = [source for source in sources if source not in feature_key_set]
            if unknown_sources:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown selected_operation sources: {', '.join(unknown_sources)}",
                )
            key = str(raw.get("key") or _operation_key(left, right, operator))
            label = str(raw.get("label") or _operation_label(left, right, operator, label_map))
            specs.append({"kind": kind, "operator": operator, "sources": [left, right], "key": key, "label": label})
        return specs

    from itertools import combinations as _combinations
    available_pairs = [(k1, k2) for k1, k2 in _combinations(feature_keys, 2)]
    available_pair_keys = {f"{k1}__{k2}" for k1, k2 in available_pairs}
    if requested_interactions is None:
        requested_set = available_pair_keys
    else:
        requested_set = {str(key) for key in requested_interactions}
        unknown_interactions = sorted(requested_set - available_pair_keys)
        if unknown_interactions:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown selected_interactions: {', '.join(unknown_interactions)}",
            )
    return [
        {
            "kind": "interaction",
            "operator": "product",
            "sources": [k1, k2],
            "key": f"{k1}__{k2}",
            "label": _operation_label(k1, k2, "product", label_map),
        }
        for k1, k2 in available_pairs
        if f"{k1}__{k2}" in requested_set
    ]


def _build_dummy_interaction_cols(k1: str, k2: str, X_df, cat_info: Dict) -> List[tuple]:
    """
    Create interaction feature columns using dummy encoding for categorical features.
    Returns list of (column_name, values) tuples.
    - num × num: single product column keyed by the display key
    - cat × num: one indicator×numeric column per level of the categorical feature
    - num × cat: one indicator×numeric column per level of the categorical feature
    - cat × cat: one indicator-pair column per (level1, level2) combination

    Internal column names use '___c{i}' / '___r{j}' / '___c{i}r{j}' suffixes so they
    are clearly distinct from the display key (which uses a single '__' separator).
    """
    is_cat1 = k1 in cat_info
    is_cat2 = k2 in cat_info
    display_key = f"{k1}__{k2}"

    if not is_cat1 and not is_cat2:
        vals = X_df[k1].values.astype(float) * X_df[k2].values.astype(float)
        return [(display_key, vals)]

    if is_cat1 and not is_cat2:
        num_vals = X_df[k2].values.astype(float)
        return [
            (f"{display_key}___c{i}", (X_df[k1].astype(str) == str(level)).values.astype(float) * num_vals)
            for i, level in enumerate(cat_info[k1])
        ]

    if not is_cat1 and is_cat2:
        num_vals = X_df[k1].values.astype(float)
        return [
            (f"{display_key}___r{j}", (X_df[k2].astype(str) == str(level)).values.astype(float) * num_vals)
            for j, level in enumerate(cat_info[k2])
        ]

    # cat × cat: single label-encoded product (avoids C₁×C₂ column explosion).
    # Both categoricals are encoded as integer indices, so the product is a composite integer
    # that IGANN can still learn a shape on, similar to the original approach.
    v1 = _get_numeric_col(k1, X_df, cat_info)
    v2 = _get_numeric_col(k2, X_df, cat_info)
    return [(display_key, v1 * v2)]


def _build_operation_cols(operation_spec: Dict, X_df, cat_info: Dict) -> List[tuple]:
    left, right = operation_spec["sources"]
    operator = operation_spec["operator"]
    key = operation_spec["key"]
    is_cat_left = left in cat_info
    is_cat_right = right in cat_info

    if operator == "product":
        raw_cols = _build_dummy_interaction_cols(left, right, X_df, cat_info)
        if len(raw_cols) == 1 and raw_cols[0][0] == f"{left}__{right}":
            return [(key, raw_cols[0][1])]
        renamed_cols = []
        prefix = f"{left}__{right}"
        for col_name, values in raw_cols:
            suffix = col_name[len(prefix):] if col_name.startswith(prefix) else ""
            renamed_cols.append((f"{key}{suffix}", values))
        return renamed_cols

    if is_cat_left or is_cat_right:
        raise HTTPException(
            status_code=400,
            detail=f"Operator '{operator}' currently supports numerical-numerical feature pairs only.",
        )

    left_vals = X_df[left].values.astype(float)
    right_vals = X_df[right].values.astype(float)
    return [(key, _operation_scalar_values(left_vals, right_vals, operator))]


def _build_2d_grid_from_dummies(
    k1: str, k2: str,
    dummy_cols: List[str],
    dummy_shapes: Dict[str, Dict],
    X_train_df,
    cat_info: Dict,
    label_map: Dict,
    n_grid: int = 15,
) -> Dict:
    """
    Build a 2D grid shape dict for an interaction pair from per-dummy shape functions.
    - num × num: delegates to _build_2d_grid (single product shape, unchanged)
    - cat × num: z[row=num_j][col=cat_i] = g_i(num_j)
    - num × cat: z[row=cat_j][col=num_i] = g_j(num_i)
    - cat × cat: z[row=cat2_j][col=cat1_i] = g_{i,j}(1)
    z convention matches _build_2d_grid: z[row=y-axis][col=x-axis].
    """
    is_cat1 = k1 in cat_info
    is_cat2 = k2 in cat_info
    display_key = f"{k1}__{k2}"
    base: Dict = {
        "key": display_key,
        "label": f"{label_map.get(k1, k1)} × {label_map.get(k2, k2)}",
        "label2": label_map.get(k2, k2),
        "editableX": [],
        "editableY": [],
    }

    def _interp(shape_fn: Dict, x_val: float) -> float:
        sx = _coerce_numeric_points(shape_fn.get("x"))
        sy = _coerce_numeric_points(shape_fn.get("y"))
        if len(sx) > 1:
            return float(np.interp(x_val, sx, sy))
        return float(sy[0]) if len(sy) == 1 else 0.0

    if not is_cat1 and not is_cat2 or (is_cat1 and is_cat2):
        # num × num or cat × cat: single product/label-encoded column, reuse existing logic
        return _build_2d_grid(k1, k2, dummy_shapes.get(display_key, {}), X_train_df, cat_info, label_map, n_grid)

    if is_cat1 and not is_cat2:
        # dummy_cols[i] is the shape for cat1[i] × num2
        categories1 = cat_info[k1]
        x2_grid = np.linspace(float(X_train_df[k2].min()), float(X_train_df[k2].max()), n_grid)
        z = [[_interp(dummy_shapes.get(col, {}), num_val) for col in dummy_cols] for num_val in x2_grid]
        return {**base, "editableZ": z, "xCategories": [str(c) for c in categories1], "gridX2": x2_grid.tolist()}

    if not is_cat1 and is_cat2:
        # dummy_cols[j] is the shape for num1 × cat2[j]
        categories2 = cat_info[k2]
        x1_grid = np.linspace(float(X_train_df[k1].min()), float(X_train_df[k1].max()), n_grid)
        z = [[_interp(dummy_shapes.get(col, {}), num_val) for num_val in x1_grid] for col in dummy_cols]
        return {**base, "editableZ": z, "gridX": x1_grid.tolist(), "yCategories": [str(c) for c in categories2]}

    # cat × cat: dummy_cols ordered as (i=0,j=0),(i=0,j=1),...,(i=n1-1,j=n2-1)
    # z[row=cat2_j][col=cat1_i] = g_{i,j}(1)
    categories1, categories2 = cat_info[k1], cat_info[k2]
    n1, n2 = len(categories1), len(categories2)
    z_vals = {(idx // n2, idx % n2): _interp(dummy_shapes.get(col, {}), 1.0) for idx, col in enumerate(dummy_cols)}
    z = [[z_vals.get((i, j), 0.0) for i in range(n1)] for j in range(n2)]
    return {**base, "editableZ": z, "xCategories": [str(c) for c in categories1], "yCategories": [str(c) for c in categories2]}


def _build_2d_grid(k1: str, k2: str, shape_1d: Dict, X_train_df, cat_info: Dict, label_map: Dict, n_grid: int = 15) -> Dict:
    """Convert a 1D interaction shape function (f(feat1*feat2)) into a 2D grid shape dict."""
    sx = _coerce_numeric_points(shape_1d.get("x"))
    sy = _coerce_numeric_points(shape_1d.get("y"))

    is_cat1, is_cat2 = k1 in cat_info, k2 in cat_info

    # x-axis (feat1)
    if is_cat1:
        x1_enc = np.arange(len(cat_info[k1]), dtype=float)
    else:
        x1_enc = np.linspace(float(X_train_df[k1].min()), float(X_train_df[k1].max()), n_grid)

    # y-axis (feat2)
    if is_cat2:
        x2_enc = np.arange(len(cat_info[k2]), dtype=float)
    else:
        x2_enc = np.linspace(float(X_train_df[k2].min()), float(X_train_df[k2].max()), n_grid)

    # z[row=y][col=x] = f(x1[col] * x2[row])
    z = []
    for x2 in x2_enc:
        row = []
        for x1 in x1_enc:
            t = float(x1 * x2)
            if len(sx) > 1:
                val = float(np.interp(t, sx, sy))
            elif len(sy) == 1:
                val = float(sy[0])
            else:
                val = 0.0
            row.append(val)
        z.append(row)

    result: Dict = {
        "key": f"{k1}__{k2}",
        "label": f"{label_map.get(k1, k1)} × {label_map.get(k2, k2)}",
        "label2": label_map.get(k2, k2),
        # 1-D product shape — used by the frontend worker to evaluate contributions
        "editableX": sx,
        "editableY": sy,
        # 2-D display grid
        "editableZ": z,
    }
    if is_cat1:
        result["xCategories"] = [str(c) for c in cat_info[k1]]
    else:
        result["gridX"] = x1_enc.tolist()
    if is_cat2:
        result["yCategories"] = [str(c) for c in cat_info[k2]]
    else:
        result["gridX2"] = x2_enc.tolist()

    return result


def _build_2d_grid_for_operation(operation_spec: Dict, shape_1d: Dict, X_train_df, cat_info: Dict, n_grid: int = 15) -> Dict:
    left, right = operation_spec["sources"]
    operator = operation_spec["operator"]
    if operator == "product":
        result = _build_2d_grid(left, right, shape_1d, X_train_df, cat_info, {}, n_grid)
        result["key"] = operation_spec["key"]
        result["label"] = operation_spec["label"]
        return result

    if left in cat_info or right in cat_info:
        raise HTTPException(
            status_code=400,
            detail=f"Operator '{operator}' currently supports numerical-numerical feature pairs only.",
        )

    sx = _coerce_numeric_points(shape_1d.get("x"))
    sy = _coerce_numeric_points(shape_1d.get("y"))
    x1_enc = np.linspace(float(X_train_df[left].min()), float(X_train_df[left].max()), n_grid)
    x2_enc = np.linspace(float(X_train_df[right].min()), float(X_train_df[right].max()), n_grid)

    z = []
    for x2 in x2_enc:
        row = []
        for x1 in x1_enc:
            t = float(_operation_scalar_values(np.array([x1]), np.array([x2]), operator)[0])
            if len(sx) > 1:
                val = float(np.interp(t, sx, sy))
            elif len(sy) == 1:
                val = float(sy[0])
            else:
                val = 0.0
            row.append(val)
        z.append(row)

    return {
        "key": operation_spec["key"],
        "label": operation_spec["label"],
        "label2": right,
        "editableX": sx,
        "editableY": sy,
        "editableZ": z,
        "gridX": x1_enc.tolist(),
        "gridX2": x2_enc.tolist(),
    }



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

def linearize_features(
    feature_dict: Dict,
    features_to_reinit: List[str],
    features_train: Dict[str, List],
    y_train: np.ndarray,
    cat_info: Dict,
    num_points: int,
) -> Dict:
    """
    Replace shape functions for reinit features with a joint linear fit on
    residuals computed from the kept features' contributions.
    Numerical features get a linear shape; categorical features get mean residuals per category.
    """
    from sklearn.linear_model import LinearRegression

    if not features_to_reinit:
        return feature_dict

    result = {k: dict(v) for k, v in feature_dict.items()}

    # Sum contributions of kept features (everything not being reinitialized)
    kept_keys = [k for k in feature_dict if k not in features_to_reinit]
    kept_contribs = np.zeros(len(y_train), dtype=float)
    for key in kept_keys:
        contribs = np.array(
            evaluate_contribs(feature_dict[key], features_train.get(key, []), cat_info.get(key)),
            dtype=float,
        )
        if len(contribs) == len(y_train):
            kept_contribs += contribs

    residuals = y_train - kept_contribs

    # Joint linear regression for numerical reinit features
    reinit_numerical = [k for k in features_to_reinit if k not in cat_info and k in features_train]
    if reinit_numerical:
        X_reinit = np.column_stack(
            [np.array(features_train[k], dtype=float) for k in reinit_numerical]
        )
        lr = LinearRegression(fit_intercept=True).fit(X_reinit, residuals)
        for i, key in enumerate(reinit_numerical):
            vals = np.array(features_train[key], dtype=float)
            x_min, x_max = float(vals.min()), float(vals.max())
            x_points = (
                np.linspace(x_min, x_max, num_points).tolist()
                if x_min != x_max
                else [x_min] * num_points
            )
            y_points = (lr.coef_[i] * np.array(x_points)).tolist()
            result[key] = {"datatype": "numerical", "x": x_points, "y": y_points}

    # Mean-residual encoding for categorical reinit features
    reinit_categorical = [k for k in features_to_reinit if k in cat_info and k in features_train]
    for key in reinit_categorical:
        categories = cat_info.get(key, [])
        feat_vals = features_train.get(key, [])
        cat_y: List[float] = []
        for cat in categories:
            mask = np.array([str(v) == str(cat) for v in feat_vals])
            cat_y.append(float(np.mean(residuals[mask])) if mask.any() else 0.0)
        result[key] = {
            "datatype": "categorical",
            "x": [str(c) for c in categories],
            "y": cat_y,
        }

    return result


def build_train_response(
    request: TrainRequest,
    edited_partials: List[Dict] | None = None,
    locked_features: List[str] | None = None,
    feature_modes: Dict[str, str] | None = None,
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

    requested_features = [str(feature) for feature in (request.selected_features or [])]
    if requested_features:
        unknown_features = [feature for feature in requested_features if feature not in X_proc.columns]
        if unknown_features:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown selected_features: {', '.join(unknown_features)}",
            )
        X_proc = X_proc.loc[:, requested_features].copy()
        cat_info = {key: value for key, value in cat_info.items() if key in requested_features}
        labels = {key: labels.get(key, key) for key in requested_features}

    if X_proc.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No features selected for training.")

    # Train/test split for metrics and visual validation.
    feature_keys = list(X_proc.columns)
    X_train_df, X_test_df, y_train_arr, y_test_arr = train_test_split(
        X_proc, y_full, test_size=0.2, random_state=request.seed
    )
    y_train = np.array(y_train_arr).astype(float).flatten()
    y_test = np.array(y_test_arr).astype(float).flatten()
    use_scale_y = bool(request.scale_y) if task_type == "regression" else False

    # Parse feature modes before building interaction pairs so deactivated pairs can be excluded.
    locked_set: Set[str] = set()
    deactivate_set: Set[str] = set()
    if feature_modes:
        locked_set = {k for k, m in feature_modes.items() if m == "lock"}
        deactivate_set = {k for k, m in feature_modes.items() if m == "deactivate"}
    elif locked_features:
        locked_set = {str(f) for f in locked_features}

    operation_specs = _normalize_operation_specs(
        feature_keys,
        request.selected_interactions,
        request.selected_operations,
        labels,
    )
    active_operation_specs = [
        spec for spec in operation_specs
        if spec["key"] not in deactivate_set
        and spec["sources"][0] not in deactivate_set
        and spec["sources"][1] not in deactivate_set
    ]
    # interaction_dummy_cols: operation key -> list of internal column names used for model fitting.
    interaction_dummy_cols: Dict[str, List[str]] = {}
    all_dummy_keys: List[str] = []
    extra_train_cols: Dict[str, np.ndarray] = {}
    extra_test_cols: Dict[str, np.ndarray] = {}
    for spec in active_operation_specs:
        display_key = spec["key"]
        dummies = _build_operation_cols(spec, X_train_df, cat_info)
        col_names = [col for col, _ in dummies]
        for col, vals in dummies:
            extra_train_cols[col] = vals
        if len(X_test_df):
            for col, vals in _build_operation_cols(spec, X_test_df, cat_info):
                extra_test_cols[col] = vals
        interaction_dummy_cols[display_key] = col_names
        all_dummy_keys.extend(col_names)
    # Build augmented DataFrames in one concat to avoid fragmentation warnings.
    X_train_aug = pd.concat([X_train_df, pd.DataFrame(extra_train_cols, index=X_train_df.index)], axis=1)
    X_test_aug = pd.concat([X_test_df, pd.DataFrame(extra_test_cols, index=X_test_df.index)], axis=1) if len(X_test_df) else X_test_df.copy()

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
        fit_cols = [col for col in feature_keys if col not in locked_set and col not in deactivate_set]
        fit_cols_all = fit_cols + all_dummy_keys
        if not fit_cols_all:
            raise HTTPException(
                status_code=400,
                detail="No features left for refit.",
            )
        use_preserved_partials = bool(locked_set)
        if use_preserved_partials:
            feature_dict = build_feature_dict_from_partials(edited_partials, cat_info)

            # Remove deactivated features from the base dict.
            for key in list(deactivate_set):
                feature_dict.pop(key, None)

            # All non-locked, non-deactivated features are reinitialized from linear on residuals.
            # Interaction features are always reinitialized (never locked/deactivated).
            reinit_cols = [k for k in feature_keys if k not in locked_set and k not in deactivate_set]
            reinit_cols_all = reinit_cols + all_dummy_keys
            if reinit_cols_all:
                features_for_reinit = {k: X_train_df[k].tolist() for k in feature_keys}
                features_for_reinit.update({dk: X_train_aug[dk].tolist() for dk in all_dummy_keys})
                feature_dict = linearize_features(
                    feature_dict, reinit_cols_all, features_for_reinit, y_train, cat_info, num_points
                )

            igann.fit_from_shape_functions(
                X_train_aug[fit_cols_all],
                y_train,
                feature_dict,
            )
        else:
            # Default refinement path: retrain the current active basis from scratch rather than
            # seeding from previously edited partials. This avoids path-dependent behavior when
            # interactions/features are removed before refinement.
            igann.fit(X_train_aug[fit_cols_all], y_train)
    else:
        igann.fit(X_train_aug, y_train)

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

    # Map feature keys to human-readable labels for display (add operation labels too).
    label_map = dict(labels)
    for spec in active_operation_specs:
        label_map[spec["key"]] = spec["label"]

    features_train = {k: X_train_df[k].tolist() for k in feature_keys}
    features_test = {k: X_test_df[k].tolist() for k in feature_keys} if len(X_test_df) else {}
    # Normalize categorical values to strings to match IGANN shape keys
    for cat_key in cat_info.keys():
        if cat_key in features_train:
            features_train[cat_key] = [str(v) for v in features_train[cat_key]]
        if cat_key in features_test:
            features_test[cat_key] = [str(v) for v in features_test[cat_key]]
    test_len = len(X_test_df)
    # Interaction per-row contributions are computed after shape extraction below.

    # Shape functions:
    # - default: interactive GAM wrapper dict or base IGANN dict
    # - refit-from-edits: merge edited base dict with freshly derived shapes from current model
    if edited_partials and model_type == "igann_interactive" and locked_set:
        base_shapes = build_feature_dict_from_partials(edited_partials, cat_info)
        learned_shapes = igann.get_shape_functions_as_dict()
        shape_functions = merge_learned_shapes_preserve_base_grid(base_shapes, learned_shapes, feature_keys)
        # Add learned dummy interaction shapes directly from the model.
        for dummy_key in all_dummy_keys:
            if dummy_key in learned_shapes:
                shape_functions[dummy_key] = learned_shapes[dummy_key]
        # Remove deactivated features from returned shapes.
        for key in list(deactivate_set):
            shape_functions.pop(key, None)
            for dummy_key in interaction_dummy_cols.get(key, []):
                shape_functions.pop(dummy_key, None)
    else:
        shape_functions = (
            igann.get_gam_feature_dict()
            if getattr(igann, "GAM", None) is not None
            else igann.get_shape_functions_as_dict()
        )
    if not shape_functions:
        raise HTTPException(status_code=500, detail="Model did not produce shape functions.")
    all_model_keys = feature_keys + all_dummy_keys
    shape_functions = normalize_numeric_shape_points(shape_functions, all_model_keys, cat_info, num_points)
    if center_shapes and edited_partials and model_type == "igann_interactive":
        shape_functions, _ = center_shape_functions_for_data(
            shape_functions, feature_keys, features_train, cat_info
        )

    def get_shape(key: str) -> Dict:
        return shape_functions.get(key, {})

    # Build per-row contributions for each feature and aggregate totals.
    active_keys = [k for k in feature_keys if k not in deactivate_set]
    contribs_train: List[np.ndarray] = []
    contribs_test: List[np.ndarray] = []
    for key in active_keys:
        shape_fn = get_shape(key)
        contribs = np.array(evaluate_contribs(shape_fn, features_train[key], cat_info.get(key)))
        contribs_train.append(contribs)
        if test_len:
            contribs_t = np.array(evaluate_contribs(shape_fn, features_test[key], cat_info.get(key)))
            contribs_test.append(contribs_t)

    # Interaction contributions: sum dummy shapes per display key, store in features_train for
    # frontend importance calculation (variance of actual per-row contributions).
    for spec in active_operation_specs:
        display_key = spec["key"]
        dummy_cols_for_pair = interaction_dummy_cols[display_key]
        pair_train = np.zeros(len(y_train))
        pair_test = np.zeros(test_len) if test_len else np.array([])
        for col_name in dummy_cols_for_pair:
            shape_fn = shape_functions.get(col_name, {})
            pair_train += np.array(evaluate_contribs(shape_fn, X_train_aug[col_name].tolist(), None))
            if test_len:
                pair_test += np.array(evaluate_contribs(shape_fn, X_test_aug[col_name].tolist(), None))
        contribs_train.append(pair_train)
        if test_len:
            contribs_test.append(pair_test)
        features_train[display_key] = pair_train.tolist()
        if test_len:
            features_test[display_key] = pair_test.tolist()

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

    # Build partials: each active (non-deactivated) feature gets editable x/y for the UI.
    partials = []
    for term_idx, key in enumerate(active_keys):
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

    # Build 2D grid visualizations for operation shape functions.
    interaction_shapes = []
    for spec in active_operation_specs:
        display_key = spec["key"]
        dummy_cols_for_pair = interaction_dummy_cols[display_key]
        dummy_shapes_for_pair = {col: shape_functions.get(col, {}) for col in dummy_cols_for_pair}
        if any(dummy_shapes_for_pair.values()):
            if spec["operator"] == "product" and (spec["sources"][0] in cat_info or spec["sources"][1] in cat_info):
                interaction_shapes.append(
                    _build_2d_grid_from_dummies(
                        spec["sources"][0],
                        spec["sources"][1],
                        dummy_cols_for_pair,
                        dummy_shapes_for_pair,
                        X_train_df,
                        cat_info,
                        label_map,
                        n_grid=15,
                    )
                )
            else:
                interaction_shapes.append(
                    _build_2d_grid_for_operation(
                        spec,
                        dummy_shapes_for_pair.get(display_key, {}),
                        X_train_df,
                        cat_info,
                        n_grid=15,
                    )
                )

    version_id = str(int(time.time() * 1000))
    is_refit = bool(edited_partials)

    return {
        "model": {
            "dataset": request.dataset,
            "model_type": model_type,
            "task": task_type,
            "selected_features": feature_keys,
            "selected_interactions": [spec["key"] for spec in active_operation_specs if spec["operator"] == "product"],
            "selected_operations": active_operation_specs,
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
            "featureLabels": {k: label_map[k] for k in feature_keys},
        },
        "version": {
            "versionId": version_id,
            "timestamp": int(time.time() * 1000),
            "source": "refit" if is_refit else "train",
            "center_shapes": center_shapes,
            "locked_features": list(locked_set) if feature_modes else [str(f) for f in (locked_features or [])],
            "feature_modes": {str(k): str(v) for k, v in (feature_modes or {}).items()},
            "refit_from_edits": is_refit,
            "intercept": intercept_val,
            "trainMetrics": train_metrics,
            "testMetrics": test_metrics,
            "shapes": shapes + interaction_shapes,
        },
    }


@app.post("/refit")
def refit(request: RefitRequest):
    """Refit from frontend-edited shape functions and return refreshed shapes/data."""
    response = build_train_response(
        request,
        edited_partials=request.partials,
        locked_features=request.locked_features,
        feature_modes=request.feature_modes or None,
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
            "locked_features": [str(f) for f in (payload.get("locked_features") or [])],
            "feature_modes": payload.get("feature_modes") or {},
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
