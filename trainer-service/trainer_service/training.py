from __future__ import annotations

import time
from typing import Dict, List

from fastapi import HTTPException
from igann import IGANN, IGANN_interactive
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from trainer_service.preprocessing import (
    preprocess_bike_hourly,
    preprocess_mimic4_mean_100_full,
)
from trainer_service.schemas import TrainRequest


def interpolate_numeric(values: List[float], xs: List[float], ys: List[float]) -> List[float]:
    """Linear interpolation helper for numeric shape functions."""
    if xs is None or ys is None:
        return [0.0 for _ in values]
    if len(xs) == 0 or len(ys) == 0 or len(xs) != len(ys):
        return [0.0 for _ in values]
    pairs = sorted(zip(list(xs), list(ys)), key=lambda pair: pair[0])
    sorted_xs, sorted_ys = zip(*pairs)
    result: List[float] = []
    for value in values:
        if value <= sorted_xs[0]:
            result.append(sorted_ys[0])
            continue
        if value >= sorted_xs[-1]:
            result.append(sorted_ys[-1])
            continue
        hi = next(i for i, candidate in enumerate(sorted_xs) if candidate >= value)
        lo = hi - 1
        ratio = (value - sorted_xs[lo]) / max(1e-9, sorted_xs[hi] - sorted_xs[lo])
        result.append(sorted_ys[lo] * (1 - ratio) + sorted_ys[hi] * ratio)
    return result


def _coerce_numeric_points(values) -> List[float]:
    """Normalize shape point containers without relying on ambiguous array truthiness."""
    if values is None:
        return []
    return [float(value) for value in values]


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
        mapping = {category: vals[i] if i < len(vals) else 0.0 for i, category in enumerate(cats)}
        contribs: List[float] = []
        for value in feat_values:
            if isinstance(value, str):
                category_name = value
            else:
                index = int(round(value))
                category_name = categories[index] if categories and index < len(categories) else None
            contribs.append(mapping.get(category_name, 0.0))
        return contribs
    xs = shape_fn.get("x", [])
    ys = shape_fn.get("y", [])
    return interpolate_numeric(feat_values, xs, ys)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for additive classification scores."""
    return 1 / (1 + np.exp(-np.clip(values, -500, 500)))


def _model_intercept(model) -> float:
    """Return the intercept belonging to the exported additive shape functions."""
    gam = getattr(model, "GAM", None)
    if gam is not None and hasattr(gam, "intercept_"):
        return float(np.asarray(gam.intercept_).reshape(-1)[0])

    linear_model = getattr(model, "linear_model", None)
    if linear_model is not None and hasattr(linear_model, "intercept_"):
        return float(np.asarray(linear_model.intercept_).reshape(-1)[0])

    return 0.0


def normalize_numeric_shape_points(
    shape_functions: Dict,
    feature_keys: List[str],
    cat_info: Dict,
    num_points: int,
) -> Dict:
    """Ensure numeric shape functions use exactly ``num_points`` knots."""
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

        pairs = sorted(zip(xs_raw[:pair_count], ys_raw[:pair_count]), key=lambda pair: pair[0])
        xs_sorted = [pair[0] for pair in pairs]
        ys_sorted = [pair[1] for pair in pairs]
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


def _get_numeric_col(key: str, x_frame, cat_info: Dict) -> np.ndarray:
    """Return numerical values for a feature column, label-encoding categoricals."""
    if key in cat_info:
        categories = cat_info[key]
        cat_to_idx = {str(category): float(i) for i, category in enumerate(categories)}
        return np.array([cat_to_idx.get(str(value), 0.0) for value in x_frame[key]], dtype=float)
    return x_frame[key].values.astype(float)


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
        safe_denominator = np.where(np.abs(right_vals) < 1e-9, 1e-9, right_vals)
        return left_vals / safe_denominator
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

    from itertools import combinations as combinations

    available_pairs = [(k1, k2) for k1, k2 in combinations(feature_keys, 2)]
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


def _build_dummy_interaction_cols(k1: str, k2: str, x_frame, cat_info: Dict) -> List[tuple]:
    """
    Create interaction feature columns using dummy encoding for categorical features.

    Internal column names use explicit suffixes so they stay distinct from the display key.
    """
    is_cat1 = k1 in cat_info
    is_cat2 = k2 in cat_info
    display_key = f"{k1}__{k2}"

    if not is_cat1 and not is_cat2:
        vals = x_frame[k1].values.astype(float) * x_frame[k2].values.astype(float)
        return [(display_key, vals)]

    if is_cat1 and not is_cat2:
        num_vals = x_frame[k2].values.astype(float)
        return [
            (f"{display_key}___c{i}", (x_frame[k1].astype(str) == str(level)).values.astype(float) * num_vals)
            for i, level in enumerate(cat_info[k1])
        ]

    if not is_cat1 and is_cat2:
        num_vals = x_frame[k1].values.astype(float)
        return [
            (f"{display_key}___r{j}", (x_frame[k2].astype(str) == str(level)).values.astype(float) * num_vals)
            for j, level in enumerate(cat_info[k2])
        ]

    v1 = _get_numeric_col(k1, x_frame, cat_info)
    v2 = _get_numeric_col(k2, x_frame, cat_info)
    return [(display_key, v1 * v2)]


def _build_operation_cols(operation_spec: Dict, x_frame, cat_info: Dict) -> List[tuple]:
    left, right = operation_spec["sources"]
    operator = operation_spec["operator"]
    key = operation_spec["key"]
    is_cat_left = left in cat_info
    is_cat_right = right in cat_info

    if operator == "product":
        raw_cols = _build_dummy_interaction_cols(left, right, x_frame, cat_info)
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

    left_vals = x_frame[left].values.astype(float)
    right_vals = x_frame[right].values.astype(float)
    return [(key, _operation_scalar_values(left_vals, right_vals, operator))]


def _build_2d_grid_from_dummies(
    k1: str,
    k2: str,
    dummy_cols: List[str],
    dummy_shapes: Dict[str, Dict],
    x_train,
    cat_info: Dict,
    label_map: Dict,
    n_grid: int = 15,
) -> Dict:
    """
    Build a 2D grid shape dict for an interaction pair from per-dummy shape functions.
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

    if (not is_cat1 and not is_cat2) or (is_cat1 and is_cat2):
        return _build_2d_grid(k1, k2, dummy_shapes.get(display_key, {}), x_train, cat_info, label_map, n_grid)

    if is_cat1 and not is_cat2:
        categories1 = cat_info[k1]
        x2_grid = np.linspace(float(x_train[k2].min()), float(x_train[k2].max()), n_grid)
        z = [[_interp(dummy_shapes.get(col, {}), num_val) for col in dummy_cols] for num_val in x2_grid]
        return {**base, "editableZ": z, "xCategories": [str(category) for category in categories1], "gridX2": x2_grid.tolist()}

    if not is_cat1 and is_cat2:
        categories2 = cat_info[k2]
        x1_grid = np.linspace(float(x_train[k1].min()), float(x_train[k1].max()), n_grid)
        z = [[_interp(dummy_shapes.get(col, {}), num_val) for num_val in x1_grid] for col in dummy_cols]
        return {**base, "editableZ": z, "gridX": x1_grid.tolist(), "yCategories": [str(category) for category in categories2]}

    categories1, categories2 = cat_info[k1], cat_info[k2]
    n1, n2 = len(categories1), len(categories2)
    z_vals = {(idx // n2, idx % n2): _interp(dummy_shapes.get(col, {}), 1.0) for idx, col in enumerate(dummy_cols)}
    z = [[z_vals.get((i, j), 0.0) for i in range(n1)] for j in range(n2)]
    return {
        **base,
        "editableZ": z,
        "xCategories": [str(category) for category in categories1],
        "yCategories": [str(category) for category in categories2],
    }


def _build_2d_grid(k1: str, k2: str, shape_1d: Dict, x_train, cat_info: Dict, label_map: Dict, n_grid: int = 15) -> Dict:
    """Convert a 1D interaction shape function into a 2D grid shape dict."""
    sx = _coerce_numeric_points(shape_1d.get("x"))
    sy = _coerce_numeric_points(shape_1d.get("y"))

    is_cat1, is_cat2 = k1 in cat_info, k2 in cat_info

    if is_cat1:
        x1_encoded = np.arange(len(cat_info[k1]), dtype=float)
    else:
        x1_encoded = np.linspace(float(x_train[k1].min()), float(x_train[k1].max()), n_grid)

    if is_cat2:
        x2_encoded = np.arange(len(cat_info[k2]), dtype=float)
    else:
        x2_encoded = np.linspace(float(x_train[k2].min()), float(x_train[k2].max()), n_grid)

    z = []
    for x2 in x2_encoded:
        row = []
        for x1 in x1_encoded:
            term = float(x1 * x2)
            if len(sx) > 1:
                value = float(np.interp(term, sx, sy))
            elif len(sy) == 1:
                value = float(sy[0])
            else:
                value = 0.0
            row.append(value)
        z.append(row)

    result: Dict = {
        "key": f"{k1}__{k2}",
        "label": f"{label_map.get(k1, k1)} × {label_map.get(k2, k2)}",
        "label2": label_map.get(k2, k2),
        "editableX": sx,
        "editableY": sy,
        "editableZ": z,
    }
    if is_cat1:
        result["xCategories"] = [str(category) for category in cat_info[k1]]
    else:
        result["gridX"] = x1_encoded.tolist()
    if is_cat2:
        result["yCategories"] = [str(category) for category in cat_info[k2]]
    else:
        result["gridX2"] = x2_encoded.tolist()

    return result


def _build_2d_grid_for_operation(operation_spec: Dict, shape_1d: Dict, x_train, cat_info: Dict, n_grid: int = 15) -> Dict:
    left, right = operation_spec["sources"]
    operator = operation_spec["operator"]
    if operator == "product":
        result = _build_2d_grid(left, right, shape_1d, x_train, cat_info, {}, n_grid)
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
    x1_encoded = np.linspace(float(x_train[left].min()), float(x_train[left].max()), n_grid)
    x2_encoded = np.linspace(float(x_train[right].min()), float(x_train[right].max()), n_grid)

    z = []
    for x2 in x2_encoded:
        row = []
        for x1 in x1_encoded:
            term = float(_operation_scalar_values(np.array([x1]), np.array([x2]), operator)[0])
            if len(sx) > 1:
                value = float(np.interp(term, sx, sy))
            elif len(sy) == 1:
                value = float(sy[0])
            else:
                value = 0.0
            row.append(value)
        z.append(row)

    return {
        "key": operation_spec["key"],
        "label": operation_spec["label"],
        "label2": right,
        "editableX": sx,
        "editableY": sy,
        "editableZ": z,
        "gridX": x1_encoded.tolist(),
        "gridX2": x2_encoded.tolist(),
    }


def _load_dataset(request: TrainRequest):
    if request.dataset == "mimic4_mean_100_full":
        return preprocess_mimic4_mean_100_full(request.seed, request.sample_size)
    return preprocess_bike_hourly(request.seed)


def build_dataset_feature_summary(dataset: str, seed: int = 3) -> Dict:
    supported_datasets = {"bike_hourly", "mimic4_mean_100_full"}
    if dataset not in supported_datasets:
        raise HTTPException(
            status_code=400,
            detail="Only bike_hourly and mimic4_mean_100_full are supported.",
        )

    request = TrainRequest(dataset=dataset, seed=seed)
    x_processed, _y_full, cat_info, labels, descriptions, _operations = _load_dataset(request)
    features = []

    for key in x_processed.columns:
        label = labels.get(key, key)
        description = descriptions.get(key, "")
        if key in cat_info:
            counts = x_processed[key].astype(str).value_counts(dropna=False).to_dict()
            categories = [
                {"label": category, "count": int(counts.get(str(category), 0))}
                for category in cat_info[key]
            ]
            features.append({
                "key": key,
                "label": label,
                "description": description,
                "kind": "categorical",
                "categories": categories,
            })
            continue

        values = pd.to_numeric(x_processed[key], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            bins = []
            min_value = None
            max_value = None
        else:
            counts, edges = np.histogram(values, bins=min(24, max(8, int(np.sqrt(len(values))))))
            bins = [int(count) for count in counts.tolist()]
            min_value = float(edges[0])
            max_value = float(edges[-1])
        features.append({
            "key": key,
            "label": label,
            "description": description,
            "kind": "continuous",
            "bins": bins,
            "min": min_value,
            "max": max_value,
        })

    return {"dataset": dataset, "features": features}


def _calc_metrics(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
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


def build_train_response(request: TrainRequest):
    supported_datasets = {"bike_hourly", "mimic4_mean_100_full"}
    if request.dataset not in supported_datasets:
        raise HTTPException(
            status_code=400,
            detail="Only bike_hourly and mimic4_mean_100_full are supported.",
        )
    model_type = request.model_type if request.model_type in {"igann", "igann_interactive"} else "igann_interactive"
    center_shapes = bool(getattr(request, "center_shapes", False))

    num_points = max(2, min(250, request.points or 250))
    n_estimators = max(10, min(500, request.n_estimators))
    boost_rate = max(0.01, min(1.0, request.boost_rate))
    init_reg = max(0.01, min(10.0, request.init_reg))
    elm_alpha = max(0.0001, min(10.0, request.elm_alpha))
    early_stopping = max(5, min(200, request.early_stopping))

    classification_datasets = {"mimic4_mean_100_full"}
    task_type = "classification" if request.dataset in classification_datasets else "regression"
    igann_task = task_type

    x_processed, y_full, cat_info, labels, descriptions, preprocessing_operations = _load_dataset(request)

    requested_features = [str(feature) for feature in (request.selected_features or [])]
    if requested_features:
        unknown_features = [feature for feature in requested_features if feature not in x_processed.columns]
        if unknown_features:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown selected_features: {', '.join(unknown_features)}",
            )
        x_processed = x_processed.loc[:, requested_features].copy()
        cat_info = {key: value for key, value in cat_info.items() if key in requested_features}
        labels = {key: labels.get(key, key) for key in requested_features}

    if x_processed.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No features selected for training.")

    feature_keys = list(x_processed.columns)
    stratify = None
    if task_type == "classification":
        unique_targets, target_counts = np.unique(y_full, return_counts=True)
        if len(unique_targets) > 1 and int(np.min(target_counts)) >= 2:
            stratify = y_full
    x_train_df, x_test_df, y_train_arr, y_test_arr = train_test_split(
        x_processed,
        y_full,
        test_size=0.2,
        random_state=request.seed,
        stratify=stratify,
    )
    y_train = np.array(y_train_arr).astype(float).flatten()
    y_test = np.array(y_test_arr).astype(float).flatten()
    use_scale_y = bool(request.scale_y) if task_type == "regression" else False

    if preprocessing_operations:
        operation_specs = _normalize_operation_specs(feature_keys, None, preprocessing_operations, labels)
    else:
        operation_specs = []
    active_operation_specs = operation_specs

    interaction_dummy_cols: Dict[str, List[str]] = {}
    all_dummy_keys: List[str] = []
    extra_train_cols: Dict[str, np.ndarray] = {}
    extra_test_cols: Dict[str, np.ndarray] = {}
    for spec in active_operation_specs:
        display_key = spec["key"]
        dummies = _build_operation_cols(spec, x_train_df, cat_info)
        col_names = [col for col, _ in dummies]
        for col, vals in dummies:
            extra_train_cols[col] = vals
        if len(x_test_df):
            for col, vals in _build_operation_cols(spec, x_test_df, cat_info):
                extra_test_cols[col] = vals
        interaction_dummy_cols[display_key] = col_names
        all_dummy_keys.extend(col_names)

    x_train_aug = pd.concat([x_train_df, pd.DataFrame(extra_train_cols, index=x_train_df.index)], axis=1)
    x_test_aug = pd.concat([x_test_df, pd.DataFrame(extra_test_cols, index=x_test_df.index)], axis=1) if len(x_test_df) else x_test_df.copy()

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
        model_kwargs["GAM_detail"] = num_points
    igann = model_cls(**model_kwargs)

    igann.fit(x_train_aug, y_train)

    if center_shapes and model_type == "igann_interactive" and hasattr(igann, "center_shape_functions"):
        igann.center_shape_functions(x_train_aug, update_intercept=True)

    label_map = dict(labels)
    for spec in active_operation_specs:
        label_map[spec["key"]] = spec["label"]

    features_train = {key: x_train_df[key].tolist() for key in feature_keys}
    features_test = {key: x_test_df[key].tolist() for key in feature_keys} if len(x_test_df) else {}
    for cat_key in cat_info.keys():
        if cat_key in features_train:
            features_train[cat_key] = [str(value) for value in features_train[cat_key]]
        if cat_key in features_test:
            features_test[cat_key] = [str(value) for value in features_test[cat_key]]
    test_len = len(x_test_df)

    shape_functions = igann.get_gam_feature_dict() if getattr(igann, "GAM", None) is not None else igann.get_shape_functions_as_dict()
    if not shape_functions:
        raise HTTPException(status_code=500, detail="Model did not produce shape functions.")
    all_model_keys = feature_keys + all_dummy_keys
    shape_functions = normalize_numeric_shape_points(shape_functions, all_model_keys, cat_info, num_points)
    def get_shape(key: str) -> Dict:
        return shape_functions.get(key, {})

    active_keys = feature_keys
    contribs_train: List[np.ndarray] = []
    contribs_test: List[np.ndarray] = []
    for key in active_keys:
        shape_fn = get_shape(key)
        contribs = np.array(evaluate_contribs(shape_fn, features_train[key], cat_info.get(key)))
        contribs_train.append(contribs)
        if test_len:
            contribs_test.append(np.array(evaluate_contribs(shape_fn, features_test[key], cat_info.get(key))))

    for spec in active_operation_specs:
        display_key = spec["key"]
        dummy_cols_for_pair = interaction_dummy_cols[display_key]
        pair_train = np.zeros(len(y_train))
        pair_test = np.zeros(test_len) if test_len else np.array([])
        for col_name in dummy_cols_for_pair:
            shape_fn = shape_functions.get(col_name, {})
            pair_train += np.array(evaluate_contribs(shape_fn, x_train_aug[col_name].tolist(), None))
            if test_len:
                pair_test += np.array(evaluate_contribs(shape_fn, x_test_aug[col_name].tolist(), None))
        contribs_train.append(pair_train)
        if test_len:
            contribs_test.append(pair_test)
        features_train[display_key] = pair_train.tolist()
        if test_len:
            features_test[display_key] = pair_test.tolist()

    total_train = np.sum(np.stack(contribs_train, axis=0), axis=0) if contribs_train else np.zeros_like(y_train)
    total_test = np.sum(np.stack(contribs_test, axis=0), axis=0) if contribs_test else np.zeros_like(y_test)
    if task_type == "classification":
        intercept_val = _model_intercept(igann)
        preds_train = _sigmoid(total_train + intercept_val)
        preds_test = _sigmoid(total_test + intercept_val) if len(total_test) else np.array([])
    else:
        intercept_val = float(np.mean(y_train - total_train)) if len(y_train) else 0.0
        preds_train = total_train + intercept_val
        preds_test = total_test + intercept_val if len(total_test) else np.array([])

    train_metrics = _calc_metrics(task_type, y_train, preds_train)
    test_metrics = _calc_metrics(task_type, y_test, preds_test)

    partials = []
    for key in active_keys:
        shape_fn = get_shape(key)
        if key in cat_info:
            categories = cat_info[key]
            mapping = {category: 0.0 for category in categories}
            x_vals = shape_fn.get("x", [])
            y_vals = shape_fn.get("y", [])
            for i, category in enumerate(x_vals):
                if category in mapping:
                    mapping[category] = y_vals[i] if i < len(y_vals) else 0.0
            contribs = [mapping.get(category, 0.0) for category in categories]
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
            partials.append(
                {
                    "key": key,
                    "label": label_map.get(key, key),
                    "gridX": [],
                    "curve": [],
                    "scatterX": features_train[key],
                    "trueSignal": None,
                    "editableX": list(shape_fn.get("x", [])),
                    "editableY": list(shape_fn.get("y", [])),
                }
            )

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
                        x_train_df,
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
                        x_train_df,
                        cat_info,
                        n_grid=15,
                    )
                )

    timestamp = int(time.time() * 1000)
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
            "trainY": y_train.tolist(),
            "testY": y_test.tolist(),
            "categories": cat_info,
            "featureLabels": {key: label_map[key] for key in feature_keys},
            "featureDescriptions": {key: descriptions.get(key, "") for key in feature_keys},
        },
        "version": {
            "versionId": str(timestamp),
            "timestamp": timestamp,
            "source": "train",
            "center_shapes": center_shapes,
            "intercept": intercept_val,
            "trainMetrics": train_metrics,
            "testMetrics": test_metrics,
            "shapes": shapes + interaction_shapes,
        },
    }
