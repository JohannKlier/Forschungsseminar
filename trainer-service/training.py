from __future__ import annotations

import time
from typing import Dict, List

from fastapi import HTTPException
from igann import IGANN, IGANN_interactive
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from preprocessing import (
    preprocess_bike_hourly,
    preprocess_mimic4_mean_100_full,
)
from schemas import TrainRequest



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
    if not xs or not ys:
        return [0.0 for _ in feat_values]
    return np.interp(feat_values, xs, ys).tolist()


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
        target_y = np.interp(target_x, xs_sorted, ys_sorted).tolist()

        normalized[key] = {
            **shape_fn,
            "datatype": "numerical",
            "x": target_x,
            "y": target_y,
        }
    return normalized


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

    if is_cat1 and is_cat2:
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
    x_processed, _y_full, cat_info, labels, descriptions, interaction_specs = _load_dataset(request)
    dummy_keys = {col for spec in interaction_specs for col in spec["dummy_cols"]}
    features = []

    for key in (col for col in x_processed.columns if col not in dummy_keys):
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
        return {
            "acc": float(accuracy_score(y_true >= 0.5, y_pred >= 0.5)),
            "count": int(len(y_true)),
        }
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "count": int(len(y_true)),
    }


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
    elm_alpha = max(0.0, min(10.0, request.elm_alpha))
    early_stopping = max(5, min(200, request.early_stopping))
    n_hid = max(1, min(100, request.n_hid))

    classification_datasets = {"mimic4_mean_100_full"}
    task_type = "classification" if request.dataset in classification_datasets else "regression"
    igann_task = task_type

    x_processed, y_full, cat_info, labels, descriptions, interaction_specs = _load_dataset(request)

    all_dummy_keys_set = {col for spec in interaction_specs for col in spec["dummy_cols"]}

    requested_features = [str(feature) for feature in (request.selected_features or [])]
    if requested_features:
        requested_set = set(requested_features)
        interaction_specs = [spec for spec in interaction_specs if all(s in requested_set for s in spec["sources"])]
        all_dummy_keys_set = {col for spec in interaction_specs for col in spec["dummy_cols"]}
        x_processed = x_processed.loc[:, requested_features + sorted(all_dummy_keys_set)].copy()
        cat_info = {k: v for k, v in cat_info.items() if k in requested_set}
        labels = {k: labels.get(k, k) for k in requested_features}

    feature_keys = [col for col in x_processed.columns if col not in all_dummy_keys_set]
    interaction_dummy_cols = {spec["key"]: spec["dummy_cols"] for spec in interaction_specs}
    all_dummy_keys = [col for spec in interaction_specs for col in spec["dummy_cols"]]

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

    model_cls = IGANN_interactive if model_type == "igann_interactive" else IGANN
    model_kwargs = dict(
        task=igann_task,
        n_estimators=n_estimators,
        boost_rate=boost_rate,
        init_reg=init_reg,
        elm_alpha=elm_alpha,
        early_stopping=early_stopping,
        n_hid=n_hid,
        device="cpu",
        random_state=request.seed,
        verbose=0,
        scale_y=use_scale_y,
    )
    if model_type == "igann_interactive":
        model_kwargs["GAM_detail"] = num_points
    igann = model_cls(**model_kwargs)

    igann.fit(x_train_df, y_train)

    if center_shapes and model_type == "igann_interactive" and hasattr(igann, "center_shape_functions"):
        igann.center_shape_functions(x_train_df, update_intercept=True)

    label_map = dict(labels)
    for spec in interaction_specs:
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

    contribs_train: List[np.ndarray] = []
    contribs_test: List[np.ndarray] = []
    for key in feature_keys:
        shape_fn = get_shape(key)
        contribs = np.array(evaluate_contribs(shape_fn, features_train[key], cat_info.get(key)))
        contribs_train.append(contribs)
        if test_len:
            contribs_test.append(np.array(evaluate_contribs(shape_fn, features_test[key], cat_info.get(key))))

    for spec in interaction_specs:
        display_key = spec["key"]
        dummy_cols_for_pair = interaction_dummy_cols[display_key]
        pair_train = np.zeros(len(y_train))
        pair_test = np.zeros(test_len) if test_len else np.array([])
        for col_name in dummy_cols_for_pair:
            shape_fn = shape_functions.get(col_name, {})
            pair_train += np.array(evaluate_contribs(shape_fn, x_train_df[col_name].tolist(), None))
            if test_len:
                pair_test += np.array(evaluate_contribs(shape_fn, x_test_df[col_name].tolist(), None))
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

    shapes = []
    for key in feature_keys:
        shape_fn = get_shape(key)
        shape: Dict = {"key": key, "label": label_map.get(key, key)}
        if key in cat_info:
            categories = cat_info[key]
            mapping = {category: 0.0 for category in categories}
            x_vals = shape_fn.get("x", [])
            y_vals = shape_fn.get("y", [])
            for i, category in enumerate(x_vals):
                if category in mapping:
                    mapping[category] = y_vals[i] if i < len(y_vals) else 0.0
            shape["categories"] = categories
            shape["editableX"] = list(range(len(categories)))
            shape["editableY"] = [mapping.get(category, 0.0) for category in categories]
        else:
            shape["editableX"] = list(shape_fn.get("x", []))
            shape["editableY"] = list(shape_fn.get("y", []))
        shapes.append(shape)

    interaction_shapes = []
    for spec in interaction_specs:
        display_key = spec["key"]
        dummy_cols_for_pair = interaction_dummy_cols[display_key]
        dummy_shapes_for_pair = {col: shape_functions.get(col, {}) for col in dummy_cols_for_pair}
        if any(dummy_shapes_for_pair.values()):
            if spec["operator"] == "product" and (spec["sources"][0] in cat_info or spec["sources"][1] in cat_info):
                interaction_shapes.append(_build_2d_grid_from_dummies(
                    spec["sources"][0], spec["sources"][1],
                    dummy_cols_for_pair, dummy_shapes_for_pair,
                    x_train_df, cat_info, label_map, n_grid=15,
                ))
            else:
                interaction_shapes.append(_build_2d_grid_for_operation(
                    spec, dummy_shapes_for_pair.get(display_key, {}),
                    x_train_df, cat_info, n_grid=15,
                ))

    timestamp = int(time.time() * 1000)
    return {
        "model": {
            "dataset": request.dataset,
            "model_type": model_type,
            "task": task_type,
            "selected_features": feature_keys,
            "selected_interactions": [spec["key"] for spec in interaction_specs if spec["operator"] == "product"],
            "selected_operations": interaction_specs,
            "seed": request.seed,
            "n_estimators": n_estimators,
            "boost_rate": boost_rate,
            "init_reg": init_reg,
            "elm_alpha": elm_alpha,
            "early_stopping": early_stopping,
            "n_hid": n_hid,
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
