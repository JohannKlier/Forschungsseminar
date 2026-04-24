from __future__ import annotations

import numpy as np


def sort_category_values(values):
    def sort_key(value):
        as_string = str(value)
        try:
            return (0, float(as_string), as_string)
        except ValueError:
            return (1, as_string.lower(), as_string)

    return sorted([str(value) for value in values], key=sort_key)


def _get_numeric_col(key: str, df, cat_info: dict) -> np.ndarray:
    if key in cat_info:
        categories = cat_info[key]
        cat_to_idx = {str(c): float(i) for i, c in enumerate(categories)}
        return np.array([cat_to_idx.get(str(v), 0.0) for v in df[key]], dtype=float)
    return df[key].values.astype(float)


def build_interaction_cols(df, k1: str, k2: str, operator: str, cat_info: dict) -> list:
    """Compute interaction feature columns for a given operator.

    Returns a list of (col_name, values) tuples. Product interactions involving
    categorical features produce one dummy column per category level.
    """
    display_key = f"{k1}__{k2}" if operator == "product" else f"{k1}__{operator}__{k2}"
    is_cat1 = k1 in cat_info
    is_cat2 = k2 in cat_info

    if operator == "product":
        if not is_cat1 and not is_cat2:
            return [(display_key, df[k1].values.astype(float) * df[k2].values.astype(float))]
        if is_cat1 and not is_cat2:
            num_vals = df[k2].values.astype(float)
            return [
                (f"{display_key}___c{i}", (df[k1].astype(str) == str(level)).values.astype(float) * num_vals)
                for i, level in enumerate(cat_info[k1])
            ]
        if not is_cat1 and is_cat2:
            num_vals = df[k1].values.astype(float)
            return [
                (f"{display_key}___r{j}", (df[k2].astype(str) == str(level)).values.astype(float) * num_vals)
                for j, level in enumerate(cat_info[k2])
            ]
        v1 = _get_numeric_col(k1, df, cat_info)
        v2 = _get_numeric_col(k2, df, cat_info)
        return [(display_key, v1 * v2)]

    left_vals = df[k1].values.astype(float)
    right_vals = df[k2].values.astype(float)
    if operator == "sum":
        vals = left_vals + right_vals
    elif operator == "difference":
        vals = left_vals - right_vals
    elif operator == "ratio":
        vals = left_vals / np.where(np.abs(right_vals) < 1e-9, 1e-9, right_vals)
    elif operator == "absolute_difference":
        vals = np.abs(left_vals - right_vals)
    else:
        raise ValueError(f"Unsupported interaction operator: {operator}")
    return [(display_key, vals)]


_OPERATOR_SYMBOLS = {
    "product": "×",
    "sum": "+",
    "difference": "−",
    "ratio": "/",
    "absolute_difference": "|Δ|",
}


def make_interaction_spec(df, k1: str, k2: str, operator: str, cat_info: dict,
                          label_k1: str = None, label_k2: str = None) -> tuple:
    """Build interaction columns and a spec dict for a feature pair.

    Returns (spec, columns) where columns is a dict {col_name: np.ndarray}.
    Add the columns to x_processed and include spec in interaction_specs before
    returning from the preprocessor.

    spec keys: key, label, sources, operator, dummy_cols
    """
    label_k1 = label_k1 or k1
    label_k2 = label_k2 or k2
    sym = _OPERATOR_SYMBOLS.get(operator, operator)
    key = f"{k1}__{k2}" if operator == "product" else f"{k1}__{operator}__{k2}"

    cols = build_interaction_cols(df, k1, k2, operator, cat_info)
    spec = {
        "key": key,
        "label": f"{label_k1} {sym} {label_k2}",
        "sources": [k1, k2],
        "operator": operator,
        "dummy_cols": [name for name, _ in cols],
    }
    columns = {name: vals for name, vals in cols}
    return spec, columns
