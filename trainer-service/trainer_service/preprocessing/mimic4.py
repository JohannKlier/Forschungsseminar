from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from trainer_service.paths import DATA_DIR
from trainer_service.preprocessing.common import sort_category_values


MIMIC4_CATEGORICAL_FEATURES = ["Eth", "Sex"]
MIMIC4_SAMPLE_SIZE = 1000
MIMIC4_TARGET = "mortality"


def _sample_balanced_by_target(df: pd.DataFrame, target: str, sample_size: int, seed: int) -> pd.DataFrame:
    groups = [group for _, group in df.groupby(target, dropna=False)]
    if len(groups) < 2 or len(df) <= sample_size:
        return df

    per_group = sample_size // len(groups)
    remainder = sample_size % len(groups)
    sampled_parts = []
    for index, group in enumerate(groups):
        target_size = per_group + (1 if index < remainder else 0)
        sampled_parts.append(group.sample(n=min(target_size, len(group)), random_state=seed))

    return pd.concat(sampled_parts, axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)


def preprocess_mimic4_mean_100_full(seed: int):
    """Minimal preprocessing for the local MIMIC-IV mean-imputed feature export."""
    data_path = Path("/data/mimic4_mean_100_full.csv")
    if not data_path.exists():
        data_path = DATA_DIR / "mimic4_mean_100_full.csv"
    if not data_path.exists():
        raise FileNotFoundError("Missing mimic4_mean_100_full.csv in trainer-service/data.")

    df = pd.read_csv(data_path)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()
    df = df.loc[:, df.columns.astype(str).str.strip() != ""]
    if MIMIC4_TARGET not in df.columns:
        raise ValueError("mimic4_mean_100_full.csv missing mortality column.")

    df.replace({"": np.nan, " ": np.nan, "-": np.nan}, inplace=True)
    df.dropna(subset=[MIMIC4_TARGET], inplace=True)
    y = pd.to_numeric(df[MIMIC4_TARGET], errors="coerce")
    keep_mask = y.notna()
    df = df.loc[keep_mask].copy()
    y = y.loc[keep_mask].astype(float)
    df[MIMIC4_TARGET] = y

    if len(df) > MIMIC4_SAMPLE_SIZE:
        df = _sample_balanced_by_target(df, MIMIC4_TARGET, MIMIC4_SAMPLE_SIZE, seed)
        y = df[MIMIC4_TARGET].astype(float)

    x_frame = df.drop(columns=[MIMIC4_TARGET], errors="ignore")
    cat_features = [feature for feature in MIMIC4_CATEGORICAL_FEATURES if feature in x_frame.columns]
    num_features = [feature for feature in x_frame.columns if feature not in cat_features]

    for feature in num_features:
        x_frame[feature] = pd.to_numeric(x_frame[feature], errors="coerce")
    for feature in cat_features:
        x_frame[feature] = x_frame[feature].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})

    num_transformer = Pipeline([("num_imputer", SimpleImputer(strategy="median"))])
    cat_transformer = Pipeline([("cat_imputer", SimpleImputer(strategy="most_frequent"))])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    x_processed = column_transformer.fit_transform(x_frame)
    cast_map = {col: "object" for col in cat_features if col in x_processed.columns}
    if cast_map:
        x_processed = x_processed.astype(cast_map)

    cat_info = {
        col: sort_category_values(x_processed[col].dropna().unique().tolist())
        for col in cat_features
        if col in x_processed.columns
    }
    labels = {col: col for col in x_processed.columns}
    return x_processed, y.to_numpy(), cat_info, labels
