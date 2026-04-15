from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from trainer_service.paths import DATA_DIR


def preprocess_breast_cancer():
    """Preprocess the Breast Cancer Wisconsin dataset."""
    data_path = Path("/data/breastCancer.csv")
    if not data_path.exists():
        data_path = DATA_DIR / "breastCancer.csv"
    if not data_path.exists():
        raise FileNotFoundError("Missing breastCancer.csv in trainer-service/data.")

    df = pd.read_csv(data_path)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()
    df = df.loc[:, df.columns.astype(str).str.strip() != ""]
    if "diagnosis" not in df.columns:
        raise ValueError("breastCancer.csv missing diagnosis column.")

    y = df["diagnosis"].astype(str).str.strip().map(lambda value: 1.0 if value.upper() == "M" else 0.0)
    x_frame = df.drop(columns=["diagnosis", "id"], errors="ignore")

    num_transformer = Pipeline([("num_imputer", SimpleImputer(strategy="median"))])
    x_processed = num_transformer.fit_transform(x_frame)
    x_processed = pd.DataFrame(x_processed, columns=x_frame.columns)

    cat_info = {}
    labels = {col: col for col in x_frame.columns}
    return x_processed, y.to_numpy(), cat_info, labels
