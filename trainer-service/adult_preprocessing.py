from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def _load_adult(path: Path, skip_rows: int = 0) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        names=ADULT_COLUMNS,
        header=None,
        skiprows=skip_rows,
        skipinitialspace=True,
        comment="|",
    )
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df["income"] = df["income"].astype(str).str.strip().str.replace(".", "", regex=False)
    return df


def preprocess_adult_income(seed: int):
    """Preprocess the UCI Adult income dataset."""
    data_root = Path(__file__).parent / "data"
    train_path = data_root / "adult.data"
    test_path = data_root / "adult.test"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing adult.data or adult.test in trainer-service/data.")

    train_df = _load_adult(train_path)
    test_df = _load_adult(test_path, skip_rows=1)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    y = df["income"].apply(lambda v: 1.0 if str(v).startswith(">") else 0.0)
    X = df.drop(columns=["income"])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    num_features = [feature for feature in X.columns if feature not in cat_features]

    num_transformer = Pipeline([("num_imputer", SimpleImputer(strategy="median"))])
    cat_transformer = Pipeline([("cat_imputer", SimpleImputer(strategy="most_frequent"))])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    X_proc = column_transformer.fit_transform(X)
    cast_map = {col: "object" for col in cat_features if col in X_proc.columns}
    if cast_map:
        X_proc = X_proc.astype(cast_map)

    cat_info = {
        col: sorted([str(c) for c in X_proc[col].dropna().unique().tolist()]) for col in cat_features if col in X_proc.columns
    }
    labels = {col: col for col in X_proc.columns}
    return X_proc, y.to_numpy(), cat_info, labels
