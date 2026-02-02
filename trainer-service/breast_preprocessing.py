from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def preprocess_breast_cancer():
    """Preprocess the Breast Cancer Wisconsin dataset."""
    data_path = Path("/data/breastCancer.csv")
    if not data_path.exists():
        data_path = Path(__file__).parent / "data" / "breastCancer.csv"
    if not data_path.exists():
        raise FileNotFoundError("Missing breastCancer.csv in trainer-service/data.")

    df = pd.read_csv(data_path)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()
    df = df.loc[:, df.columns.astype(str).str.strip() != ""]
    if "diagnosis" not in df.columns:
        raise ValueError("breastCancer.csv missing diagnosis column.")

    y = df["diagnosis"].astype(str).str.strip().map(lambda v: 1.0 if v.upper() == "M" else 0.0)
    X = df.drop(columns=["diagnosis", "id"], errors="ignore")

    num_transformer = Pipeline([("num_imputer", SimpleImputer(strategy="median"))])
    X_proc = num_transformer.fit_transform(X)
    X_proc = pd.DataFrame(X_proc, columns=X.columns)

    cat_info = {}
    labels = {col: col for col in X.columns}
    return X_proc, y.to_numpy(), cat_info, labels
