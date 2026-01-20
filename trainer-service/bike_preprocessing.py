from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def preprocess_bike_hourly(seed: int):
    """Replicate preprocessing from test.ipynb for bike.csv."""
    bike_path = Path("/data/bike.csv")
    if not bike_path.exists():
        bike_path = Path(__file__).parent / "data" / "bike.csv"
    df = pd.read_csv(bike_path)

    def scale_values(values, new_min, new_max):
        arr = np.array(values)
        old_min, old_max = float(np.min(arr)), float(np.max(arr))
        denom = old_max - old_min if old_max != old_min else 1e-9
        return (arr - old_min) / denom * (new_max - new_min) + new_min

    df["Time of Day"] = df["hr"]
    df["Windspeed"] = scale_values(df["windspeed"], 0, 67)
    df["Temperature"] = scale_values(df["temp"], -8, 39)
    df["Perceived Temperature"] = scale_values(df["atemp"], -16, 50)
    df["Humidity"] = scale_values(df["hum"], 0, 100)

    df["Season"] = df["season"].replace({1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"})
    df["Weathersituation"] = df["weathersit"].replace({1: "Clear", 2: "Cloudy", 3: "Light Rain", 4: "Heavy Rain"})
    df["Type of Day"] = np.where(
        (df["workingday"] == 1) & (df["holiday"] == 0),
        "Working Day",
        np.where((df["workingday"] == 0) & (df["holiday"] == 0), "Weekend", "Holiday"),
    )

    df.dropna(subset=["cnt"], inplace=True)
    df.replace("-", np.nan, inplace=True)
    df.dropna(inplace=True)

    y = df["cnt"].astype(float)

    feature_to_drop = [
        "dteday",
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "cnt",
        "instant",
        "workingday",
        "casual",
        "registered",
        "weekday",
        "Perceived Temperature",
        "Season",
    ]

    X = df.drop(columns=feature_to_drop, errors="ignore")

    cat_features = ["Weathersituation", "Time of Day", "Type of Day"]
    num_features = [feature for feature in X.columns if feature not in cat_features]

    num_transformer = Pipeline([("num_imputer", SimpleImputer(strategy="mean"))])
    cat_transformer = Pipeline([("cat_imputer", SimpleImputer(strategy="most_frequent"))])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    X_proc = column_transformer.fit_transform(X)
    # Ensure categorical columns are objects
    cast_map = {col: "object" for col in cat_features if col in X_proc.columns}
    if cast_map:
        X_proc = X_proc.astype(cast_map)

    cat_info = {
        col: sorted([str(c) for c in X_proc[col].dropna().unique().tolist()]) for col in cat_features if col in X_proc.columns
    }
    labels = {col: col for col in X_proc.columns}
    return X_proc, y.to_numpy(), cat_info, labels
