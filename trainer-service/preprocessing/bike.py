from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from paths import DATA_DIR
from .common import sort_category_values


BIKE_DESCRIPTIONS: dict[str, str] = {
    "Time of Day": "Hour of the day when rentals were counted.",
    "Windspeed": "Normalized wind speed converted to an estimated km/h scale.",
    "Temperature": "Air temperature converted to an estimated Celsius scale.",
    "Humidity": "Relative humidity on a 0 to 100 scale.",
    "Weathersituation": "Observed weather condition, from clear to rain.",
    "Type of Day": "Whether the observation falls on a working day, weekend, or holiday.",
}


def preprocess_bike_hourly(seed: int):
    """Replicate preprocessing from the original trainer notebook."""
    del seed
    bike_path = DATA_DIR / "bike.csv"
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

    x_frame = df.drop(columns=feature_to_drop, errors="ignore")

    cat_features = ["Weathersituation", "Time of Day", "Type of Day"]
    num_features = [feature for feature in x_frame.columns if feature not in cat_features]

    num_transformer = Pipeline([("num_imputer", SimpleImputer(strategy="mean"))])
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
    return x_processed, y.to_numpy(), cat_info, labels, BIKE_DESCRIPTIONS, []
