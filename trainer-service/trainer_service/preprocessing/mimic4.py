from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from trainer_service.paths import DATA_DIR
from trainer_service.preprocessing.common import sort_category_values


# ── Winsorizer ───────────────────────────────────────────────────────────────
# Clips each numeric column to [mean − n_sigma·σ, mean + n_sigma·σ] fitted on
# training data.  Mirrors the feature_engine Winsorizer used in the notebook
# (capping_method="gaussian", fold=4, tail="both") without adding a dependency.

class GaussianWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_sigma: float = 4.0):
        self.n_sigma = n_sigma

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_ = arr.mean(axis=0) - self.n_sigma * arr.std(axis=0)
        self.upper_ = arr.mean(axis=0) + self.n_sigma * arr.std(axis=0)
        return self

    def transform(self, X):
        clipped = np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(clipped, index=X.index, columns=X.columns)
        return clipped

    def set_output(self, *, transform=None):
        return self


# ── Dataset constants ────────────────────────────────────────────────────────

MIMIC4_TARGET = "mortality"
MIMIC4_SAMPLE_SIZE = 1000  # default rows drawn when no sample_size is passed

# Only these two columns are treated as categorical; everything else is numeric.
MIMIC4_CATEGORICAL_FEATURES = ["Eth", "Sex"]


# ── Feature catalog ──────────────────────────────────────────────────────────
# Edit descriptions here — they are served directly to the frontend.
# Keys must match the column names in mimic4_mean_100_full.csv after preprocessing.

MIMIC4_DESCRIPTIONS: dict[str, str] = {
    # Demographics
    "Age":              "Patient age at ICU admission.",
    "Eth":              "Patient ethnicity.",
    "Sex":              "Patient sex.",
    # Stay info
    "LOS":              "Length of ICU stay in days.",
    # Vitals
    "HR+100%mean":      "Mean heart rate (beats/min).",
    "RR+100%mean":      "Mean respiratory rate (breaths/min).",
    "SBP+100%mean":     "Mean systolic blood pressure (mmHg).",
    "DBP+100%mean":     "Mean diastolic blood pressure (mmHg).",
    "MBP+100%mean":     "Mean mean arterial pressure (mmHg).",
    "Temp+100%mean":    "Mean body temperature (°C).",
    # Anthropometrics
    "Weight+100%mean":  "Mean body weight during ICU stay (kg).",
    "Height+100%mean":  "Mean height during ICU stay (cm).",
    "Bmi+100%mean":     "Mean body mass index during stay.",
    # Neurological
    "GCST+100%mean":    "Mean Glasgow Coma Scale total score (3–15); lower = more impaired.",
    # Respiratory / blood gas
    "FiO2+100%mean":    "Mean fraction of inspired oxygen (0–1).",
    "PaO2+100%mean":    "Mean partial pressure of arterial oxygen (mmHg).",
    "PaCO2+100%mean":   "Mean partial pressure of arterial CO₂ (mmHg).",
    "Ph+100%mean":      "Mean arterial blood pH.",
    "HCO3+100%mean":    "Mean serum bicarbonate (mEq/L) — acid-base balance.",
    # Metabolic / glucose
    "GLU+100%mean":     "Mean blood glucose level (mg/dL).",
    "Lactate+100%mean": "Mean blood lactate (mmol/L) — tissue perfusion marker.",
    "AnionGAP+100%mean":"Mean anion gap (mEq/L) — metabolic acidosis indicator.",
    # Electrolytes
    "Kalium+100%mean":  "Mean serum potassium (mEq/L).",
    "Natrium+100%mean": "Mean serum sodium (mEq/L).",
    # Kidney
    "Kreatinin+100%mean":"Mean serum creatinine (mg/dL) — kidney function marker.",
    "Urea+100%mean":    "Mean blood urea nitrogen (mg/dL).",
    # Liver
    "Bilirubin+100%mean":"Mean total bilirubin (mg/dL) — liver function marker.",
    "ALAT+100%mean":    "Mean alanine aminotransferase (U/L) — liver enzyme.",
    "ASAT+100%mean":    "Mean aspartate aminotransferase (U/L) — liver enzyme.",
    "Albumin+100%mean": "Mean serum albumin (g/dL) — nutritional and hepatic marker.",
    # Haematology
    "Hb+100%mean":      "Mean hemoglobin concentration (g/dL).",
    "Leukocyten+100%mean":"Mean white blood cell count (10³/μL).",
    "Thrombocyten+100%mean":"Mean platelet count (10³/μL).",
    "Quick+100%mean":   "Mean Quick / prothrombin time (%) — coagulation marker.",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sample_balanced_by_target(df: pd.DataFrame, target: str, sample_size: int, seed: int) -> pd.DataFrame:
    """Draw `sample_size` rows with equal representation per target class."""
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


# ── Public preprocessing entry point ─────────────────────────────────────────

def preprocess_mimic4_mean_100_full(seed: int, sample_size: int | None = None):
    """Load and preprocess the MIMIC-IV mean-imputed export.

    Returns (X, y, cat_info, labels, descriptions) where X is a DataFrame of
    features, y is the binary mortality target as a numpy array, cat_info maps
    categorical column names to their sorted category lists, labels maps each
    column name to its display name, and descriptions maps column names to
    human-readable explanations for the frontend.
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    data_path = Path("/data/mimic4_mean_100_full.csv")
    if not data_path.exists():
        data_path = DATA_DIR / "mimic4_mean_100_full.csv"
    if not data_path.exists():
        raise FileNotFoundError("Missing mimic4_mean_100_full.csv in trainer-service/data.")

    df = pd.read_csv(data_path)

    # ── 2. Clean target ───────────────────────────────────────────────────────
    # Replace common blank/placeholder values before parsing the target column.
    df.replace({"": np.nan, " ": np.nan, "-": np.nan}, inplace=True)
    df.dropna(subset=[MIMIC4_TARGET], inplace=True)
    y = pd.to_numeric(df[MIMIC4_TARGET], errors="coerce")
    keep_mask = y.notna()
    df = df.loc[keep_mask].copy()
    y = y.loc[keep_mask].astype(float)
    df[MIMIC4_TARGET] = y

    # ── 3. Sample ─────────────────────────────────────────────────────────────
    # Subsample to keep training time acceptable; balance classes so the model
    # sees enough positive (mortality=1) examples.
    effective_sample_size = sample_size if sample_size is not None else MIMIC4_SAMPLE_SIZE
    if len(df) > effective_sample_size:
        df = _sample_balanced_by_target(df, MIMIC4_TARGET, effective_sample_size, seed)
        y = df[MIMIC4_TARGET].astype(float)

    # ── 4. Split features ─────────────────────────────────────────────────────
    x_frame = df.drop(columns=[MIMIC4_TARGET], errors="ignore")
    cat_features = [f for f in MIMIC4_CATEGORICAL_FEATURES if f in x_frame.columns]
    num_features = [f for f in x_frame.columns if f not in cat_features]

    # Coerce dtypes so the imputer receives the right input types.
    for feature in num_features:
        x_frame[feature] = pd.to_numeric(x_frame[feature], errors="coerce")
    for feature in cat_features:
        x_frame[feature] = x_frame[feature].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})

    # ── 5. Impute → Winsorize ─────────────────────────────────────────────────
    # Numeric:     median imputation (robust to outliers common in ICU data),
    #              then Gaussian winsorization at ±4 σ (matches notebook pipeline).
    # Categorical: most-frequent imputation.
    num_transformer = Pipeline([
        ("num_imputer",  SimpleImputer(strategy="median")),
        ("winsorizer",   GaussianWinsorizer(n_sigma=4.0)),
    ])
    cat_transformer = Pipeline([("cat_imputer", SimpleImputer(strategy="most_frequent"))])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    x_processed = column_transformer.fit_transform(x_frame)

    # Restore object dtype for categorical columns so downstream code can
    # detect them via dtype rather than relying on the cat_info dict.
    cast_map = {col: "object" for col in cat_features if col in x_processed.columns}
    if cast_map:
        x_processed = x_processed.astype(cast_map)

    # ── 6. Build metadata ─────────────────────────────────────────────────────
    cat_info = {
        col: sort_category_values(x_processed[col].dropna().unique().tolist())
        for col in cat_features
        if col in x_processed.columns
    }
    labels = {col: col for col in x_processed.columns}

    return x_processed, y.to_numpy(), cat_info, labels, MIMIC4_DESCRIPTIONS, []
