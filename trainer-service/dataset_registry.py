from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from preprocessing import preprocess_bike_hourly, preprocess_mimic4_mean_100_full


@dataclass
class DatasetConfig:
    id: str
    label: str
    summary: str
    task_type: Literal["regression", "classification"]
    # Normalised signature: (seed: int, sample_size: int | None = None) → (X, y, cat_info, labels, interaction_specs)
    preprocessor: Callable[..., Any]
    # Human-readable descriptions served to the frontend; keys match preprocessed column names.
    descriptions: dict[str, str] = field(default_factory=dict)
    # None means "all available features"; set to a list to preselect for a study task.
    default_features: list[str] | None = None
    # Per-dataset overrides for training hyperparameter defaults.
    training_defaults: dict[str, Any] = field(default_factory=dict)


REGISTRY: dict[str, DatasetConfig] = {
    "bike_hourly": DatasetConfig(
        id="bike_hourly",
        label="Bike sharing (hourly)",
        summary="Hourly rentals with weather/seasonality.",
        task_type="regression",
        preprocessor=lambda seed, sample_size=None: preprocess_bike_hourly(seed),
        descriptions={
            "Time of Day":      "Hour of the day when rentals were counted.",
            "Windspeed":        "Normalized wind speed converted to an estimated km/h scale.",
            "Temperature":      "Air temperature converted to an estimated Celsius scale.",
            "Humidity":         "Relative humidity on a 0 to 100 scale.",
            "Weathersituation": "Observed weather condition, from clear to rain.",
            "Type of Day":      "Whether the observation falls on a working day, weekend, or holiday.",
        },
    ),
    "mimic4_mean_100_full": DatasetConfig(
        id="mimic4_mean_100_full",
        label="MIMIC-IV mortality",
        summary="ICU cohort with demographics, length of stay, and mean vital/lab features.",
        task_type="classification",
        preprocessor=preprocess_mimic4_mean_100_full,
        descriptions={
            # Demographics
            "Age":                   "Patient age at ICU admission.",
            "Eth":                   "Patient ethnicity.",
            "Sex":                   "Patient sex.",
            # Stay info
            "LOS":                   "Length of ICU stay in days.",
            # Vitals
            "HR+100%mean":           "Mean heart rate (beats/min).",
            "RR+100%mean":           "Mean respiratory rate (breaths/min).",
            "SBP+100%mean":          "Mean systolic blood pressure (mmHg).",
            "DBP+100%mean":          "Mean diastolic blood pressure (mmHg).",
            "MBP+100%mean":          "Mean mean arterial pressure (mmHg).",
            "Temp+100%mean":         "Mean body temperature (°C).",
            # Anthropometrics
            "Weight+100%mean":       "Mean body weight during ICU stay (kg).",
            "Height+100%mean":       "Mean height during ICU stay (cm).",
            "Bmi+100%mean":          "Mean body mass index during stay.",
            # Neurological
            "GCST+100%mean":         "Mean Glasgow Coma Scale total score (3–15); lower = more impaired.",
            # Respiratory / blood gas
            "FiO2+100%mean":         "Mean fraction of inspired oxygen (0–1).",
            "PaO2+100%mean":         "Mean partial pressure of arterial oxygen (mmHg).",
            "PaCO2+100%mean":        "Mean partial pressure of arterial CO₂ (mmHg).",
            "Ph+100%mean":           "Mean arterial blood pH.",
            "HCO3+100%mean":         "Mean serum bicarbonate (mEq/L) — acid-base balance.",
            # Metabolic / glucose
            "GLU+100%mean":          "Mean blood glucose level (mg/dL).",
            "Lactate+100%mean":      "Mean blood lactate (mmol/L) — tissue perfusion marker.",
            "AnionGAP+100%mean":     "Mean anion gap (mEq/L) — metabolic acidosis indicator.",
            # Electrolytes
            "Kalium+100%mean":       "Mean serum potassium (mEq/L).",
            "Natrium+100%mean":      "Mean serum sodium (mEq/L).",
            # Kidney
            "Kreatinin+100%mean":    "Mean serum creatinine (mg/dL) — kidney function marker.",
            "Urea+100%mean":         "Mean blood urea nitrogen (mg/dL).",
            # Liver
            "Bilirubin+100%mean":    "Mean total bilirubin (mg/dL) — liver function marker.",
            "ALAT+100%mean":         "Mean alanine aminotransferase (U/L) — liver enzyme.",
            "ASAT+100%mean":         "Mean aspartate aminotransferase (U/L) — liver enzyme.",
            "Albumin+100%mean":      "Mean serum albumin (g/dL) — nutritional and hepatic marker.",
            # Haematology
            "Hb+100%mean":           "Mean hemoglobin concentration (g/dL).",
            "Leukocyten+100%mean":   "Mean white blood cell count (10³/μL).",
            "Thrombocyten+100%mean": "Mean platelet count (10³/μL).",
            "Quick+100%mean":        "Mean Quick / prothrombin time (%) — coagulation marker.",
        },
        # Set to a list of feature key strings to preselect specific features for the study.
        # Example: default_features=["Age", "LOS", "HR+100%mean", "GCST+100%mean"]
        default_features=None,
        training_defaults={
            "seed": 3,
            "n_estimators": 100,
            "boost_rate": 0.1,
            "init_reg": 1.0,
            "elm_alpha": 1.0,
            "early_stopping": 50,
            "n_hid": 10,
            "sample_size": 1000,
        },
    ),
}


def get_dataset(dataset_id: str) -> DatasetConfig:
    cfg = REGISTRY.get(dataset_id)
    if cfg is None:
        raise KeyError(dataset_id)
    return cfg
