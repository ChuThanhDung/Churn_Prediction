"""Dataset loading, cleaning, and prediction input validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "telco_customer_churn.csv"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
PREDICTOR_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    column for column in PREDICTOR_COLUMNS if column not in NUMERIC_COLUMNS
]

CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


class InputValidationError(ValueError):
    """Raised when a prediction input cannot be safely scored."""


def load_raw_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the IBM Telco Customer Churn CSV."""
    return pd.read_csv(path)


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges and remove rows where that value is unavailable."""
    cleaned = frame.copy()
    if "TotalCharges" not in cleaned.columns:
        raise InputValidationError("Missing required column: TotalCharges")

    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned = cleaned.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    return cleaned


def prepare_training_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return predictors and encoded churn target from a raw dataset."""
    cleaned = clean_data(frame)
    missing = [column for column in PREDICTOR_COLUMNS + [TARGET_COLUMN] if column not in cleaned]
    if missing:
        raise InputValidationError(f"Missing required columns: {', '.join(missing)}")

    predictors = cleaned[PREDICTOR_COLUMNS].copy()
    target = cleaned[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise InputValidationError("Churn must contain only 'No' or 'Yes'.")
    return predictors, target.astype(int)


def validate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize one or more rows before inference."""
    missing = [column for column in PREDICTOR_COLUMNS if column not in frame.columns]
    if missing:
        raise InputValidationError(f"Missing required columns: {', '.join(missing)}")

    validated = frame.copy()
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(validated[column], errors="coerce")
        invalid = converted.isna() & validated[column].notna()
        if invalid.any() or converted.isna().any():
            raise InputValidationError(f"Column '{column}' must contain numeric values.")
        validated[column] = converted

    if ((validated["SeniorCitizen"] < 0) | (validated["SeniorCitizen"] > 1)).any():
        raise InputValidationError("SeniorCitizen must be 0 or 1.")
    if (validated["tenure"] < 0).any():
        raise InputValidationError("tenure cannot be negative.")
    if (validated[["MonthlyCharges", "TotalCharges"]] < 0).any().any():
        raise InputValidationError("Charge values cannot be negative.")

    for column, allowed_values in CATEGORY_OPTIONS.items():
        invalid_values = sorted(
            set(validated[column].dropna().astype(str)) - set(allowed_values)
        )
        if invalid_values:
            values = ", ".join(invalid_values)
            raise InputValidationError(f"Column '{column}' contains invalid values: {values}")
        if validated[column].isna().any():
            raise InputValidationError(f"Column '{column}' cannot contain blank values.")

    return validated


def build_schema() -> dict[str, Any]:
    """Return the public input contract persisted alongside model artifacts."""
    return {
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "predictor_columns": PREDICTOR_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "category_options": CATEGORY_OPTIONS,
        "optional_batch_columns": [ID_COLUMN, TARGET_COLUMN],
    }
