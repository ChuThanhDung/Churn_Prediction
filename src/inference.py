"""Artifact loading and prediction helpers for Streamlit and tests."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.data import PREDICTOR_COLUMNS, validate_prediction_frame
from src.models import DEFAULT_MODEL, MODEL_LABELS, MODEL_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@lru_cache(maxsize=len(MODEL_NAMES))
def load_model(model_name: str = DEFAULT_MODEL):
    """Load and cache a trained pipeline."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}")
    artifact_path = ARTIFACT_DIR / f"{model_name}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing artifact: {artifact_path}. Run: python -m scripts.train_models"
        )
    return joblib.load(artifact_path)


def risk_level(probability: float) -> str:
    """Map a churn probability to a presentation-friendly risk level."""
    if probability < 0.35:
        return "Low"
    if probability < 0.65:
        return "Medium"
    return "High"


def predict_batch(
    frame: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Score one or more customers and append prediction columns."""
    validated = validate_prediction_frame(frame)
    model = load_model(model_name)
    predictions = model.predict(validated[PREDICTOR_COLUMNS])
    probabilities = model.predict_proba(validated[PREDICTOR_COLUMNS])[:, 1]

    result = frame.copy()
    result["prediction"] = ["Yes" if prediction == 1 else "No" for prediction in predictions]
    result["churn_probability"] = probabilities.round(6)
    result["risk_level"] = [risk_level(float(value)) for value in probabilities]
    result["model"] = MODEL_LABELS[model_name]
    return result


def predict_customer(
    customer: dict[str, object],
    model_name: str = DEFAULT_MODEL,
) -> dict[str, object]:
    """Score one customer represented by the Streamlit form."""
    prediction = predict_batch(pd.DataFrame([customer]), model_name).iloc[0]
    return {
        "prediction": prediction["prediction"],
        "churn_probability": float(prediction["churn_probability"]),
        "risk_level": prediction["risk_level"],
        "model": prediction["model"],
    }
