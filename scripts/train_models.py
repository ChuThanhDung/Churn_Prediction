"""Train, evaluate, and persist the three portfolio dashboard models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data import DEFAULT_DATA_PATH, build_schema, load_raw_data, prepare_training_data
from src.models import MODEL_LABELS, MODEL_NAMES, build_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def evaluate_model(model, X_val, y_val) -> dict[str, float]:
    """Calculate the churn-focused validation metrics used in the dashboard."""
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_val, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, predictions)),
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
        "auc": float(roc_auc_score(y_val, probabilities)),
    }


def train_and_save() -> dict[str, dict[str, float]]:
    """Train validation models, refit all data, and save deployment artifacts."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    predictors, target = prepare_training_data(load_raw_data(DEFAULT_DATA_PATH))
    X_train, X_val, y_train, y_val = train_test_split(
        predictors,
        target,
        test_size=0.4,
        random_state=42,
        stratify=target,
    )

    metrics: dict[str, dict[str, float]] = {}
    for model_name in MODEL_NAMES:
        evaluation_model = build_model(model_name)
        evaluation_model.fit(X_train, y_train)
        metrics[model_name] = {
            "display_name": MODEL_LABELS[model_name],
            **evaluate_model(evaluation_model, X_val, y_val),
        }

        final_model = build_model(model_name)
        final_model.fit(predictors, target)
        joblib.dump(final_model, ARTIFACT_DIR / f"{model_name}.joblib", compress=3)
        print(f"Saved {MODEL_LABELS[model_name]}")

    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "schema.json").write_text(
        json.dumps(build_schema(), indent=2),
        encoding="utf-8",
    )
    return metrics


if __name__ == "__main__":
    saved_metrics = train_and_save()
    print(json.dumps(saved_metrics, indent=2))
