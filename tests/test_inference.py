"""Artifact loading and prediction behavior tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.inference import load_model, predict_batch, predict_customer
from src.models import MODEL_NAMES


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_artifacts_load_and_predict_probability(model_name, sample_customer) -> None:
    model = load_model(model_name)
    probability = model.predict_proba(pd.DataFrame([sample_customer]))[0, 1]
    assert 0 <= probability <= 1


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_single_customer_prediction(model_name, sample_customer) -> None:
    prediction = predict_customer(sample_customer, model_name)
    assert prediction["prediction"] in {"Yes", "No"}
    assert 0 <= prediction["churn_probability"] <= 1
    assert prediction["risk_level"] in {"Low", "Medium", "High"}


def test_batch_prediction_preserves_customer_id(sample_customer) -> None:
    sample_customer["customerID"] = "TEST-0001"
    result = predict_batch(pd.DataFrame([sample_customer]))
    assert result.loc[0, "customerID"] == "TEST-0001"
    assert {"prediction", "churn_probability", "risk_level", "model"} <= set(result)
