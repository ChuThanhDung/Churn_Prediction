"""Dataset cleaning and schema validation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import (
    PREDICTOR_COLUMNS,
    InputValidationError,
    clean_data,
    load_raw_data,
    prepare_training_data,
    validate_prediction_frame,
)


def test_clean_data_removes_blank_total_charges() -> None:
    cleaned = clean_data(load_raw_data())
    assert len(cleaned) == 7032
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_prepare_training_data_returns_19_predictors() -> None:
    predictors, target = prepare_training_data(load_raw_data())
    assert predictors.columns.tolist() == PREDICTOR_COLUMNS
    assert predictors.shape == (7032, 19)
    assert set(target.unique()) == {0, 1}


def test_validation_reports_missing_columns(sample_customer) -> None:
    sample_customer.pop("Contract")
    with pytest.raises(InputValidationError, match="Missing required columns: Contract"):
        validate_prediction_frame(pd.DataFrame([sample_customer]))


def test_validation_reports_invalid_numeric_value(sample_customer) -> None:
    sample_customer["tenure"] = "not-a-number"
    with pytest.raises(InputValidationError, match="tenure.*numeric"):
        validate_prediction_frame(pd.DataFrame([sample_customer]))
