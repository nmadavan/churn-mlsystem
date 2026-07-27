"""Tests for the shared preprocessing module.

Focus areas:
  - the tenure=0 / blank-TotalCharges edge case is handled with no NaN or inf
  - engineered features compute the values I expect
  - a single serving-style row transforms to the same width as training (no skew)
"""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    ADDON_SERVICES,
    build_preprocessor,
    clean_and_engineer,
)


def _base_row(**overrides):
    """A minimal valid raw record; override individual fields per test."""
    row = {
        "customerID": "TEST-0001",
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 50.0, "TotalCharges": "600.0",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_new_customer_blank_total_charges_no_nan_or_inf():
    # A brand-new customer: tenure 0 and a blank TotalCharges (as in the raw data).
    df = clean_and_engineer(_base_row(tenure=0, TotalCharges=" "))
    assert df["TotalCharges"].iloc[0] == 0.0
    for col in ["charges_per_tenure_month", "monthly_to_total_ratio"]:
        val = df[col].iloc[0]
        assert np.isfinite(val), f"{col} should be finite, got {val}"


def test_num_addon_services_counts_yes_values():
    # Subscribe to exactly two add-ons; expect a count of 2.
    df = clean_and_engineer(_base_row(OnlineSecurity="Yes", StreamingTV="Yes"))
    assert df["num_addon_services"].iloc[0] == 2
    assert set(ADDON_SERVICES).issubset(_base_row().columns)  # guards a rename


def test_binary_flag_features():
    m2m = clean_and_engineer(_base_row(Contract="Month-to-month"))
    assert m2m["is_month_to_month"].iloc[0] == 1
    fiber = clean_and_engineer(_base_row(InternetService="Fiber optic"))
    assert fiber["has_fiber"].iloc[0] == 1


def test_serving_row_matches_training_width():
    # Fit the transformer on the full training data...
    raw = pd.read_csv("data/raw/telco_churn.csv").drop(columns=["Churn"])
    pre = build_preprocessor().fit(clean_and_engineer(raw))
    train_width = pre.transform(clean_and_engineer(raw)).shape[1]

    # ...then a single request with NO customerID must produce the same width.
    request = _base_row().drop(columns=["customerID"])
    serve_width = pre.transform(clean_and_engineer(request)).shape[1]
    assert serve_width == train_width


def test_clean_handles_missing_id_and_target():
    # Serving payloads carry neither customerID nor Churn; must not raise.
    request = _base_row().drop(columns=["customerID"])
    out = clean_and_engineer(request)
    assert "customerID" not in out.columns
    assert "Churn" not in out.columns
