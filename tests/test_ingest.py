"""Tests for record-level validation in ingestion.

The key discipline: bad rows are quarantined, good rows still pass, and domain
knowledge (blank TotalCharges for tenure=0 customers) is respected.
"""
import pandas as pd

from src.config import load_config
from src.ingest import validate_records

CFG = load_config()


def _row(**overrides):
    row = {
        "customerID": "TEST-0001", "gender": "Female", "SeniorCitizen": 0,
        "Partner": "No", "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year",
        "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
        "MonthlyCharges": 50.0, "TotalCharges": "600.0", "Churn": "No",
    }
    row.update(overrides)
    return row


def test_clean_batch_has_no_rejects():
    df = pd.DataFrame([_row(customerID="A"), _row(customerID="B")])
    valid, rejected = validate_records(df, CFG)
    assert len(valid) == 2 and len(rejected) == 0


def test_new_customer_blank_total_charges_is_valid():
    # tenure=0 with a blank TotalCharges is legitimate, NOT a reject.
    df = pd.DataFrame([_row(tenure=0, TotalCharges=" ")])
    valid, rejected = validate_records(df, CFG)
    assert len(valid) == 1 and len(rejected) == 0


def test_bad_rows_are_quarantined_not_dropped():
    df = pd.DataFrame([
        _row(customerID="GOOD"),
        _row(customerID="NEG", tenure=-3),
        _row(customerID="TEXT", MonthlyCharges="abc"),
        _row(customerID="", tenure=5),
    ])
    valid, rejected = validate_records(df, CFG)
    assert set(valid["customerID"]) == {"GOOD"}
    assert len(rejected) == 3
    assert "reject_reason" in rejected.columns
    reasons = " ".join(rejected["reject_reason"])
    assert "bad_tenure" in reasons and "bad_monthly_charges" in reasons and "missing_id" in reasons


def test_multiple_reasons_on_one_row():
    df = pd.DataFrame([_row(customerID="X", tenure=-1, MonthlyCharges=-5)])
    _, rejected = validate_records(df, CFG)
    reason = rejected["reject_reason"].iloc[0]
    assert "bad_tenure" in reason and "bad_monthly_charges" in reason
