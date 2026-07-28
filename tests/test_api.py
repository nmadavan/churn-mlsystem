"""Tests for the FastAPI inference service.

Uses TestClient (in-process, no running server). The whole module is skipped if no
model has been trained yet, so `pytest` still passes on a fresh clone.
"""
import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.config import load_config
from src.registry import current_best

pytestmark = pytest.mark.skipif(
    current_best(load_config()) is None,
    reason="no trained model; run `python -m src.train` first",
)

VALID_PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 85.0,
}


def test_health_reports_loaded_model():
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_predict_returns_valid_response():
    with TestClient(app) as client:
        resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["churn_prediction"] in {"Yes", "No"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_version"]


def test_predict_rejects_missing_field():
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
    with TestClient(app) as client:
        resp = client.post("/predict", json=bad)
    assert resp.status_code == 422  # Pydantic validation error
