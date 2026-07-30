"""Tests for the FastAPI inference service.

Uses TestClient (in-process, no running server). The whole module is skipped when the
trained model *file* is absent (e.g. a fresh clone before training, since model.pkl is
gitignored), so `pytest` still passes without a model present. Note: checking the
registry pointer alone is not enough, because current_best.json is committed while the
.pkl it references is not.
"""
import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.config import load_config, resolve_path
from src.registry import current_best


def _model_file_available() -> bool:
    pointer = current_best(load_config())
    return pointer is not None and resolve_path(pointer["model_path"]).exists()


pytestmark = pytest.mark.skipif(
    not _model_file_available(),
    reason="no trained model file; run `python -m src.train` first",
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
