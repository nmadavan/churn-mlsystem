"""FastAPI inference service for churn prediction.

Endpoints:
    GET  /health   -> liveness + which model version is loaded
    POST /predict  -> churn probability for one customer

The service loads the promoted pipeline (current_best.json) once at startup. Because
it reuses src.preprocessing through the saved pipeline, the exact training-time
feature transformation is applied to every request -- no separate serving code path.

Run:
    uvicorn src.app:app --port 8000
"""
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config
from src.registry import load_current_model

cfg = load_config()
DECISION_THRESHOLD = 0.5
STATE: dict = {"model": None, "version": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup rather than per request.
    try:
        model, pointer = load_current_model(cfg)
        STATE["model"], STATE["version"] = model, pointer["version"]
    except FileNotFoundError:
        STATE["model"], STATE["version"] = None, None  # /health will report not-ready
    yield
    STATE.clear()


app = FastAPI(title="Churn Prediction API", version="1.0", lifespan=lifespan)


class CustomerFeatures(BaseModel):
    """One customer's raw fields -- the same schema as the source data."""
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 1, "PhoneService": "No",
                "MultipleLines": "No phone service", "InternetService": "DSL",
                "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    }


class PredictionResponse(BaseModel):
    # model_version starts with "model_", which Pydantic treats as protected; opt out.
    model_config = {"protected_namespaces": ()}

    churn_prediction: str = Field(description="Yes/No at the decision threshold")
    churn_probability: float = Field(description="P(churn), 0..1")
    model_version: str
    threshold: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": STATE["model"] is not None,
        "model_version": STATE["version"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded. Run `python -m src.train`.")

    # One-row DataFrame -> the saved pipeline runs clean_and_engineer + preprocessing.
    row = pd.DataFrame([features.model_dump()])
    prob = float(STATE["model"].predict_proba(row)[0, 1])
    label = (cfg["target"]["positive_label"] if prob >= DECISION_THRESHOLD
             else cfg["target"]["negative_label"])

    return PredictionResponse(
        churn_prediction=label,
        churn_probability=round(prob, 4),
        model_version=STATE["version"],
        threshold=DECISION_THRESHOLD,
    )
