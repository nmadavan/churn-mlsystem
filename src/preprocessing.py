"""Shared cleaning and feature engineering for churn prediction.

Both the training pipeline and the serving API import from this module, so the
exact same transformation is applied offline and online. That single shared code
path is the defence against training-serving skew.

Design:
    clean_and_engineer()  -> deterministic, row-wise cleaning + feature creation
                             (no fitting needed; safe to run on one row or millions)
    build_preprocessor()  -> a ColumnTransformer that must be *fitted* on training
                             data (learns one-hot categories and scaler statistics)
    make_feature_pipeline(model) -> ties both steps to a model in one sklearn
                             Pipeline, saved as a single artifact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Columns that identify a row but carry no predictive signal.
ID_COLUMNS = ["customerID"]

# The six optional add-on services. "Yes" means the customer subscribes.
ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Feature groups the ColumnTransformer operates on, AFTER clean_and_engineer runs.
NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "num_addon_services", "charges_per_tenure_month",
    "monthly_to_total_ratio", "is_month_to_month", "has_fiber",
]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_bucket",
]

_TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
_TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49+"]


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw columns and add engineered features. Pure and stateless.

    Accepts the raw schema (with or without the target/ID columns) so it works
    identically on a training file and on a single request at serving time.
    """
    df = df.copy()

    # Drop identifier columns if present (absent in serving requests).
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    # TotalCharges arrives as text because 11 brand-new customers (tenure=0)
    # have a blank value. Coerce to numeric; those blanks become 0.0 because a
    # customer who has never completed a billing cycle has charged ~0 so far.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Feature 1: count of add-on services subscribed (0-6). An engagement signal.
    df["num_addon_services"] = sum((df[c] == "Yes").astype(int) for c in ADDON_SERVICES)

    # Feature 2: average spend per month of tenure. +1 avoids divide-by-zero for
    # tenure=0 customers.
    df["charges_per_tenure_month"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Feature 3: how large this month's bill is relative to lifetime spend. High
    # for new / front-loaded customers, who churn more.
    df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # Feature 4: isolate the highest-risk contract type (42.7% churn in EDA).
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # Feature 5: fibre-optic customers churn notably more; expose it directly.
    df["has_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)

    # Feature 6: tenure banding. Churn falls steeply across these bands.
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=_TENURE_BINS, labels=_TENURE_LABELS
    ).astype(str)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer: scale numerics, one-hot categoricals.

    handle_unknown='ignore' keeps serving robust: an unexpected category value in
    a live request produces all-zeros for that column instead of crashing.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_feature_pipeline(model) -> Pipeline:
    """Compose engineering -> preprocessing -> model into one fittable Pipeline.

    The whole thing is saved as a single artifact, so loading the model at serving
    time also loads the exact preprocessing used in training.
    """
    return Pipeline([
        ("engineer", FunctionTransformer(clean_and_engineer)),
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
