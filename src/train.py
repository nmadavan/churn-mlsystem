"""Training pipeline: load -> split -> train baseline + candidate -> evaluate ->
promote -> save.

Run:
    python -m src.train

Trains two models on the same features:
  - baseline : Logistic Regression (simple, fast, interpretable)
  - candidate: Random Forest (more capacity)
Both use class_weight to counter the ~27%/73% churn imbalance. The candidate is
promoted only if it clears the guardrail in evaluate.should_promote.
"""
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import load_config, resolve_path
from src.evaluate import compute_metrics, should_promote, write_report
from src.preprocessing import make_feature_pipeline
from src.registry import current_best, promote, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("train")


def load_training_data(cfg: dict) -> pd.DataFrame:
    """Prefer the ingested training table; fall back to raw if it doesn't exist."""
    training_path = resolve_path(cfg["data"]["training_data_path"])
    if training_path.exists():
        log.info("Loading training table: %s", training_path.name)
        return pd.read_csv(training_path)
    log.warning("No training table found; falling back to raw data. Run ingestion first.")
    return pd.read_csv(resolve_path(cfg["data"]["raw_path"]))


def split_data(df: pd.DataFrame, cfg: dict):
    """Stratified train/val/test split so each keeps the ~27% churn ratio."""
    target = cfg["target"]["column"]
    y = (df[target] == cfg["target"]["positive_label"]).astype(int)
    X = df.drop(columns=[target])
    rs = cfg["split"]["random_state"]
    strat = y if cfg["split"]["stratify"] else None

    # First carve off the test set, then split the remainder into train/val.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], random_state=rs, stratify=strat)
    strat_rest = y_rest if cfg["split"]["stratify"] else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=cfg["split"]["val_size"], random_state=rs, stratify=strat_rest)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_models():
    """Return (name, pipeline) pairs for the baseline and candidate."""
    baseline = make_feature_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced"))
    candidate = make_feature_pipeline(
        RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1))
    return ("LogisticRegression", baseline), ("RandomForest", candidate)


def _prob(pipeline, X):
    return pipeline.predict_proba(X)[:, 1]


def run(cfg: dict) -> dict:
    df = load_training_data(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, cfg)
    log.info("Split sizes -> train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

    (base_name, baseline), (cand_name, candidate) = build_models()
    baseline.fit(X_train, y_train)
    candidate.fit(X_train, y_train)

    base_val = compute_metrics(y_val, _prob(baseline, X_val))
    cand_val = compute_metrics(y_val, _prob(candidate, X_val))
    log.info("Validation ROC AUC -> baseline=%.4f candidate=%.4f",
             base_val["roc_auc"], cand_val["roc_auc"])

    do_promote, reason = should_promote(cand_val, base_val, cfg)
    winner_name, winner_pipe = (cand_name, candidate) if do_promote else (base_name, baseline)
    log.info("Promote candidate? %s (%s)", do_promote, reason)

    # Final unbiased estimate for the winner on the untouched test set.
    winner_test = compute_metrics(y_test, _prob(winner_pipe, X_test))

    version = save_model(
        winner_pipe,
        {"winner_model": winner_name,
         "promotion": {"promote": do_promote, "reason": reason},
         "baseline_val_metrics": base_val,
         "candidate_val_metrics": cand_val,
         "test_metrics": winner_test},
        cfg)
    log.info("Saved winner %s as %s", winner_name, version)

    # Cross-run guardrail: only repoint current_best if this winner is at least as
    # good on test ROC AUC as whatever is already live.
    live = current_best(cfg)
    if live is None or winner_test["roc_auc"] >= live["metrics"]["roc_auc"]:
        promote(version, winner_test, cfg)
        log.info("Promoted %s to current_best (test ROC AUC=%.4f)", version, winner_test["roc_auc"])
    else:
        log.info("Kept existing current_best %s (test ROC AUC=%.4f >= new %.4f)",
                 live["version"], live["metrics"]["roc_auc"], winner_test["roc_auc"])

    report = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "baseline": {"model": base_name, "val_metrics": base_val},
        "candidate": {"model": cand_name, "val_metrics": cand_val},
        "promotion": {"promote": do_promote, "reason": reason},
        "winner": {"model": winner_name, "version": version, "test_metrics": winner_test},
    }
    write_report(report, cfg)
    log.info("Wrote evaluation report to %s", cfg["evaluate"]["report_dir"])
    return report


if __name__ == "__main__":
    run(load_config())
