"""Monitoring: a working drift / data-quality check plus the retraining trigger.

Two things a production system watches for after deployment:
  1. Data / feature drift  -- has the incoming distribution moved away from what the
     model was trained on? Measured here with PSI (Population Stability Index).
  2. Model degradation     -- has quality on recent labelled feedback dropped?

The retraining trigger combines those with a data-volume signal. This module
implements the drift check for real; the trigger is a plain function that a
scheduler could call.

Run:
    python -m src.monitoring                       # simulate a drifted batch (demo)
    python -m src.monitoring data/incoming/day1.csv  # check a real batch (no drift)
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import load_config, resolve_path
from src.registry import current_best, load_current_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("monitoring")


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI between a reference (expected) and a recent (actual) numeric sample.

    Rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant drift.
    Bins are set from the reference quantiles so each reference bin is ~equally full.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    exp_frac = np.histogram(expected, edges)[0] / len(expected)
    act_frac = np.histogram(actual, edges)[0] / len(actual)
    exp_frac = np.clip(exp_frac, 1e-6, None)  # avoid log(0) / divide-by-zero
    act_frac = np.clip(act_frac, 1e-6, None)
    return float(np.sum((act_frac - exp_frac) * np.log(act_frac / exp_frac)))


def drift_report(reference: pd.DataFrame, recent: pd.DataFrame, cfg: dict) -> dict:
    """Per-feature PSI, mean/std shift, and null count for the recent batch."""
    warn = cfg["monitoring"]["drift_psi_warn"]
    report = {}
    for col in cfg["monitoring"]["monitored_numeric_features"]:
        ref = pd.to_numeric(reference[col], errors="coerce").dropna()
        cur_raw = pd.to_numeric(recent[col], errors="coerce")
        cur = cur_raw.dropna()
        psi = population_stability_index(ref.to_numpy(), cur.to_numpy())
        report[col] = {
            "psi": round(psi, 4),
            "drift": psi > warn,
            "ref_mean": round(float(ref.mean()), 2),
            "recent_mean": round(float(cur.mean()), 2),
            "ref_std": round(float(ref.std()), 2),
            "recent_std": round(float(cur.std()), 2),
            "recent_nulls": int(cur_raw.isna().sum()),
        }
    return report


def recent_feedback_auc(recent: pd.DataFrame, cfg: dict) -> float | None:
    """If the recent batch carries labels, score it with the live model for AUC.

    This is the "quality on recent labelled feedback" signal. Returns None when the
    batch has no target column (unlabelled traffic).
    """
    target = cfg["target"]["column"]
    if target not in recent.columns:
        return None
    model, _ = load_current_model(cfg)
    y = (recent[target] == cfg["target"]["positive_label"]).astype(int)
    if y.nunique() < 2:
        return None  # AUC undefined without both classes
    prob = model.predict_proba(recent.drop(columns=[target]))[:, 1]
    return round(float(roc_auc_score(y, prob)), 4)


def retraining_decision(new_rows: int, recent_auc: float | None,
                        reference_auc: float | None, max_psi: float, cfg: dict) -> tuple[bool, list[str]]:
    """Combine signals into a retrain / don't-retrain decision. Pure function.

    Retrain if ANY of: enough new data, model degradation on feedback, or feature drift.
    """
    reasons = []
    min_rows = cfg["monitoring"]["min_new_rows_for_retrain"]
    auc_drop = cfg["monitoring"]["auc_drop_retrain"]
    psi_warn = cfg["monitoring"]["drift_psi_warn"]

    if new_rows >= min_rows:
        reasons.append(f"data volume: {new_rows} new rows >= {min_rows}")
    if recent_auc is not None and reference_auc is not None and (reference_auc - recent_auc) >= auc_drop:
        reasons.append(f"model degradation: AUC {recent_auc} vs reference {reference_auc} "
                       f"(drop >= {auc_drop})")
    if max_psi > psi_warn:
        reasons.append(f"feature drift: max PSI {round(max_psi, 4)} > {psi_warn}")
    return (len(reasons) > 0, reasons)


def _simulate_drifted_batch(reference: pd.DataFrame) -> pd.DataFrame:
    """Shift the numeric distributions so the demo shows a clear drift warning."""
    drifted = reference.sample(2000, replace=True, random_state=7).copy()
    drifted["MonthlyCharges"] = drifted["MonthlyCharges"] + 30
    drifted["tenure"] = (drifted["tenure"] + 25).clip(upper=72)
    return drifted


def main() -> int:
    parser = argparse.ArgumentParser(description="Run drift/quality check and retraining trigger.")
    parser.add_argument("batch", nargs="?", help="recent batch CSV (default: simulate drift)")
    args = parser.parse_args()

    cfg = load_config()
    ref_path = resolve_path(cfg["data"]["training_data_path"])
    if not ref_path.exists():
        ref_path = resolve_path(cfg["data"]["raw_path"])
    reference = pd.read_csv(ref_path)

    if args.batch:
        recent = pd.read_csv(args.batch)
        label = args.batch
    else:
        recent = _simulate_drifted_batch(reference)
        label = "simulated drifted batch"
        log.info("No batch given; simulating a drifted batch for demonstration.")

    log.info("Comparing '%s' (%d rows) against training reference (%d rows)",
             label, len(recent), len(reference))

    report = drift_report(reference, recent, cfg)
    print(f"\n{'feature':16} {'psi':>8} {'ref_mean':>9} {'recent_mean':>12} {'nulls':>6}  drift")
    for col, r in report.items():
        print(f"{col:16} {r['psi']:>8.4f} {r['ref_mean']:>9} {r['recent_mean']:>12} "
              f"{r['recent_nulls']:>6}  {'YES' if r['drift'] else 'no'}")

    max_psi = max(r["psi"] for r in report.values())
    for col, r in report.items():
        if r["drift"]:
            log.warning("DRIFT on %s: PSI %.4f exceeds %.2f (mean %.2f -> %.2f)",
                        col, r["psi"], cfg["monitoring"]["drift_psi_warn"],
                        r["ref_mean"], r["recent_mean"])

    reference_auc = (current_best(cfg) or {}).get("metrics", {}).get("roc_auc")
    recent_auc = recent_feedback_auc(recent, cfg)
    should_retrain, reasons = retraining_decision(
        new_rows=len(recent), recent_auc=recent_auc,
        reference_auc=reference_auc, max_psi=max_psi, cfg=cfg)

    print(f"\nRecent-feedback AUC: {recent_auc}  (reference {reference_auc})")
    print(f"Retrain? {should_retrain}")
    for r in reasons:
        print(f"  - {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
