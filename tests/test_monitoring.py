"""Tests for the PSI drift metric and the retraining trigger."""
import numpy as np

from src.config import load_config
from src.monitoring import population_stability_index, retraining_decision

CFG = load_config()  # min_new_rows=5000, auc_drop=0.05, drift_psi_warn=0.20


def test_psi_near_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert population_stability_index(x, x) < 0.01


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(3, 1, 5000)  # mean shifted by 3 std
    assert population_stability_index(a, b) > 0.2


def test_retrain_on_feature_drift():
    should, reasons = retraining_decision(10, 0.85, 0.85, max_psi=0.5, cfg=CFG)
    assert should and any("drift" in r for r in reasons)


def test_retrain_on_data_volume():
    should, reasons = retraining_decision(10_000, 0.85, 0.85, max_psi=0.01, cfg=CFG)
    assert should and any("data volume" in r for r in reasons)


def test_retrain_on_auc_drop():
    should, reasons = retraining_decision(10, recent_auc=0.78, reference_auc=0.85, max_psi=0.01, cfg=CFG)
    assert should and any("degradation" in r for r in reasons)


def test_no_retrain_when_all_healthy():
    should, reasons = retraining_decision(10, 0.85, 0.85, max_psi=0.01, cfg=CFG)
    assert not should and reasons == []
