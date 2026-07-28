"""Tests for metric computation and the promotion guardrail."""
import numpy as np

from src.config import load_config
from src.evaluate import compute_metrics, should_promote

CFG = load_config()  # promote_min_auc=0.80, promote_max_auc_drop_vs_baseline=0.01


def test_compute_metrics_keys_and_ranges():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.9, 0.8, 0.2, 0.6])
    m = compute_metrics(y_true, y_prob)
    assert set(m) == {"roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"}
    assert all(0.0 <= v <= 1.0 for v in m.values())
    assert m["roc_auc"] == 1.0  # probabilities perfectly rank the labels


def test_promote_when_candidate_strong_and_not_a_regression():
    promote, _ = should_promote({"roc_auc": 0.86}, {"roc_auc": 0.85}, CFG)
    assert promote is True


def test_reject_when_below_min_auc():
    # Beats the baseline, but under the absolute 0.80 bar.
    promote, _ = should_promote({"roc_auc": 0.78}, {"roc_auc": 0.70}, CFG)
    assert promote is False


def test_reject_when_regression_beyond_tolerance():
    # Above 0.80, but worse than baseline by more than 0.01 (this is the real run).
    promote, _ = should_promote({"roc_auc": 0.8286}, {"roc_auc": 0.8525}, CFG)
    assert promote is False


def test_promote_within_tolerance_band():
    # Exactly 0.01 worse than baseline is allowed (>= boundary).
    promote, _ = should_promote({"roc_auc": 0.84}, {"roc_auc": 0.85}, CFG)
    assert promote is True
