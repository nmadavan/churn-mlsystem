"""Offline evaluation: metrics and the candidate-promotion guardrail.

Metric choice for churn (a ~27% positive, imbalanced problem):
  - ROC AUC       : ranking quality, threshold-independent; the headline metric.
  - PR AUC        : average precision; more informative than ROC when positives
                    are rare, because it focuses on the minority (churn) class.
  - F1            : balances precision and recall at the decision threshold.
  - accuracy      : reported for context only; misleading alone here (predicting
                    "no churn" for everyone already scores ~73%).
"""
import json

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import resolve_path


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Compute ranking and threshold metrics from true labels and P(churn)."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }


def should_promote(candidate: dict, baseline: dict, cfg: dict) -> tuple[bool, str]:
    """Guardrail: promote the candidate only if it clears both bars.

    1. Absolute quality: candidate ROC AUC >= promote_min_auc.
    2. No regression:    candidate ROC AUC not worse than baseline by more than
                         promote_max_auc_drop_vs_baseline.
    """
    min_auc = cfg["evaluate"]["promote_min_auc"]
    max_drop = cfg["evaluate"]["promote_max_auc_drop_vs_baseline"]

    passes_min = candidate["roc_auc"] >= min_auc
    passes_vs_baseline = candidate["roc_auc"] >= baseline["roc_auc"] - max_drop
    promote = passes_min and passes_vs_baseline

    reason = (
        f"candidate AUC {candidate['roc_auc']:.4f} vs baseline {baseline['roc_auc']:.4f}; "
        f"min_auc>={min_auc} -> {passes_min}; "
        f"not worse than baseline by >{max_drop} -> {passes_vs_baseline}"
    )
    return promote, reason


def write_report(report: dict, cfg: dict) -> None:
    """Write the evaluation report as both JSON (machine) and Markdown (human)."""
    report_dir = resolve_path(cfg["evaluate"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    (report_dir / "eval_report.md").write_text(_render_markdown(report))


def _render_markdown(report: dict) -> str:
    metric_keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]
    base, cand = report["baseline"]["val_metrics"], report["candidate"]["val_metrics"]

    lines = [
        "# Evaluation Report",
        "",
        f"_Generated: {report['created_at']}_  ",
        f"Training rows: {report['n_train']} | Val: {report['n_val']} | Test: {report['n_test']}",
        "",
        "## Baseline vs Candidate (validation set)",
        "",
        f"| Metric | Baseline ({report['baseline']['model']}) | Candidate ({report['candidate']['model']}) |",
        "|--------|----------|-----------|",
    ]
    for k in metric_keys:
        lines.append(f"| {k} | {base[k]} | {cand[k]} |")

    lines += [
        "",
        "## Promotion decision",
        "",
        f"- **Promote candidate:** {report['promotion']['promote']}",
        f"- **Rule:** {report['promotion']['reason']}",
        f"- **Winner:** {report['winner']['model']} (version {report['winner']['version']})",
        "",
        "## Winner performance (held-out test set)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k in metric_keys:
        lines.append(f"| {k} | {report['winner']['test_metrics'][k]} |")
    lines.append("")
    return "\n".join(lines)
