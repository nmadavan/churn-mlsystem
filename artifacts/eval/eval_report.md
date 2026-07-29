# Evaluation Report

_Generated: 2026-07-29T00:33:00.572892+00:00_  
Training rows: 4507 | Val: 1127 | Test: 1409

## Baseline vs Candidate (validation set)

| Metric | Baseline (LogisticRegression) | Candidate (RandomForest) |
|--------|----------|-----------|
| roc_auc | 0.8525 | 0.8286 |
| pr_auc | 0.6814 | 0.6297 |
| f1 | 0.6341 | 0.5494 |
| precision | 0.533 | 0.6715 |
| recall | 0.7826 | 0.4649 |
| accuracy | 0.7604 | 0.7977 |

## Promotion decision

- **Promote candidate:** False
- **Rule:** candidate AUC 0.8286 vs baseline 0.8525; min_auc>=0.8 -> True; not worse than baseline by >0.01 -> False
- **Winner:** LogisticRegression (version v1)

## Winner performance (held-out test set)

| Metric | Value |
|--------|-------|
| roc_auc | 0.8469 |
| pr_auc | 0.659 |
| f1 | 0.6134 |
| precision | 0.5052 |
| recall | 0.7807 |
| accuracy | 0.7388 |
