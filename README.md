# Churn — Mini Production ML System

A small end-to-end machine learning system for telco customer churn prediction:
data ingestion, a reproducible training pipeline, an online + batch inference service,
and a monitoring/retraining plan.

## Setup

```bash
# From the churn-mlsystem/ directory
python3 -m venv .venv
source .venv/bin/activate
python scripts/check_env.py --install   # installs only what is missing
```

## Layout

| Path | Purpose |
|------|---------|
| `config.yaml` | Paths, columns, and thresholds (single source of truth) |
| `src/preprocessing.py` | Shared cleaning + feature code (used by training and serving) |
| `src/ingest.py` | Batch ingestion of new data files |
| `src/train.py` | Training pipeline: baseline vs candidate, evaluation, promotion |
| `src/app.py` | FastAPI inference service |
| `scripts/` | Env check, batch scoring, load test |
| `models/` | Versioned model artifacts + `current_best.json` registry |
| `docs/` | Design document and architecture diagram |

## Run order

Documented per component as the system is built (see `docs/design_doc.md`).
