# Mini Production ML System — Telco Customer Churn

**Design Document**

## 1. Problem definition and metrics

The system predicts whether a telco customer will churn (cancel service), framed as binary
classification. The intended users are the retention team, who consume predictions in two
ways: an on-demand risk score when an agent opens a customer's profile, and a nightly score
of the entire customer base to target a retention campaign. The input is a customer's
account and service attributes; the output is a churn probability plus a Yes/No label and
the model version that produced it.

The dataset is the public Telco Churn set: 7,043 customers, 21 columns, with `Churn`
(Yes/No) as the label. Churn is the minority class at 26.5%, and that imbalance drives the
metric choice. I report ROC AUC as the headline metric because it measures ranking quality
independent of any threshold, which suits a use case where the retention team may act on the
top-N highest-risk customers rather than a fixed cut-off. I also report PR AUC (average
precision), which is more informative than ROC AUC when positives are rare because it
focuses on the churn class, along with F1, precision, and recall. Accuracy is reported only
for context: predicting "no churn" for everyone already scores 73.5%, so accuracy alone is
misleading. Recall matters most for the business, because a missed churner is a lost
customer, whereas a false positive only costs a cheap retention offer.

## 2. Data and feature design

**Cleaning and assumptions.** `TotalCharges` arrives as text because 11 brand-new customers
(tenure = 0) have a blank value. I coerce the column to numeric and fill those blanks with
0.0, on the assumption that a customer who has never completed a billing cycle has charged
approximately zero so far. I drop `customerID` as a non-predictive identifier and map the
target to 1/0.

**Engineered features.** I add six non-trivial features, each motivated by exploratory
analysis (the statistics below are reproducible via `scripts/eda.py`):

- `tenure_bucket` — tenure banded into ranges. Churn falls steeply across bands (52.9% at
  0–6 months down to 9.5% at 49+ months).
- `num_addon_services` — a count of the six optional add-on services a customer subscribes
  to, capturing engagement in a single number.
- `charges_per_tenure_month` — total charges divided by (tenure + 1); the +1 guards against
  division by zero for new customers.
- `monthly_to_total_ratio` — this month's charge relative to lifetime spend, which is high
  for new or front-loaded customers who churn more.
- `is_month_to_month` — isolates the highest-risk contract type (42.7% churn in analysis).
- `has_fiber` — fibre-optic customers churn notably more, so I expose it directly.

After one-hot encoding the remaining categoricals and scaling the numerics, the model sees
55 features.

**Offline vs online and training–serving skew.** All six engineered features are row-local
arithmetic computable from a single request, so there is no historical lookup and therefore
no offline/online split to reconcile. If a feature required history (for example, support
calls in the last 30 days), it would need a feature store or a precompute-and-lookup table;
I note this as a contrast rather than a requirement here. To prevent training–serving skew, a
single module (`preprocessing.py`) implements all cleaning and feature logic and is imported
by both training and serving through one saved scikit-learn Pipeline. Because training and
serving execute the identical code and the identical fitted transformer, they cannot drift
apart. A test asserts that a single serving request transforms to the same 55 columns as
training.

## 3. Data pipeline

Ingestion (`ingest.py`) merges a new daily CSV into a growing training table and applies two
levels of validation. At the file level, a schema mismatch — a column added, removed, or
renamed — rejects the whole file, because every row is affected and this usually signals an
upstream change that needs a human. At the record level, individual bad rows (negative
tenure, non-numeric charges, missing id, invalid label) are quarantined to a reject file
with a reason code while the good rows are still ingested, so a few bad records never fail
the whole batch. This mirrors long-standing batch-processing discipline: route errors to a
reject dataset for reprocessing rather than abort the job. Validation encodes domain
knowledge — a blank `TotalCharges` on a tenure-0 customer is treated as valid, not
quarantined. Ingestion also merges by `customerID` (upsert), so re-processing a file is
idempotent, and it logs every run (rows in, new, updated, rejected, and total) to an audit
log. The reject rate per batch becomes a data-quality signal for monitoring.

## 4. Model choice and evaluation

I train two models on the same features: a Logistic Regression baseline (simple, fast,
interpretable) and a Random Forest candidate (more capacity). Both use class weighting to
counter the imbalance. The data is split into stratified train/validation/test sets
(4,507 / 1,127 / 1,409), each preserving the 26.5% churn ratio.

On the validation set, the baseline outperformed the candidate: ROC AUC 0.8525 versus 0.8286.
A promotion guardrail decides which model to release: promote the candidate only if its ROC
AUC is at least 0.80 and not worse than the baseline by more than 0.01. The candidate failed
the second condition, so the guardrail correctly kept the baseline and prevented a
regression. This is a deliberate design outcome, not a shortfall: on well-engineered tabular
features a linear model is often competitive, and while the Random Forest had higher accuracy
(0.80 vs 0.76), its recall was far lower (0.46 vs 0.78) — worse for catching churners, which
is the point of the system. The promoted model's held-out test performance was ROC AUC
0.8469, with recall 0.78; test tracking validation indicates no severe overfitting.

Trained models are stored in a lightweight file-based registry: each version lives under
`models/v<N>/` (the serialized pipeline plus a metadata JSON), and a single `current_best.json`
pointer records which version is live, along with its metrics. This is a miniature of a tool
like MLflow — versioned artifacts plus one "which model is in production" pointer — and it is
what the serving layer and the rollback path both read.

## 5. Serving and inference pattern

The design is a hybrid of online and batch inference, because the two consumption modes have
different requirements. The online service (FastAPI) exposes `/predict`, which validates a
customer JSON with a typed schema (a malformed request is rejected with HTTP 422 before it
reaches the model), and `/health`, which reports the loaded model version. The model is
loaded once at startup rather than per request. The batch path (`batch_score.py`) scores a
whole CSV in a single vectorized pass.

I measured both (a representative run). Online latency over 300 requests was 6.68 ms average
and 7.19 ms at p95 (p99 7.80 ms); I report the percentile as well as the average because tail
latency is what users actually experience. Batch throughput was about 347,000 rows per second
— over two thousand times more efficient per row than the sequential online path, because it
avoids per-request HTTP, JSON parsing, and validation overhead. This justifies the hybrid
choice against the standard question of whether a human is waiting: for an agent looking at a
live profile, the ~6.7 ms online call is well under the ~100 ms "instant" threshold; for
scoring the whole base overnight, no human waits, so the batch path is the right tool. The service is also containerized (Dockerfile),
and the image trains the model during the build so it is fully self-contained and reproducible
across machines and operating systems.

## 6. Monitoring plan and retraining strategy

Monitoring spans three layers, each with an owner and an alert. Infrastructure metrics
(latency, p95, error rate) are for the on-call engineer, alerting if p95 exceeds 100 ms or
the error rate exceeds 1%. Data and feature metrics (row counts, null counts, PSI drift per
feature, and the ingestion reject rate) are for the ML engineer, alerting if PSI exceeds 0.20.
Model and business metrics (ROC AUC on recent labelled feedback, and realized versus predicted
churn) are for the data scientist, alerting if feedback AUC drops by 0.05 or more against the
promoted model.

The drift check is implemented, not just described. It computes the Population Stability Index
(PSI) per numeric feature between the training reference and a recent batch and logs a warning
above 0.20. On a simulated drifted batch it reported PSI of 5.0 and 4.3 on shifted features; on
a real in-distribution batch it reported roughly 0.0, correctly staying silent. The retraining
trigger is a pure function that recommends retraining if any of three signals fire: enough new
data has accumulated, feedback AUC has dropped past the threshold, or feature drift exceeds the
PSI threshold.

**Incident scenario.** Suppose an upstream export drops the `Contract` column. The file-level
schema check rejects the file immediately, so the corrupt data never enters the training table.
Had a column's values silently changed instead — for example a units change inflating
`MonthlyCharges` — the PSI drift check would flag it on the next monitoring run and the
retraining trigger would fire. The response is to alert the responsible engineer, fix the
upstream pipeline, reprocess the quarantined rows, and, if the live model has degraded, roll
back by pointing `current_best.json` at the previous good version before retraining on clean
data.

## 7. Key trade-offs, limitations, and future work

I chose the simpler model over the more complex one, favouring interpretability and recall
over raw accuracy; for churn that is the right trade-off. That said, the Random Forest
candidate was left largely untuned and evaluation used a single stratified split rather than
cross-validation, so the baseline's win reflects a fair-but-not-exhaustive comparison at this
scope; tuning the candidate and adding cross-validation are natural next steps. The decision threshold is fixed at
0.5 and could be tuned to the retention budget as future work. The registry is file-based
rather than a tracking server, which is sufficient at this scale but would become a managed
artifact store in a larger system; likewise, the model binary is trained on setup because it
is cheap here (about two seconds), whereas an expensive model would be published once to an
artifact registry and pulled by each deploy rather than committed to source control or
retrained per deploy. There is no feature store because all features are row-local; one would
be required for history-based features. The system is single-node and single-model, so
autoscaling and multi-tenant concerns are out of scope. These are conscious boundaries for an
assignment focused on applying production-ML practices end to end rather than maximizing any
single metric.

## Architecture

See `architecture.svg`: raw data → ingestion (schema + record validation) → training table →
training pipeline (baseline vs candidate + guardrail) → model registry → serving (online API
and batch) → monitoring (PSI drift, data quality) → retraining trigger, closing the loop.
