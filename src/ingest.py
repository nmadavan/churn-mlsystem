"""Batch ingestion: merge a new data file into the growing training table.

Reads a daily CSV, checks its schema, upserts rows by customer id into
data/training_data.csv, and logs what was ingested (rows, date) to an audit log.

Run:
    python -m src.ingest data/incoming/day1.csv
"""
import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("ingest")


def expected_columns(cfg: dict) -> list[str]:
    """The schema new files must match: the training table's if it exists, else raw."""
    training = resolve_path(cfg["data"]["training_data_path"])
    reference = training if training.exists() else resolve_path(cfg["data"]["raw_path"])
    return list(pd.read_csv(reference, nrows=0).columns)


def check_schema(incoming: pd.DataFrame, expected: list[str]) -> tuple[set, set]:
    """Return (missing, unexpected) columns. Empty sets mean the schema is valid."""
    missing = set(expected) - set(incoming.columns)
    unexpected = set(incoming.columns) - set(expected)
    return missing, unexpected


def validate_records(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (valid, rejected). Rejected rows get a `reject_reason` column.

    File-level schema is checked separately; this catches per-record data-quality
    problems so a few bad rows are quarantined instead of failing the whole batch.
    Domain knowledge is encoded here: a blank TotalCharges is NOT a reject (it is the
    legitimate tenure=0 case), but a non-numeric one is.
    """
    id_col = cfg["data"]["id_column"]
    target = cfg["target"]["column"]
    labels = {cfg["target"]["positive_label"], cfg["target"]["negative_label"]}

    reasons = pd.Series("", index=df.index)

    def flag(invalid_mask: pd.Series, text: str) -> None:
        m = invalid_mask.fillna(True)  # a NaN comparison result means "can't validate" -> reject
        reasons.loc[m] = (reasons.loc[m] + ";" + text).str.lstrip(";")

    ids = df[id_col].astype(str).str.strip()
    flag(df[id_col].isna() | (ids == ""), "missing_id")

    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    flag(tenure.isna() | (tenure < 0) | (tenure != tenure.round()), "bad_tenure")

    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    flag(monthly.isna() | (monthly < 0), "bad_monthly_charges")

    senior = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    flag(~senior.isin([0, 1]), "bad_senior_flag")

    if target in df.columns:
        flag(~df[target].isin(labels), "bad_target_label")

    total_raw = df["TotalCharges"].astype(str).str.strip()
    total_blank = total_raw.isin(["", "nan", "NaN"])       # allowed (tenure=0 customers)
    total_num = pd.to_numeric(df["TotalCharges"], errors="coerce")
    flag(~total_blank & (total_num.isna() | (total_num < 0)), "bad_total_charges")

    is_valid = reasons == ""
    valid = df[is_valid].copy()
    rejected = df[~is_valid].copy()
    rejected["reject_reason"] = reasons[~is_valid]
    return valid, rejected


def ingest(file_path: str, cfg: dict) -> dict:
    """Validate and merge one file. Returns a summary dict; raises on schema failure."""
    src_path = Path(file_path)
    incoming = pd.read_csv(src_path)

    # Guardrail 1 (file level): a schema mismatch (upstream column added/removed/
    # renamed) affects every row, so the whole file is rejected rather than
    # silently corrupting the training table.
    missing, unexpected = check_schema(incoming, expected_columns(cfg))
    if missing or unexpected:
        raise ValueError(
            f"Schema mismatch in {src_path.name}: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )

    # Guardrail 2 (record level): quarantine individual bad rows to a reject file,
    # then ingest the good ones. A few bad records never fail the whole batch.
    valid, rejected = validate_records(incoming, cfg)
    rows_rejected = len(rejected)
    if rows_rejected:
        reject_path = _write_rejects(rejected, src_path, cfg)
        log.warning(
            "Quarantined %d/%d records from %s -> %s",
            rows_rejected, len(incoming), src_path.name, reject_path.name,
        )

    id_col = cfg["data"]["id_column"]
    training_path = resolve_path(cfg["data"]["training_data_path"])

    if training_path.exists():
        existing = pd.read_csv(training_path)
        updated_ids = set(existing[id_col]) & set(valid[id_col])
        # Upsert: keep='last' lets an incoming row overwrite an existing customer.
        merged = pd.concat([existing, valid]).drop_duplicates(
            subset=id_col, keep="last"
        )
        rows_updated = len(updated_ids)
        rows_new = len(valid) - rows_updated
    else:
        merged = valid.drop_duplicates(subset=id_col, keep="last")
        rows_new, rows_updated = len(merged), 0

    training_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(training_path, index=False)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": src_path.name,
        "rows_in_file": len(incoming),
        "rows_new": rows_new,
        "rows_updated": rows_updated,
        "rows_rejected": rows_rejected,
        "total_rows": len(merged),
    }
    _append_audit_log(summary, cfg)
    log.info(
        "Ingested %s: %d rows (%d new, %d updated, %d rejected) -> %d total on %s",
        summary["source_file"], summary["rows_in_file"], summary["rows_new"],
        summary["rows_updated"], summary["rows_rejected"], summary["total_rows"],
        summary["timestamp"],
    )
    return summary


def _write_rejects(rejected: pd.DataFrame, src_path: Path, cfg: dict) -> Path:
    """Write quarantined rows (with reject_reason) to the rejects directory."""
    rejects_dir = resolve_path(cfg["data"]["rejects_dir"])
    rejects_dir.mkdir(parents=True, exist_ok=True)
    reject_path = rejects_dir / f"{src_path.stem}_rejects.csv"
    rejected.to_csv(reject_path, index=False)
    return reject_path


def _append_audit_log(summary: dict, cfg: dict) -> None:
    """Append one row to the ingestion audit log (created with a header if new)."""
    log_path = resolve_path(cfg["data"]["ingestion_log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([summary])
    row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a data file into the training table.")
    parser.add_argument("file", help="path to the incoming CSV")
    args = parser.parse_args()

    cfg = load_config()
    try:
        ingest(args.file, cfg)
    except (FileNotFoundError, ValueError) as err:
        log.error("Ingestion failed: %s", err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
