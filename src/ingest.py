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


def ingest(file_path: str, cfg: dict) -> dict:
    """Validate and merge one file. Returns a summary dict; raises on schema failure."""
    src_path = Path(file_path)
    incoming = pd.read_csv(src_path)

    # Guardrail: a schema mismatch (upstream column added/removed/renamed) is
    # rejected here rather than silently corrupting the training table.
    missing, unexpected = check_schema(incoming, expected_columns(cfg))
    if missing or unexpected:
        raise ValueError(
            f"Schema mismatch in {src_path.name}: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )

    id_col = cfg["data"]["id_column"]
    training_path = resolve_path(cfg["data"]["training_data_path"])

    if training_path.exists():
        existing = pd.read_csv(training_path)
        updated_ids = set(existing[id_col]) & set(incoming[id_col])
        # Upsert: keep='last' lets an incoming row overwrite an existing customer.
        merged = pd.concat([existing, incoming]).drop_duplicates(
            subset=id_col, keep="last"
        )
        rows_updated = len(updated_ids)
        rows_new = len(incoming) - rows_updated
    else:
        merged = incoming.drop_duplicates(subset=id_col, keep="last")
        rows_new, rows_updated = len(merged), 0

    training_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(training_path, index=False)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": src_path.name,
        "rows_in_file": len(incoming),
        "rows_new": rows_new,
        "rows_updated": rows_updated,
        "total_rows": len(merged),
    }
    _append_audit_log(summary, cfg)
    log.info(
        "Ingested %s: %d rows (%d new, %d updated) -> %d total on %s",
        summary["source_file"], summary["rows_in_file"], summary["rows_new"],
        summary["rows_updated"], summary["total_rows"], summary["timestamp"],
    )
    return summary


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
