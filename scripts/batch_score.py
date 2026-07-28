"""Batch scoring: score every row of a CSV with the promoted model in one pass.

This is the batch inference pattern (contrast with the online /predict API): no
human waiting, so we optimise for throughput by vectorising over all rows at once.

Run:
    python -m scripts.batch_score                     # scores the training table
    python -m scripts.batch_score data/incoming/day1.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.registry import load_current_model

THRESHOLD = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a CSV with the promoted model.")
    parser.add_argument("input", nargs="?", help="CSV to score (default: training table)")
    parser.add_argument("-o", "--output", help="output CSV path")
    args = parser.parse_args()

    cfg = load_config()
    model, pointer = load_current_model(cfg)

    input_path = Path(args.input) if args.input else resolve_path(cfg["data"]["training_data_path"])
    df = pd.read_csv(input_path)

    # Time only the scoring, not the disk I/O, for a fair rows/sec figure.
    start = time.perf_counter()
    probs = model.predict_proba(df)[:, 1]
    elapsed = time.perf_counter() - start

    id_col = cfg["data"]["id_column"]
    scored = pd.DataFrame({
        id_col: df[id_col] if id_col in df.columns else np.arange(len(df)),
        "churn_probability": probs.round(4),
        "churn_prediction": np.where(probs >= THRESHOLD,
                                     cfg["target"]["positive_label"],
                                     cfg["target"]["negative_label"]),
    })

    out_path = Path(args.output) if args.output else resolve_path(
        f"data/scored/{input_path.stem}_scored.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)

    rows_per_sec = len(df) / elapsed if elapsed else float("inf")
    print(f"Scored {len(df)} rows with model {pointer['version']} in {elapsed:.3f}s "
          f"({rows_per_sec:,.0f} rows/sec)")
    print(f"Predicted churn: {int((probs >= THRESHOLD).sum())} / {len(df)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
