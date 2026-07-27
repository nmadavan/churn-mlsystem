"""Split the raw dataset into a few 'daily' CSVs to demonstrate ingestion.

Deterministic: the same split every run. Writes day1.csv, day2.csv, day3.csv into
the configured incoming directory.

Run:
    python scripts/make_demo_batches.py
"""
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path

N_BATCHES = 3


def main() -> None:
    cfg = load_config()
    raw = pd.read_csv(resolve_path(cfg["data"]["raw_path"]))
    incoming_dir = resolve_path(cfg["data"]["incoming_dir"])
    incoming_dir.mkdir(parents=True, exist_ok=True)

    # Split the row indices (an ndarray) rather than the DataFrame directly, then
    # slice with iloc. Same result, without the DataFrame-swapaxes deprecation noise.
    for i, idx in enumerate(np.array_split(np.arange(len(raw)), N_BATCHES), start=1):
        out = incoming_dir / f"day{i}.csv"
        raw.iloc[idx].to_csv(out, index=False)
        print(f"wrote {out.name}: {len(idx)} rows")


if __name__ == "__main__":
    main()
