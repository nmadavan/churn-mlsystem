"""Exploratory analysis behind the feature choices.

Reproduces the churn statistics cited in the design document so they are verifiable.
Read-only; prints to stdout.

Run:
    python -m scripts.eda
"""
import pandas as pd

from src.config import load_config, resolve_path
from src.preprocessing import _TENURE_BINS, _TENURE_LABELS


def _churn_rate(df: pd.DataFrame, by: str) -> pd.Series:
    rate = df.groupby(by, observed=True)["_churn"].mean().mul(100).round(1)
    return rate.astype(str) + " %"


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(resolve_path(cfg["data"]["raw_path"]))
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")

    target = cfg["target"]["column"]
    df["_churn"] = (df[target] == cfg["target"]["positive_label"]).astype(int)

    print("\nClass balance (Churn):")
    print((df[target].value_counts(normalize=True) * 100).round(1).astype(str) + " %")

    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print(f"\nBlank TotalCharges rows: {int(tc.isna().sum())} "
          f"(tenure values: {sorted(df.loc[tc.isna(), 'tenure'].unique())})")

    print("\nChurn rate by Contract:")
    print(_churn_rate(df, "Contract").to_string())

    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=_TENURE_BINS, labels=_TENURE_LABELS).astype(str)
    print("\nChurn rate by tenure bucket:")
    print(_churn_rate(df, "tenure_bucket").reindex(_TENURE_LABELS).to_string())


if __name__ == "__main__":
    main()
