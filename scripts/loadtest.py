"""Load test the online /predict endpoint: report average and p95 latency.

Latency percentiles matter more than the average for an online service: p95 is the
experience of the slowest 1-in-20 requests, which is what users actually complain
about. Start the server first, then run this.

    uvicorn src.app:app --port 8000        # in one terminal
    python -m scripts.loadtest --n 300     # in another

"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import requests

from src.config import load_config, resolve_path


def build_payloads(cfg: dict, n: int) -> list[dict]:
    """Sample n real customers and shape them into request bodies."""
    raw = pd.read_csv(resolve_path(cfg["data"]["raw_path"]))
    cols_to_drop = [cfg["data"]["id_column"], cfg["target"]["column"]]
    sample = raw.drop(columns=cols_to_drop).sample(n, replace=True, random_state=42)
    sample["TotalCharges"] = pd.to_numeric(sample["TotalCharges"], errors="coerce").fillna(0.0)
    return sample.to_dict(orient="records")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure /predict latency.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--n", type=int, default=300, help="number of requests")
    args = parser.parse_args()

    cfg = load_config()
    payloads = build_payloads(cfg, args.n)

    # A couple of warm-up calls so first-request overhead doesn't skew the stats.
    try:
        for p in payloads[:2]:
            requests.post(args.url, json=p, timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"Could not reach {args.url}. Start the server first:\n"
              f"  uvicorn src.app:app --port 8000", file=sys.stderr)
        return 1

    latencies_ms = []
    wall_start = time.perf_counter()
    for p in payloads:
        t0 = time.perf_counter()
        r = requests.post(args.url, json=p, timeout=5)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
    wall_elapsed = time.perf_counter() - wall_start

    lat = np.array(latencies_ms)
    print(f"Requests:          {args.n}")
    print(f"Avg latency:       {lat.mean():.2f} ms")
    print(f"p50 latency:       {np.percentile(lat, 50):.2f} ms")
    print(f"p95 latency:       {np.percentile(lat, 95):.2f} ms")
    print(f"p99 latency:       {np.percentile(lat, 99):.2f} ms")
    print(f"Max latency:       {lat.max():.2f} ms")
    print(f"Throughput:        {args.n / wall_elapsed:,.0f} req/sec (sequential)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
