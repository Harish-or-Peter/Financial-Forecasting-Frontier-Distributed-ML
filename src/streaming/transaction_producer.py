"""Synthetic banking-transaction producer for the Spark-Streaming demo.

Generates realistic transaction records and writes them as line-delimited JSON
into a watched directory that the streaming consumer reads from. Accounts are
seeded from the real bank.csv so that job-category aggregations downstream are
representative of the customer base.

Usage
-----
    # Generate 10 batches of 60 transactions each, 3 s apart (default):
    python src/streaming/transaction_producer.py

    # Continuous mode (Ctrl-C to stop):
    python src/streaming/transaction_producer.py --continuous --interval 2

    # Custom path and batch size:
    python src/streaming/transaction_producer.py --out stream_input --batches 20 --size 80
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def load_accounts(data_csv: Path, n_accounts: int = 50, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Seed a population of accounts from the real bank dataset."""
    df = pd.read_csv(data_csv)
    sample = df[["job", "age", "balance"]].sample(n_accounts, random_state=seed).reset_index(drop=True)
    sample["account_id"] = [f"ACC-{i:04d}" for i in range(len(sample))]
    return sample


def make_transactions(accounts: pd.DataFrame, n: int, suspicious_rate: float = 0.05) -> list[dict]:
    """Generate `n` plausible transactions, with a small fraction marked as high-value outliers.

    Amounts are drawn from a log-normal distribution (mean=4.5, sigma=1.0) which approximates
    real retail-bank transaction sizes (lots of small swipes, a few large transfers). The
    ``suspicious_rate`` fraction is multiplied 20–60x to simulate fraud-suspect outliers that
    the windowed aggregation will flag.
    """
    rows: list[dict] = []
    now = datetime.now()
    for _ in range(n):
        acc = accounts.sample(1).iloc[0]
        amount = float(round(np.random.lognormal(mean=4.5, sigma=1.0), 2))
        if random.random() < suspicious_rate:
            amount *= random.uniform(20, 60)
        rows.append(
            {
                "txn_id": f"T{int(time.time() * 1000)}-{random.randint(1000, 9999)}",
                "account_id": acc["account_id"],
                "job": acc["job"],
                "amount": round(amount, 2),
                "currency": "EUR",
                "txn_time": (now - timedelta(seconds=random.randint(0, 60))).isoformat(),
            }
        )
    return rows


def write_batch(out_dir: Path, batch_id: int, rows: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"batch_{batch_id:06d}.json"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--data", default="data/bank.csv", type=Path,
                   help="Source bank.csv to seed account population from")
    p.add_argument("--out", default="stream_input", type=Path,
                   help="Directory to drop streaming batches into")
    p.add_argument("--batches", default=10, type=int, help="Number of batches to produce")
    p.add_argument("--size", default=60, type=int, help="Transactions per batch")
    p.add_argument("--interval", default=3.0, type=float, help="Seconds between batches")
    p.add_argument("--continuous", action="store_true",
                   help="Run forever instead of stopping after --batches")
    p.add_argument("--accounts", default=50, type=int, help="How many seed accounts")
    p.add_argument("--seed", default=RANDOM_SEED, type=int)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    accounts = load_accounts(args.data, n_accounts=args.accounts, seed=args.seed)
    print(f"Seeded {len(accounts)} accounts from {args.data}")
    print(f"Writing batches of {args.size} txn(s) to {args.out} every {args.interval}s")

    i = 0
    try:
        while True:
            batch = make_transactions(accounts, args.size)
            path = write_batch(args.out, i, batch)
            print(f"[{i:>4}] wrote {path.name}  ({len(batch)} txn)")
            i += 1
            if not args.continuous and i >= args.batches:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"Done. Total batches written: {i}")


if __name__ == "__main__":
    main()
