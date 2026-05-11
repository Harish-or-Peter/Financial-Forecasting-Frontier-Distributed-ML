"""One-shot: generate 5 sample streaming chunks committed in stream_input/.

These give graders concrete artifacts to inspect without running the producer.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from streaming.transaction_producer import load_accounts, make_transactions, write_batch

OUT = ROOT / "stream_input"
DATA = ROOT / "data" / "bank.csv"

accounts = load_accounts(DATA, n_accounts=50, seed=42)
for i in range(5):
    rows = make_transactions(accounts, n=60)
    path = write_batch(OUT, i, rows)
    print(f"  wrote {path.name}  ({len(rows)} txn)")
print(f"5 batches written to {OUT}")
