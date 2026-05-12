"""Patch a handful of narrative markdown cells in the EXECUTED notebook to reflect
the actual run's metrics (Random Forest won ROC-AUC, not GBT). This keeps all the
cell outputs (110 outputs, 27 charts, model metrics, query results) intact and only
rewrites prose. Idempotent — safe to re-run.
"""

import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "Bank_Distributed_ML_Project.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

REPLACEMENTS = [
    (
        'plt.title("Top-15 Feature Importances — best GBT model")',
        'plt.title("Top-15 Feature Importances — best tree model")',
    ),
    (
        "1. **Deploy the tuned GBT pipeline as a daily lead-scoring batch job.** Score every client overnight; export the top-K to the call centre's\n   queue ranked by predicted probability. This converts the 11.5% blanket conversion rate into a precision-targeted call list — back-of-envelope\n   our model should push effective conversion on the top-decile leads to **>30%**.",
        "1. **Deploy the tuned tree-ensemble pipeline as a daily lead-scoring batch job.** On the executed run Random Forest wins ROC-AUC (0.7356) and\n   is the model the script persists; GBT wins F1 (0.8579) and would be the right pick under a fixed call-centre capacity. Score every client\n   overnight; export the top-K to the call centre's queue ranked by predicted probability. This converts the 11.5% blanket conversion rate\n   into a precision-targeted call list — back-of-envelope the top-decile leads should convert at multiples of the base rate.",
    ),
    (
        "The headline business finding is straightforward: the bank can materially lift marketing ROI by **scoring leads pre-call with a tuned\nGradient-Boosted Trees model**, **re-allocating call-centre capacity to the high-conversion months**, and **re-targeting prior-success\nclients on a strict cadence**.",
        "The headline business finding is straightforward: the bank can materially lift marketing ROI by **scoring leads pre-call with a tuned\ntree-ensemble** (Random Forest is the ROC-AUC winner on this run; GBT wins F1), **re-allocating call-centre capacity to the high-conversion\nmonths**, and **re-targeting prior-success clients on a strict cadence**.",
    ),
]

changes = 0
for cell in nb["cells"]:
    src_str = "".join(cell["source"])
    new_str = src_str
    for old, new in REPLACEMENTS:
        if old in new_str:
            new_str = new_str.replace(old, new)
    if new_str != src_str:
        # Re-split preserving newlines like nbformat does
        lines = new_str.splitlines(keepends=True)
        cell["source"] = lines
        changes += 1

NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Patched {changes} cell(s) in {NB}")
