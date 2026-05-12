"""Patch two narrative claims in the ML-model section that the executed run contradicts.

- RF "Yes — deeper trees + more trees gain a few AUC points" → reality: tuning slightly
  *decreased* AUC (0.7400 baseline -> 0.7356 tuned). Rewrite.
- GBT "Tuning typically lifts GBT slightly above RF on tabular data — confirmed below"
  → reality: GBT lost ROC-AUC to RF, won F1. Rewrite.

Idempotent.
"""

import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "Bank_Distributed_ML_Project.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

REPLACEMENTS = [
    (
        "Same `CrossValidator` framework, grid over `numTrees` and `maxDepth` — the two most impactful RF hyperparameters. "
        "Going wider is cheap on Spark because tree training is embarrassingly parallel.\n\n"
        "Yes — deeper trees + more trees gain a few AUC points; we stop at depth 10 to avoid overfitting on the small dataset.",
        "Same `CrossValidator` framework, grid over `numTrees` and `maxDepth` — the two most impactful RF hyperparameters. "
        "Going wider is cheap on Spark because tree training is embarrassingly parallel.\n\n"
        "On the executed run the tuned RF lands at **ROC-AUC 0.7356, F1 0.8472, precision 0.8652, recall 0.8859** — essentially "
        "matching the baseline because the dataset is small and the model is already close to its noise floor. We stop at depth 10 "
        "to avoid overfitting.",
    ),
    (
        "Same `CrossValidator`; grid over `maxDepth` and `maxIter`.\n\n"
        "Tuning typically lifts GBT slightly above RF on tabular data — confirmed below.",
        "Same `CrossValidator`; grid over `maxDepth` and `maxIter`.\n\n"
        "On the executed run the tuned GBT lands at **ROC-AUC 0.6960, F1 0.8579, precision 0.8630, recall 0.8871** — it actually "
        "*loses* ROC-AUC to Random Forest here, but wins on F1 and recall. On a larger dataset GBT typically pulls ahead on AUC too; "
        "on this 4.5k-row sample the variance dominates the gap.",
    ),
]

changes = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    text = "".join(cell["source"])
    new_text = text
    for old, new in REPLACEMENTS:
        if old in new_text:
            new_text = new_text.replace(old, new)
            changes += 1
    if new_text != text:
        cell["source"] = new_text.splitlines(keepends=True)

NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Applied {changes} narrative fix(es)")
