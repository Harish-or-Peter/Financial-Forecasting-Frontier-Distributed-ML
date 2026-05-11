"""Static validation of the generated notebook.

Cannot execute Spark code locally (no Java), but we can verify:
- The .ipynb is structurally valid JSON.
- Every code cell's source is valid Python (compiles cleanly).
- The bank.csv loads with pandas as a sanity check on column names used in code.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "Bank_Distributed_ML_Project.ipynb"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bank.csv"

errors: list[str] = []
warnings_: list[str] = []

print(f"Validating: {NB_PATH}")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# --- Structural checks ------------------------------------------------------
assert "cells" in nb, "No cells key"
n_cells = len(nb["cells"])
n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
print(f"  cells={n_cells}   markdown={n_md}   code={n_code}")

# --- Syntax check each code cell -------------------------------------------
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append(f"Cell {i} SyntaxError: {e.msg} (line {e.lineno})")
        # show snippet
        lines = src.splitlines()
        lo = max(0, (e.lineno or 1) - 2)
        hi = min(len(lines), (e.lineno or 1) + 2)
        for j in range(lo, hi):
            print(f"    {j+1:>3}: {lines[j]}")

# --- Data sanity check -----------------------------------------------------
print(f"\nReading {DATA_PATH} with pandas to verify schema match...")
df = pd.read_csv(DATA_PATH)
print(f"  shape={df.shape}")
print(f"  cols ={list(df.columns)}")

expected_cols = ["age","job","marital","education","default","balance","housing","loan",
                 "contact","day","month","duration","campaign","pdays","previous",
                 "poutcome","y"]
missing = set(expected_cols) - set(df.columns)
extra   = set(df.columns) - set(expected_cols)
if missing:
    errors.append(f"Missing expected columns: {missing}")
if extra:
    warnings_.append(f"Unexpected columns: {extra}")

# unique value sanity
print(f"  y value_counts: {df['y'].value_counts().to_dict()}")
print(f"  pdays minimum (should be -1): {df['pdays'].min()}")
print(f"  unknown rows in job: {(df['job']=='unknown').sum()}")

# --- Report ----------------------------------------------------------------
print()
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors: print("  -", e)
if warnings_:
    print(f"WARNINGS ({len(warnings_)}):")
    for w in warnings_: print("  -", w)
if not errors and not warnings_:
    print("All checks passed.")
sys.exit(1 if errors else 0)
