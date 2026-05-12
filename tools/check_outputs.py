"""Quick post-execution check: every code cell has at least one output, no errors."""

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "Bank_Distributed_ML_Project.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Total cells: {len(nb['cells'])}")
print(f"Code cells : {len(code_cells)}")

cells_no_output = []
cells_with_error = []
total_outputs = 0
total_images = 0

for i, c in enumerate(code_cells):
    outs = c.get("outputs", [])
    total_outputs += len(outs)
    if not outs:
        cells_no_output.append(i)
        continue
    for o in outs:
        if o.get("output_type") == "error":
            cells_with_error.append((i, o.get("ename"), o.get("evalue")))
        data = o.get("data", {})
        if "image/png" in data:
            total_images += 1

print(f"Total outputs across code cells: {total_outputs}")
print(f"Total image outputs (charts):     {total_images}")
print(f"Code cells with NO output: {len(cells_no_output)}  -> {cells_no_output}")
print(f"Code cells with ERROR:     {len(cells_with_error)}")
for i, e, v in cells_with_error:
    print(f"  cell index {i}: {e}: {v[:100] if v else ''}")

sys.exit(1 if cells_with_error else 0)
