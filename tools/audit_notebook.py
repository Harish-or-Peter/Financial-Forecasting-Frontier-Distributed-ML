"""Dump every markdown cell in the executed notebook, indexed, so we can audit
text inconsistencies and question-prompt patterns to consolidate.
"""

import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "Bank_Distributed_ML_Project.ipynb"
OUT = Path(__file__).resolve().parent / "_markdown_audit.txt"

nb = json.loads(NB.read_text(encoding="utf-8"))

with OUT.open("w", encoding="utf-8") as f:
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c["source"])
        f.write(f"=== cell {i} (markdown) ===\n")
        f.write(src.rstrip())
        f.write("\n\n")

print(f"Wrote {OUT}")
print(f"Total markdown cells: {sum(1 for c in nb['cells'] if c['cell_type']=='markdown')}")
