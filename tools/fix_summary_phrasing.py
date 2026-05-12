"""Catch the remaining 'triple, as required by the rubric' phrasing in the Project Summary."""

import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "Bank_Distributed_ML_Project.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

changes = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    text = "".join(cell["source"])
    if "triple, as required by the rubric" in text:
        new_text = text.replace(
            "Each chart is\n   paired with an explicit *why this chart / what insight / what business impact* triple, as required by the rubric.",
            "Each chart is\n   paired with a focused narrative covering the chart choice, the insight, and the business implication.",
        )
        if new_text == text:
            # Fallback for other whitespace variants
            import re
            new_text = re.sub(
                r"Each chart is\s+paired with an explicit \*why this chart / what insight / what business impact\* triple, as required by the rubric\.",
                "Each chart is paired with a focused narrative covering the chart choice, the insight, and the business implication.",
                text,
            )
        if new_text != text:
            cell["source"] = new_text.splitlines(keepends=True)
            changes += 1

NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Applied {changes} fix(es)")
