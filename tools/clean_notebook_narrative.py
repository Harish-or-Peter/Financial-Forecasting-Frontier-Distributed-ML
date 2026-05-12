"""Surgical narrative cleanup of the executed notebook.

Goals:
1. Remove AlmaBetter-template question-prompt headers ("Why did you pick the chart?",
   "What is/are the insight(s)?", etc.) but keep the answer prose intact.
2. Merge multi-cell Q-A blocks (3 question/answer pairs per chart, 2 per hypothesis test,
   2 per ML model section) into a single flowing narrative cell that reads aloud naturally.
3. Drop template sub-headers like "#### 1. Explain the ML Model..." while keeping the
   narrative below them.
4. Fix two small inconsistencies in the Project Summary + General Guidelines blocks.

Code cells and their outputs are NEVER touched.
Idempotent: safe to re-run; on the second run there are simply no patterns to match.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "Bank_Distributed_ML_Project.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(c):
    return "".join(c["source"]).rstrip()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


# --- Pattern detectors -----------------------------------------------------

# Per-chart question headers (always 3 in a row: Why -> Insight -> Business impact)
RE_CHART_Q = re.compile(
    r"^#####\s+\d+\.\s+(Why did you pick the specific chart|"
    r"What is/are the insight\(s\) found from the chart|"
    r"Will the gained insights help creating)"
)

# Hypothesis-test follow-up questions
RE_HYP_Q = re.compile(r"^#####\s+(Which statistical test have you done|Why did you choose the specific)")

# ML-model follow-up questions
RE_ML_Q = re.compile(r"^#####\s+(Which hyperparameter optimization|Have you seen any improvement)")

# Feature-engineering numbered prompts
RE_FE_Q = re.compile(
    r"^#{4,5}\s+(What all (missing value|outlier|categorical encoding|feature selection)|"
    r"Which all features you found important|Do you think|Which method have you used|"
    r"What data splitting ratio|What technique did you use)"
)

# Numbered template prompts inside hypothesis + ML model sections
# e.g. "#### 1. Explain the ML Model used..."
# "#### 1. State Your research hypothesis..."
# "#### 2. Perform an appropriate statistical test."
# "#### 2. Cross-Validation & Hyperparameter Tuning"
# "#### 3. Explain each evaluation metric's indication..."
RE_NUMBERED_PROMPT = re.compile(
    r"^#{3,5}\s+\d+\.\s+(Explain the ML Model used|State Your research hypothesis|"
    r"Perform an appropriate statistical test|Cross-?Validation\s*&\s*Hyperparameter Tuning|"
    r"Explain each evaluation metric|Which Evaluation metrics|Which ML model did you choose|"
    r"Explain the model which you have used|Save the best performing|Again Load the saved)"
)

# Generic standalone question prompt headers (e.g. "### What did you know about your dataset?")
RE_GENERIC_Q_HEADER = re.compile(r"^#{2,5}\s+.+\?\s*\*?\*?$")

# Keep these even though they end with '?'
KEEP_QUESTION_HEADERS = {
    "#### **Define Your Business Objective?**",  # serves as a section title
}


def is_md(c):
    return c["cell_type"] == "markdown"


def collapse_block(start_i, q_matcher, sep="\n\n"):
    """Walk forward from start_i collecting alternating Q-A pairs, returning
    (merged_narrative, next_index)."""
    answers = []
    j = start_i
    while j < len(cells):
        c = cells[j]
        if not is_md(c) or not q_matcher.match(src(c)):
            break
        j += 1  # skip the question
        if j < len(cells) and is_md(cells[j]):
            ans = src(cells[j])
            if ans:
                answers.append(ans)
            j += 1
        else:
            break
    return sep.join(answers), j


# --- Pass 1: walk and transform --------------------------------------------

new_cells = []
i = 0
n_charts = 0
n_hyp = 0
n_ml = 0
n_fe = 0
n_numbered = 0
n_generic = 0

while i < len(cells):
    c = cells[i]

    if is_md(c):
        s = src(c)

        # Chart Q-A block (3 pairs typical, 2 for charts 14-17)
        if RE_CHART_Q.match(s):
            text, next_i = collapse_block(i, RE_CHART_Q, sep="\n\n")
            if text:
                new_cells.append(md(text))
            n_charts += 1
            i = next_i
            continue

        # Hypothesis Q-A block (2 pairs)
        if RE_HYP_Q.match(s):
            text, next_i = collapse_block(i, RE_HYP_Q, sep="\n\n")
            if text:
                # Light prefixing so the merged paragraph reads naturally
                new_cells.append(md(text))
            n_hyp += 1
            i = next_i
            continue

        # ML model Q-A block (2 pairs)
        if RE_ML_Q.match(s):
            text, next_i = collapse_block(i, RE_ML_Q, sep="\n\n")
            if text:
                new_cells.append(md(text))
            n_ml += 1
            i = next_i
            continue

        # Feature-engineering single Q-A
        if RE_FE_Q.match(s):
            i += 1
            if i < len(cells) and is_md(cells[i]):
                ans = src(cells[i])
                if ans and ans.lower() != "answer here.":
                    new_cells.append(md(ans))
                i += 1
            n_fe += 1
            continue

        # Numbered AlmaBetter template prompts (drop header, keep answer)
        if RE_NUMBERED_PROMPT.match(s):
            # The next cell may be code (for "Perform statistical test" cells) or md
            i += 1
            n_numbered += 1
            continue  # do not emit the heading; subsequent cell(s) carry the content

        # Generic "### something?" headers (drop header, keep next md as-is)
        if (
            RE_GENERIC_Q_HEADER.match(s)
            and s not in KEEP_QUESTION_HEADERS
            and not s.startswith("# **GitHub Link -**")  # don't mismatch
        ):
            i += 1
            n_generic += 1
            continue

    new_cells.append(c)
    i += 1


# --- Pass 2: fix narrative inconsistencies in specific cells ---------------

INCONSISTENCY_REPLACEMENTS = [
    (
        # Project Summary: "fifteen-plus" -> "seventeen"; "(~11% positive)" -> "(~11.5% positive)"
        "fifteen-plus visualisations following the Univariate / Bivariate / Multivariate (UBM) rule",
        "seventeen visualisations following the Univariate / Bivariate / Multivariate (UBM) rule",
    ),
    (
        "**imbalanced banking outcome** (~11% positive class)",
        "**imbalanced banking outcome** (~11.5% positive class)",
    ),
    (
        # Project Summary: drop the rubric-y "triple, as required by the rubric" phrasing
        "Each chart is\npaired with an explicit *why this chart / what insight / what business impact* triple, as required by the rubric.",
        "Each chart is\npaired with a focused narrative covering the chart choice, the insight, and the business implication.",
    ),
    (
        # General Guidelines cell: replace the "three mandatory markdown answers" bullet
        "For each chart, three mandatory markdown answers are provided: *why this chart? · what insight? · positive/negative business impact?*",
        "For each chart, a single concise narrative paragraph below the figure covers the chart choice, the insight, and the business implication.",
    ),
]

n_inconsistencies = 0
for cell in new_cells:
    if cell["cell_type"] != "markdown":
        continue
    text = "".join(cell["source"])
    new_text = text
    for old, new in INCONSISTENCY_REPLACEMENTS:
        if old in new_text:
            new_text = new_text.replace(old, new)
            n_inconsistencies += 1
    if new_text != text:
        cell["source"] = new_text.splitlines(keepends=True)


# --- Write back -------------------------------------------------------------

print(f"Before: {len(cells)} cells")
print(f"After:  {len(new_cells)} cells (removed {len(cells)-len(new_cells)})")
print(f"  chart Q-A blocks merged   : {n_charts}")
print(f"  hypothesis Q-A merged     : {n_hyp}")
print(f"  ML model Q-A merged       : {n_ml}")
print(f"  feature-eng Q-A merged    : {n_fe}")
print(f"  numbered prompts dropped  : {n_numbered}")
print(f"  generic '?' headers dropped: {n_generic}")
print(f"  text inconsistencies fixed : {n_inconsistencies}")

nb["cells"] = new_cells
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
