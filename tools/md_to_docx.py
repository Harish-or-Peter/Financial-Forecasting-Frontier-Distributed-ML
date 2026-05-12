"""Convert docs/TECHNICAL_DOCUMENT.md to docs/TECHNICAL_DOCUMENT.docx.

Word-format output drops cleanly into Google Docs (File -> Open -> Upload).
The same approach handles the other markdown docs if needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parent.parent

DOCS = [
    ("docs/TECHNICAL_DOCUMENT.md", "docs/TECHNICAL_DOCUMENT.docx"),
    ("docs/REFLECTIVE_SUMMARY.md", "docs/REFLECTIVE_SUMMARY.docx"),
    ("docs/PROJECT_PLAN.md", "docs/PROJECT_PLAN.docx"),
    ("README.md", "docs/README.docx"),
    ("video/VIDEO_SCRIPT.md", "docs/VIDEO_SCRIPT.docx"),
]


def convert(src_rel: str, dst_rel: str) -> None:
    src = ROOT / src_rel
    dst = ROOT / dst_rel
    if not src.exists():
        print(f"  skip (missing): {src_rel}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    pypandoc.convert_file(
        str(src),
        to="docx",
        outputfile=str(dst),
        format="gfm",  # GitHub-flavored markdown (tables, fenced code, task lists)
        extra_args=[
            "--standalone",
            "--toc",                 # auto-generate table of contents
            "--toc-depth=3",
        ],
    )
    size = dst.stat().st_size
    print(f"  {src_rel} -> {dst_rel}   ({size/1024:.1f} KB)")


targets = sys.argv[1:] if len(sys.argv) > 1 else [d[0] for d in DOCS]
mapping = dict(DOCS)
for t in targets:
    if t in mapping:
        convert(t, mapping[t])
    else:
        # Single-file mode: <src.md> -> sibling .docx
        src = Path(t)
        dst = src.with_suffix(".docx")
        convert(str(src.relative_to(ROOT)), str(dst.relative_to(ROOT)))
