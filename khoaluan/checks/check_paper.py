#!/usr/bin/env python3
"""Kiểm deliverable task paper-draft.

  - check_files: paper.typ + paper.pdf tồn tại (PDF = biên dịch được)
  - check_author: paper đứng tên tác giả
  - check_refs_in_manifest: mọi URL trong refs.bib khớp một entry manifest; >= 12 ref
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "v2" / "paper"
TYP = PAPER_DIR / "paper.typ"
PDF = PAPER_DIR / "paper.pdf"
BIB = PAPER_DIR / "refs.bib"
MANIFEST = ROOT / "v2" / "papers" / "manifest.json"
MIN_REFS = 12
AUTHOR = "Nguyễn Huỳnh Hải Đăng"


def main() -> int:
    ok = True

    if TYP.exists() and PDF.exists() and PDF.stat().st_size > 10000:
        print(f"  PASS  check_files (paper.typ + paper.pdf {PDF.stat().st_size // 1024}KB)")
    else:
        print("  FAIL [no_artifact] thiếu paper.typ hoặc paper.pdf biên dịch")
        ok = False

    typ_text = TYP.read_text() if TYP.exists() else ""
    if AUTHOR in typ_text:
        print("  PASS  check_author")
    else:
        print(f"  FAIL [no_author] không thấy tên tác giả '{AUTHOR}'")
        ok = False

    manifest = json.loads(MANIFEST.read_text())
    man_urls = {p.get("url", "").rstrip("/") for p in manifest.get("papers", [])}
    bib_urls = re.findall(r"url\s*=\s*\{([^}]+)\}", BIB.read_text() if BIB.exists() else "")
    bib_urls = [u.strip().rstrip("/") for u in bib_urls]

    if len(bib_urls) < MIN_REFS:
        print(f"  FAIL [too_few_refs] {len(bib_urls)} < {MIN_REFS}")
        ok = False
    else:
        print(f"  PASS  check_min_refs ({len(bib_urls)} ref)")

    unmatched = [u for u in bib_urls if u not in man_urls]
    if unmatched:
        print(f"  FAIL [no_source] URL không khớp manifest: {unmatched}")
        ok = False
    else:
        print("  PASS  check_refs_in_manifest (mọi ref truy nguồn manifest)")

    print(f"\n{'ALL GREEN' if ok else 'HAS RED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
