#!/usr/bin/env python3
"""Kiểm tra deliverable của task fetch-references.

  - check_manifest_valid : manifest.json parse được, mỗi entry đủ trường bắt buộc
  - check_min_pdfs       : >= 12 PDF thật trong khoaluan/papers/ (mỗi PDF > 20KB)
  - check_pdf_match_manifest: mọi entry có 'pdf' không null thì file tồn tại

Exit 0 = xanh; 1 = đỏ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPERS = ROOT / "v2" / "papers"
MANIFEST = PAPERS / "manifest.json"
MIN_PDFS = 12
REQUIRED = ("cite_key", "first_author", "year", "title", "url", "key_claims")


def main() -> int:
    ok = True
    if not MANIFEST.exists():
        print(f"  FAIL [no_manifest] {MANIFEST} không tồn tại")
        return 1
    manifest = json.loads(MANIFEST.read_text())
    papers = manifest.get("papers", [])

    # check_manifest_valid
    bad = [p.get("cite_key", "?") for p in papers if not all(p.get(k) for k in REQUIRED)]
    if bad:
        print(f"  FAIL [manifest_invalid] entry thiếu trường: {bad}")
        ok = False
    else:
        print(f"  PASS  check_manifest_valid ({len(papers)} entry)")

    # check_min_pdfs
    pdfs = [p for p in PAPERS.glob("*.pdf") if p.stat().st_size > 20000]
    if len(pdfs) < MIN_PDFS:
        print(f"  FAIL [too_few_pdfs] {len(pdfs)} < {MIN_PDFS}")
        ok = False
    else:
        print(f"  PASS  check_min_pdfs ({len(pdfs)} PDF >20KB)")

    # check_pdf_match_manifest
    missing = [p["pdf"] for p in papers if p.get("pdf") and not (PAPERS / p["pdf"]).exists()]
    if missing:
        print(f"  FAIL [pdf_missing] khai báo trong manifest nhưng thiếu file: {missing}")
        ok = False
    else:
        print("  PASS  check_pdf_match_manifest")

    print(f"\n{'ALL GREEN' if ok else 'HAS RED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
