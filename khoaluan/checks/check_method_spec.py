#!/usr/bin/env python3
"""Kiểm tra ngữ nghĩa cho v2/method-spec.md (contract của task v2-method-spec).

Chạy RED trước khi viết spec (file chưa tồn tại). 4 check map 1-1 với 4 scenario:
  - check_sections_present        : đủ 7 mục bắt buộc của contract
  - check_every_citation_in_manifest: mọi [ref] khớp một key trong khoaluan/papers/manifest.json
  - check_no_core_reference       : không có chỉ thị sửa source/LLaVA
  - check_loss_not_asl            : mục loss là CE + class-balanced, không định nghĩa ASL trên token logits

Exit 0 = tất cả xanh; exit 1 = có check đỏ (in mã lỗi enum).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "v2" / "method-spec.md"
MANIFEST = ROOT / "v2" / "papers" / "manifest.json"

REQUIRED_SECTIONS = [
    "Notation",
    "Kiến trúc v2",
    "Task decomposition",
    "Định dạng I/O",
    "Hàm loss",
    "trích điểm",   # "Giao thức trích điểm & đánh giá"
    "Tham chiếu",
]


def _fail(code: str, msg: str) -> None:
    print(f"  FAIL [{code}] {msg}")


def check_sections_present(text: str) -> bool:
    ok = True
    for s in REQUIRED_SECTIONS:
        if s.lower() not in text.lower():
            _fail("missing_section", f"thiếu mục bắt buộc: '{s}'")
            ok = False
    return ok


def check_every_citation_in_manifest(text: str) -> bool:
    # Trích các khoá trích dẫn dạng [Key 2020] / [Key et al. 2021].
    cites = set(re.findall(r"\[([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?\s+\d{4}[a-z]?)\]", text))
    if not MANIFEST.exists():
        _fail("no_source", f"manifest chưa tồn tại: {MANIFEST}")
        return False
    manifest = json.loads(MANIFEST.read_text())
    keys = {e.get("cite_key", "").lower() for e in manifest.get("papers", [])}
    # Cũng cho khớp theo first-author+year rời rạc.
    author_year = {(e.get("first_author", "").lower(), str(e.get("year", ""))) for e in manifest.get("papers", [])}
    ok = True
    for c in sorted(cites):
        m = re.match(r"([A-Za-z\-]+).*?(\d{4})", c)
        ay = (m.group(1).lower(), m.group(2)) if m else (c.lower(), "")
        if c.lower() not in keys and ay not in author_year:
            _fail("no_source", f"trích dẫn không khớp manifest: [{c}]")
            ok = False
    return ok


_NEG = ("không", "khong", "no ", "not ", "never", "read-only", "đông cứng", "frozen")


def _line_has_negation(line: str) -> bool:
    low = line.lower()
    return any(n in low for n in _NEG)


def check_no_core_reference(text: str) -> bool:
    # Cấm chỉ thị KHẲNG ĐỊNH sửa source/LLaVA. Mention bị phủ định ("KHÔNG sửa...",
    # "read-only") là hợp lệ — invariant của dự án chính là cấm sửa core.
    ok = True
    for line in text.splitlines():
        if "source/llava" in line.lower() and re.search(r"(sửa|modify|edit|patch|thay đổi)", line, re.IGNORECASE):
            if not _line_has_negation(line):
                _fail("core_touched", f"chỉ thị sửa core không phủ định: {line.strip()[:80]}")
                ok = False
    return ok


def check_loss_not_asl(text: str) -> bool:
    low = text.lower()
    has_ce = "cross-entropy" in low or "cross entropy" in low or "ce next-token" in low
    has_cb = "class-balanced" in low or "class balanced" in low or "(1−β" in text or "(1-beta" in low
    if not (has_ce and has_cb):
        _fail("loss_math_invalid", "mục loss thiếu 'cross-entropy' và/hoặc 'class-balanced'")
        return False
    # Không được định nghĩa ASL như hàm loss CHÍNH. Câu phủ định ("KHÔNG dùng ASL",
    # "bỏ ASL") là hợp lệ và được KHUYẾN KHÍCH. Chỉ flag chỉ thị khẳng định dùng ASL.
    for line in text.splitlines():
        if re.search(r"(dùng|sử dụng|use|áp dụng|định nghĩa)\s+asl", line, re.IGNORECASE):
            if not _line_has_negation(line):
                _fail("loss_math_invalid", f"chỉ thị dùng ASL làm loss (không phủ định): {line.strip()[:80]}")
                return False
    # Phải có tuyên bố tường minh bỏ ASL.
    if not re.search(r"(bỏ|không\s+dùng|khong\s+dung)\s+asl", low):
        _fail("loss_math_invalid", "thiếu tuyên bố tường minh BỎ ASL")
        return False
    return True


def main() -> int:
    if not SPEC.exists():
        print(f"RED: {SPEC} chưa tồn tại — build chưa chạy.")
        return 1
    text = SPEC.read_text()
    checks = [
        ("check_sections_present", check_sections_present(text)),
        ("check_every_citation_in_manifest", check_every_citation_in_manifest(text)),
        ("check_no_core_reference", check_no_core_reference(text)),
        ("check_loss_not_asl", check_loss_not_asl(text)),
    ]
    print()
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    allok = all(ok for _, ok in checks)
    print(f"\n{'ALL GREEN' if allok else 'HAS RED'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
