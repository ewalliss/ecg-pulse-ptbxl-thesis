"""Phân hoạch 71 mã SCP-ECG của PTB-XL thành 3 task (method-spec §3).

Đọc TRỰC TIẾP từ scp_statements.csv chính thức (cột diagnostic/form/rhythm) —
KHÔNG hardcode danh sách mã (tránh sai sót). Cho 44 diagnostic / 19 form / 12 rhythm,
trùng diag&form = {DIG, LNGQT, NDT, NST_} (đúng [Wagner 2020]).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

TASKS = ("diag", "rhythm", "form")
_CSV_COL = {"diag": "diagnostic", "rhythm": "rhythm", "form": "form"}

# Quy tắc gán nhãn theo task (method-spec §3): diag dùng ngưỡng likelihood,
# rhythm/form dùng presence (likelihood=0 theo thiết kế PTB-XL).
LABEL_RULE = {"diag": "threshold", "rhythm": "presence", "form": "presence"}
DEFAULT_DIAG_THRESHOLD = 50.0


def _resolve_csv(path: str | os.PathLike | None) -> Path:
    if path:
        return Path(path)
    env = os.environ.get("ECGLLAVA_DATASET_ROOT")
    if env:
        cand = Path(env) / "scp_statements.csv"
        if cand.exists():
            return cand
    # Bản sao trong repo (Mac/local).
    repo = Path(__file__).resolve().parents[2]
    local = repo / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" / "scp_statements.csv"
    if local.exists():
        return local
    raise FileNotFoundError(
        "Không tìm thấy scp_statements.csv. Đặt ECGLLAVA_DATASET_ROOT hoặc truyền path."
    )


def load_task_vocabs(csv_path: str | os.PathLike | None = None) -> dict[str, list[str]]:
    """Trả {'diag': [...44], 'rhythm': [...12], 'form': [...19]} theo thứ tự CSV."""
    rows = list(csv.DictReader(open(_resolve_csv(csv_path))))
    vocabs = {t: [] for t in TASKS}
    for r in rows:
        code = r[""]  # cột đầu không tên = mã SCP
        for t in TASKS:
            if r.get(_CSV_COL[t], "") == "1.0":
                vocabs[t].append(code)
    return vocabs


if __name__ == "__main__":
    v = load_task_vocabs()
    for t in TASKS:
        print(f"{t:6s} ({len(v[t])}): {v[t]}")
