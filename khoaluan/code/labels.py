"""Gán nhãn nhị phân từ scp_codes của PTB-XL theo task (method-spec §3).

diag : y[k]=1 nếu k có mặt và likelihood >= threshold  (mode 'threshold', mặc định 50)
rhythm/form : y[k]=1 nếu k có mặt  (mode 'presence', vì PTB-XL gán likelihood=0)

Quy tắc tách này tránh lỗi bản cũ: ngưỡng likelihood>=50 áp đồng loạt sẽ loại sạch
form/rhythm (likelihood=0). Xem [Wagner 2020].
"""
from __future__ import annotations

from .scp_tasks import DEFAULT_DIAG_THRESHOLD, LABEL_RULE


def assign_labels(
    scp_codes: dict[str, float],
    task: str,
    vocab: list[str],
    diag_threshold: float = DEFAULT_DIAG_THRESHOLD,
) -> list[int]:
    """Trả vector nhị phân dài len(vocab) theo thứ tự vocab."""
    rule = LABEL_RULE[task]
    out = []
    for code in vocab:
        if code not in scp_codes:
            out.append(0)
        elif rule == "threshold":
            out.append(1 if float(scp_codes[code]) >= diag_threshold else 0)
        else:  # presence
            out.append(1)
    return out
