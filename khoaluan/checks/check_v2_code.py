#!/usr/bin/env python3
"""Smoke-test cho module khoaluan/code (deliverable task v2-code). KHÔNG train, KHÔNG load model.

Chứng minh:
  - vocab 3-task: 44 diag / 12 rhythm / 19 form (đọc từ scp_statements.csv thật)
  - quy tắc nhãn: rhythm/form giữ presence (likelihood=0 vẫn =1); diag dùng ngưỡng
  - build_sample: 1 ECG -> 3 mẫu hội thoại đúng định dạng
  - weighted_ce: loss hữu hạn trên tensor giả; class_balanced_weights sum=num_classes

Chạy red trước khi viết module (import fail). Exit 0 = xanh.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from khoaluan.code.scp_tasks import load_task_vocabs  # noqa: E402
from khoaluan.code.labels import assign_labels  # noqa: E402
from khoaluan.code.build_dataset import build_sample  # noqa: E402
from khoaluan.code.weighted_ce import (  # noqa: E402
    class_balanced_weights, weighted_ce_loss, IGNORE_INDEX,
)

PASS = []


def _check(name: str, cond: bool, extra: str = "") -> None:
    PASS.append(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")


def main() -> int:
    v = load_task_vocabs()
    _check("vocab_sizes", len(v["diag"]) == 44 and len(v["rhythm"]) == 12 and len(v["form"]) == 19,
           f"{len(v['diag'])}/{len(v['rhythm'])}/{len(v['form'])}")

    # Mẫu chỉ có nhãn rhythm + form (likelihood=0) và một diag dưới ngưỡng.
    scp = {"AFIB": 0.0, "PVC": 0.0, "IMI": 30.0, "NORM": 100.0}
    y_rhythm = assign_labels(scp, "rhythm", v["rhythm"])
    y_form = assign_labels(scp, "form", v["form"])
    y_diag = assign_labels(scp, "diag", v["diag"], diag_threshold=50.0)
    _check("rhythm_presence_kept", y_rhythm[v["rhythm"].index("AFIB")] == 1, "AFIB likelihood=0 vẫn =1")
    _check("form_presence_kept", y_form[v["form"].index("PVC")] == 1, "PVC likelihood=0 vẫn =1")
    _check("diag_threshold_drops_low", y_diag[v["diag"].index("IMI")] == 0, "IMI=30 < 50 -> 0")
    _check("diag_threshold_keeps_high", y_diag[v["diag"].index("NORM")] == 1, "NORM=100 >= 50 -> 1")

    samples = build_sample("12345", "images/12345.png", scp, v, label_format="binary")
    _check("build_3_tasks", len(samples) == 3 and {s["task"] for s in samples} == {"diag", "rhythm", "form"})
    diag_sample = next(s for s in samples if s["task"] == "diag")
    _check("answer_has_blocks",
           "[REASONING]" in diag_sample["conversations"][1]["value"]
           and "[LABELS]" in diag_sample["conversations"][1]["value"]
           and "<image>" in diag_sample["conversations"][0]["value"])

    # class-balanced weights
    counts = torch.tensor([1000.0, 100.0, 10.0])
    w = class_balanced_weights(counts, beta=0.999)
    _check("cb_weights_sum_eq_numclasses", abs(float(w.sum()) - len(counts)) < 1e-3, f"sum={float(w.sum()):.4f}")
    _check("cb_weights_rare_gt_common", float(w[2]) > float(w[0]), "lớp hiếm trọng số cao hơn")

    # weighted CE trên tensor giả: B=1, S=6, V=8. token cuối là token nhãn.
    torch.manual_seed(0)
    B, S, V = 1, 6, 8
    logits = torch.randn(B, S, V, requires_grad=True)
    labels = torch.tensor([[IGNORE_INDEX, 3, 4, 5, 6, 7]])  # 5 token được supervise
    label_mask = torch.zeros(B, S, dtype=torch.bool)
    label_mask[0, 5] = True  # token cuối là token nhãn ("1"/"0")
    label_w = torch.ones(B, S)
    label_w[0, 5] = 2.5
    loss, m = weighted_ce_loss(logits, labels, label_mask, label_w, lambda_prefix=1.0)
    _check("loss_finite", torch.isfinite(loss).item(), f"loss={float(loss):.4f}")
    loss.backward()
    _check("loss_backward_grad", logits.grad is not None and torch.isfinite(logits.grad).all().item())
    _check("metrics_split", int(m["n_label_tok"]) == 1 and int(m["n_prefix_tok"]) == 4,
           f"label={int(m['n_label_tok'])} prefix={int(m['n_prefix_tok'])}")

    ok = all(PASS)
    print(f"\n{'ALL GREEN' if ok else 'HAS RED'} ({sum(PASS)}/{len(PASS)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
