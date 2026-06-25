"""Gộp 44 mã diagnostic -> 5 siêu lớp PTB-XL (NORM/MI/STTC/CD/HYP) để SO TRỰC TIẾP với PULSE.

KHÔNG chạy model. Chỉ hậu xử lý file dump của eval_v2.py (`--out`, task 'diag':
{codes, scores, gt}). Mỗi mã diag thuộc đúng 1 siêu lớp theo cột diagnostic_class
của scp_statements.csv chính thức (NORM=1, MI=14, STTC=13, CD=11, HYP=5 = 44 mã).

Gộp:
  - score siêu lớp s = MAX score các mã con (logic "bất kỳ mã con dương => lớp dương",
    giữ tính không-phụ-thuộc-ngưỡng cho macro-AUC).
  - gt siêu lớp s    = 1 nếu BẤT KỲ mã con nào gt=1 (đúng định nghĩa siêu lớp PTB-XL).

Báo macro-AUC (không ngưỡng), macro-F1 (ngưỡng/lớp: val-frozen nếu có --val-dump,
ngược lại self-tuned trên chính tập test — ghi rõ), và Hamming loss (%) — đúng 3 chỉ số
PULSE công bố cho Abnormality Detection trên PTB-XL Super (AUC 82.4 / F1 74.8 / HL 11.0).

Chạy (CPU, sau khi đã có dump từ eval_v2.py):
  python -m khoaluan.code.aggregate_super5 \
      --test-dump eval_test_ours.json \
      [--val-dump eval_val_ours.json] \
      [--csv ptb-xl-.../scp_statements.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss

SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
PULSE_REF = {"macro_auc": 0.824, "macro_f1": 0.748, "hamming": 0.110}  # PULSE-7B, PTB-XL Super


def _resolve_csv(path: str | None) -> Path:
    if path:
        return Path(path)
    env = os.environ.get("ECGLLAVA_DATASET_ROOT")
    if env and (Path(env) / "scp_statements.csv").exists():
        return Path(env) / "scp_statements.csv"
    repo = Path(__file__).resolve().parents[2]
    local = repo / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" / "scp_statements.csv"
    if local.exists():
        return local
    raise FileNotFoundError("Không tìm thấy scp_statements.csv. Truyền --csv hoặc đặt ECGLLAVA_DATASET_ROOT.")


def code_to_super(csv_path: str | None) -> dict[str, str]:
    """{mã diag -> siêu lớp} từ scp_statements.csv (chỉ các dòng diagnostic=1.0)."""
    rows = list(csv.DictReader(open(_resolve_csv(csv_path))))
    return {r[""]: r["diagnostic_class"] for r in rows if r.get("diagnostic") == "1.0"}


def aggregate(dump_path: str, c2s: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Đọc task 'diag' của dump -> (scores[N,5], gt[N,5]) theo thứ tự SUPERCLASSES."""
    d = json.loads(Path(dump_path).read_text())["diag"]
    codes, scores, gt = d["codes"], np.array(d["scores"]), np.array(d["gt"])
    # cột mã con cho từng siêu lớp
    cols = {s: [i for i, c in enumerate(codes) if c2s.get(c) == s] for s in SUPERCLASSES}
    n = scores.shape[0]
    s_score = np.zeros((n, len(SUPERCLASSES)))
    s_gt = np.zeros((n, len(SUPERCLASSES)), dtype=int)
    for j, s in enumerate(SUPERCLASSES):
        idx = cols[s]
        s_score[:, j] = scores[:, idx].max(axis=1)        # bất kỳ mã con dương
        s_gt[:, j] = (gt[:, idx].sum(axis=1) > 0).astype(int)
    return s_score, s_gt


def tune_thresholds(scores: np.ndarray, gt: np.ndarray) -> list[float]:
    """Ngưỡng tối đa F1 RIÊNG từng siêu lớp (chuẩn multi-label mất cân bằng)."""
    thr = []
    for j in range(scores.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.02, 0.8, 40):
            f1 = f1_score(gt[:, j], (scores[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thr.append(round(best_t, 3))
    return thr


def metrics(scores: np.ndarray, gt: np.ndarray, thr: list[float]) -> dict:
    aucs, f1s = {}, {}
    for j, s in enumerate(SUPERCLASSES):
        col = gt[:, j]
        if col.min() == 0 and col.max() == 1:
            aucs[s] = float(roc_auc_score(col, scores[:, j]))
            f1s[s] = float(f1_score(col, (scores[:, j] >= thr[j]).astype(int), zero_division=0))
    pred = np.stack([(scores[:, j] >= thr[j]).astype(int) for j in range(len(SUPERCLASSES))], axis=1)
    return {
        "macro_auc": float(np.mean(list(aucs.values()))) if aucs else float("nan"),
        "macro_f1": float(np.mean(list(f1s.values()))) if f1s else float("nan"),
        "hamming": float(hamming_loss(gt, pred)),
        "per_class_auc": aucs, "per_class_f1": f1s, "thresholds": thr,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Gộp 44 mã diag -> 5 siêu lớp PTB-XL, so với PULSE")
    ap.add_argument("--test-dump", required=True, help="dump eval_v2 trên fold-10 (test)")
    ap.add_argument("--val-dump", default="", help="dump eval_v2 trên fold-9 (val) để đóng băng ngưỡng F1")
    ap.add_argument("--csv", default="", help="scp_statements.csv (mặc định tự dò trong repo)")
    ap.add_argument("--label", default="model", help="nhãn in ra (vd 'PULSE-base' / 'Ours')")
    ap.add_argument("--out", default="", help="ghi metrics ra JSON (tùy chọn)")
    args = ap.parse_args()

    c2s = code_to_super(args.csv or None)
    test_s, test_gt = aggregate(args.test_dump, c2s)

    if args.val_dump:
        val_s, val_gt = aggregate(args.val_dump, c2s)
        thr = tune_thresholds(val_s, val_gt)
        thr_mode = "val-frozen (ngưỡng dò trên fold-9, áp lên fold-10)"
    else:
        thr = tune_thresholds(test_s, test_gt)
        thr_mode = "self-tuned (dò trên chính test — F1 lạc quan, chỉ tham khảo)"

    m = metrics(test_s, test_gt, thr)
    print(f"\n=== {args.label}  |  PTB-XL Super (5 siêu lớp)  |  N={test_gt.shape[0]} ===")
    print(f"  ngưỡng F1: {thr_mode}")
    print(f"  {'lớp':6s} {'AUC':>7s} {'F1':>7s}  (n+ trong test)")
    for s in SUPERCLASSES:
        npos = int(test_gt[:, SUPERCLASSES.index(s)].sum())
        a = m["per_class_auc"].get(s, float('nan'))
        f = m["per_class_f1"].get(s, float('nan'))
        print(f"  {s:6s} {a:7.3f} {f:7.3f}  ({npos})")
    print(f"  {'-'*34}")
    print(f"  macro-AUC : {m['macro_auc']:.3f}   (PULSE-7B: {PULSE_REF['macro_auc']:.3f})")
    print(f"  macro-F1  : {m['macro_f1']:.3f}   (PULSE-7B: {PULSE_REF['macro_f1']:.3f})")
    print(f"  Hamming   : {m['hamming']*100:.1f}%  (PULSE-7B: {PULSE_REF['hamming']*100:.1f}%)")

    if args.out:
        Path(args.out).write_text(json.dumps(m, indent=2))
        print(f"\nDumped -> {args.out}")


if __name__ == "__main__":
    main()
