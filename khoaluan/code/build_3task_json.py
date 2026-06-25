"""Dựng dữ liệu 3-task cho v2 TỪ ptbxl_database.csv gốc (method-spec §3-4).

Khác stage2 cũ (chỉ 44 diag, đã nhị phân, mất rhythm/form): ở đây đọc thẳng
scp_codes (dict likelihood) từ PTB-XL nên dựng được CẢ 3 task với quy tắc đúng:
  diag   : likelihood >= threshold
  rhythm : presence (likelihood=0)
  form   : presence

Mỗi ECG -> 3 mẫu hội thoại (diag/rhythm/form). Tái dùng ảnh đã render
(images/{ecg_id:05d}.png). Split theo strat_fold: train=1..8, val=9, test=10.

Chạy trên máy có ảnh + csv (Windows):
  python -m khoaluan.code.build_3task_json \
      --db .../ptbxl_database.csv --images .../pulse_ptbxl_stage2/images \
      --out .../pulse_ptbxl_stage2  [--oversample-path 3] [--no-image-check] [--limit N]
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from .build_dataset import build_sample
from .scp_tasks import TASKS, load_task_vocabs

FOLD_SPLIT = {"train": set(range(1, 9)), "val": {9}, "test": {10}}


def _read_rows(db_csv: Path):
    # PTB-XL load chuẩn: pandas + ast.literal_eval(scp_codes). CSV có dấu phẩy
    # trong dict scp_codes nên csv thuần parse sai; pandas xử lý đúng.
    import pandas as pd
    # skipinitialspace: CSV PTB-XL có khoảng trắng trước dấu " (', "{...}"') nên
    # field quoted chứa dấu phẩy (dict scp_codes) bị tách sai nếu không bật cờ này.
    df = pd.read_csv(db_csv, skipinitialspace=True)
    df.columns = df.columns.str.strip()  # tên cột PTB-XL có khoảng trắng thừa
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    for _, r in df.iterrows():
        scp = {str(k): float(v) for k, v in r["scp_codes"].items()}
        yield int(r["ecg_id"]), int(r["strat_fold"]), scp


def _is_pathological(scp: dict[str, float]) -> bool:
    """Record có bệnh (không chỉ NORM) -> dùng để oversample lớp thiểu số."""
    return any(code != "NORM" for code in scp)


# Lớp ĐA SỐ mỗi task (cap để cân bằng + giảm kích thước). form: đa số = all-negative.
DOMINANT = {"diag": "NORM", "rhythm": "SR", "form": None}


def _is_majority(sample: dict, vocab: list[str]) -> bool:
    pos = {vocab[i] for i, v in enumerate(sample["label_vector"]) if v}
    dom = DOMINANT.get(sample["task"])
    return len(pos) == 0 or (dom is not None and pos == {dom})


def _cap_majority(rows: list[dict], vocabs: dict[str, list[str]], cap: int, counts: dict) -> list[dict]:
    """Giữ TẤT CẢ mẫu thiểu số (bệnh/đa nhãn); cap mẫu đa số mỗi task <= cap. Thứ tự giữ nguyên."""
    seen_maj = {t: 0 for t in TASKS}
    kept, dropped = [], 0
    for s in rows:
        if _is_majority(s, vocabs[s["task"]]):
            if seen_maj[s["task"]] < cap:
                seen_maj[s["task"]] += 1
                kept.append(s)
            else:
                dropped += 1
        else:
            kept.append(s)
    counts["majority_capped_dropped"] = dropped
    counts["majority_kept_per_task"] = seen_maj
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Dựng train/val/test _3task.json cho v2 từ PTB-XL")
    ap.add_argument("--db", required=True, help="ptbxl_database.csv")
    ap.add_argument("--images", required=True, help="thư mục ảnh đã render ({ecg_id:05d}.png)")
    ap.add_argument("--out", required=True, help="thư mục ghi *_3task.json")
    ap.add_argument("--diag-threshold", type=float, default=50.0)
    ap.add_argument("--label-format", default="binary", choices=["binary", "positive_list"])
    ap.add_argument("--oversample-path", type=int, default=1,
                    help="nhân bản record có bệnh (non-NORM) cho TRAIN để cân bằng (1=tắt)")
    ap.add_argument("--cap-majority", type=int, default=0,
                    help="giới hạn số mẫu lớp ĐA SỐ mỗi task trong TRAIN (NORM-only/SR-only/"
                         "all-negative). 0=tắt. Giảm mạnh kích thước + cân bằng, runtime khả thi.")
    ap.add_argument("--no-image-check", action="store_true",
                    help="không kiểm ảnh tồn tại (dùng để smoke-test không cần ảnh)")
    ap.add_argument("--limit", type=int, default=0, help="giới hạn số record (0=tất cả)")
    args = ap.parse_args()

    vocabs = load_task_vocabs()
    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0, "skipped_no_image": 0, "path_oversampled": 0}

    n_seen = 0
    for ecg_id, fold, scp in _read_rows(Path(args.db)):
        if args.limit and n_seen >= args.limit:
            break
        n_seen += 1
        split = next((s for s, folds in FOLD_SPLIT.items() if fold in folds), None)
        if split is None:
            continue
        image_rel = f"images/{ecg_id:05d}.png"
        if not args.no_image_check and not (images_dir / f"{ecg_id:05d}.png").exists():
            counts["skipped_no_image"] += 1
            continue
        samples = build_sample(
            f"ptbxl_{ecg_id:05d}", image_rel, scp, vocabs,
            label_format=args.label_format, diag_threshold=args.diag_threshold)
        reps = args.oversample_path if (split == "train" and _is_pathological(scp)) else 1
        if reps > 1:
            counts["path_oversampled"] += 1
        for _ in range(reps):
            buckets[split].extend(samples)
        counts[split] += len(samples)

    # Cap lớp đa số mỗi task trong TRAIN (giảm kích thước + cân bằng).
    if args.cap_majority > 0:
        buckets["train"] = _cap_majority(buckets["train"], vocabs, args.cap_majority, counts)

    for split, rows in buckets.items():
        path = out_dir / f"{split}_3task.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {split}_3task.json: {len(rows)} mẫu (3/record)")

    # Lưu vocab 3-task để eval/infer dùng.
    (out_dir / "scp_3task_vocab.json").write_text(
        json.dumps({t: vocabs[t] for t in TASKS}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  scp_3task_vocab.json: diag={len(vocabs['diag'])} "
          f"rhythm={len(vocabs['rhythm'])} form={len(vocabs['form'])}")
    print(f"  counts: {counts}")


if __name__ == "__main__":
    main()
