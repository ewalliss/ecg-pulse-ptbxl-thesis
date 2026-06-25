"""Full flow v2: data -> train, gói trong MỘT folder run mới (scope mới).

Tạo cấu trúc:
  <runs-root>/<run-name>/
      data/         (train_3task.json, val_3task.json, test_3task.json, scp_3task_vocab.json)
      checkpoints/  (LoRA checkpoints)

Ảnh KHÔNG bị copy (lớn) — train trỏ --image-folder về nơi chứa images/ sẵn có.

Dùng (Apollo):
  python -m khoaluan.code.pipeline \
      --db .../ptbxl_database.csv \
      --images .../pulse_ptbxl_stage2/images \
      --model C:/.../model \
      --run-name v2-run1 --oversample-path 3 --epochs 3 --execute

Bỏ --execute để chỉ in KẾ HOẠCH (dry-run, không chạy gì).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], execute: bool, label: str) -> None:
    print(f"\n=== [{label}] ===")
    print(" ".join(str(c) for c in cmd))
    if execute:
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!! [{label}] thất bại (exit {r.returncode}) — dừng pipeline.")
            sys.exit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Full flow v2: build 3-task data -> train (folder run mới)")
    ap.add_argument("--db", required=True, help="ptbxl_database.csv")
    ap.add_argument("--images", required=True, help="thư mục ảnh đã render ({ecg_id:05d}.png)")
    ap.add_argument("--model", required=True, help="đường dẫn PULSE-7B")
    ap.add_argument("--run-name", default="v2-run1")
    ap.add_argument("--runs-root", default=str(REPO_ROOT / "v2" / "runs"))
    ap.add_argument("--oversample-path", type=int, default=1)
    ap.add_argument("--cap-majority", type=int, default=1500,
                    help="cap mẫu lớp đa số mỗi task trong TRAIN (giảm runtime + cân bằng). 0=tắt.")
    ap.add_argument("--label-format", default="binary", choices=["binary", "positive_list"])
    ap.add_argument("--diag-threshold", type=float, default=50.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--execute", action="store_true",
                    help="THỰC SỰ chạy (mặc định chỉ in kế hoạch). Dùng trên máy có GPU.")
    args = ap.parse_args()

    images = Path(args.images)
    image_folder = images.parent  # json tham chiếu 'images/xxx.png' -> cần thư mục cha của images/

    run_dir = Path(args.runs_root) / args.run_name
    data_dir = run_dir / "data"
    ckpt_dir = run_dir / "checkpoints"
    for d in (data_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Run scope folder : {run_dir}")
    print(f"  data    -> {data_dir}")
    print(f"  ckpts   -> {ckpt_dir}")
    print(f"  images  -> {images}  (image_folder={image_folder}, không copy)")
    print(f"  mode    : {'EXECUTE' if args.execute else 'DRY-RUN (thêm --execute để chạy)'}")

    py = sys.executable
    if not args.skip_build:
        build_cmd = [
            py, "-m", "khoaluan.code.build_3task_json",
            "--db", args.db, "--images", str(images), "--out", str(data_dir),
            "--oversample-path", str(args.oversample_path),
            "--cap-majority", str(args.cap_majority),
            "--label-format", args.label_format,
            "--diag-threshold", str(args.diag_threshold),
        ]
        _run(build_cmd, args.execute, "STEP 1/2 BUILD DATA")

    if not args.skip_train:
        train_cmd = [
            py, "-m", "khoaluan.code.train_v2",
            "--data", str(data_dir),
            "--image-folder", str(image_folder),
            "--model", args.model,
            "--output", str(ckpt_dir),
            "--epochs", str(args.epochs), "--lr", str(args.lr), "--lora-r", str(args.lora_r),
        ]
        if args.execute:
            train_cmd.append("--execute")
        _run(train_cmd, args.execute, "STEP 2/2 TRAIN")

    print(f"\nDone ({'executed' if args.execute else 'dry-run'}). Scope: {run_dir}")


if __name__ == "__main__":
    main()
