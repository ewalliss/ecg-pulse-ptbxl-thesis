"""Gom CHỈ những ảnh được tham chiếu trong val/test json -> 1 thư mục để upload Colab.

Dùng (chạy trên máy Windows nơi có ảnh gốc):
  python -m khoaluan.code.collect_eval_images \
    --jsons khoaluan/runs/v2-run1/data/val_3task.json khoaluan/runs/v2-run1/data/test_3task.json \
    --images data/pulse_ptbxl_stage2/images \
    --out eval_images_subset

Sau đó nén thư mục `eval_images_subset` và upload lên Google Drive.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", nargs="+", required=True, help="các file *_3task.json")
    ap.add_argument("--images", required=True, help="thư mục ảnh gốc (chứa các .png)")
    ap.add_argument("--out", default="eval_images_subset")
    args = ap.parse_args()

    names: set[str] = set()
    for jp in args.jsons:
        for s in json.loads(Path(jp).read_text(encoding="utf-8")):
            names.add(Path(s["image"]).name)

    src = Path(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    copied, missing = 0, 0
    for n in sorted(names):
        fp = src / n
        if fp.exists():
            shutil.copy2(fp, out / n)
            copied += 1
        else:
            missing += 1
    print(f"unique images: {len(names)}  copied: {copied}  missing: {missing}  -> {out}")


if __name__ == "__main__":
    main()
