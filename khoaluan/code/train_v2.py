"""Launcher train v2 — IN ra cấu hình/lệnh, KHÔNG chạy (milestone v1 không train).

Đường train chính của v2 = CE GỐC của PULSE (next-token, instruction tuning) trên
dữ liệu 3-task+reasoning MỚI -> KHÔNG cần sửa source/LLaVA. Khác bản cũ ở chỗ:
KHÔNG bật cờ ASL-Gen; dùng đúng objective native của LLaVA.

Class-balanced weighting (khoaluan/code/weighted_ce.py) là ABLATION cần hook ở trainer —
ghi chú rõ là tuỳ chọn, không nằm trong đường train chính (để giữ ràng buộc không-sửa-core).

Dùng: python -m khoaluan.code.train_v2 --data <stage2_dir> --model <pulse> --output <ckpt> --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PULSE_TRAIN_ENTRY = REPO_ROOT / "source" / "LLaVA" / "llava" / "train" / "train.py"


def build_command(data: Path, image_folder: Path, model: str, output: Path,
                  epochs: int, lr: float, lora_r: int) -> list[str]:
    """Lệnh train dùng objective CE GỐC của PULSE (KHÔNG cờ ASL).

    image_folder = thư mục CHỨA images/ (json tham chiếu 'images/xxx.png'); có thể
    khác `data` (folder scope mới chỉ giữ json + checkpoint, ảnh nằm chỗ cũ)."""
    train_json = data / "train_3task.json"
    return [
        sys.executable, "-c",   # dùng Python của venv (có llava), KHÔNG phải 'python' hệ thống
        "from llava.train.train import train; train(attn_implementation='sdpa')",
        "--model_name_or_path", str(model),
        "--version", "llava_v1",
        "--data_path", str(train_json),
        "--image_folder", str(image_folder),
        "--vision_tower", "openai/clip-vit-large-patch14-336",
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",   # mặc định True -> resize_token_embeddings crash 4-bit
        "--image_aspect_ratio", "anyres",
        "--group_by_modality_length", "False",
        "--bits", "4", "--quant_type", "nf4", "--double_quant", "True",
        "--lora_enable", "True", "--lora_r", str(lora_r), "--lora_alpha", str(lora_r * 2),
        "--lora_dropout", "0.05", "--freeze_mm_mlp_adapter", "True",
        # KHÔNG có --aslgen_* : v2 dùng CE next-token gốc.
        "--fp16", "True",
        "--output_dir", str(output),
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "128",
        "--save_strategy", "steps", "--save_steps", "0.1", "--save_total_limit", "8",
        "--learning_rate", str(lr), "--weight_decay", "0.0",
        "--warmup_ratio", "0.03", "--lr_scheduler_type", "cosine",
        "--logging_steps", "1", "--model_max_length", "2560",
        "--gradient_checkpointing", "True", "--lazy_preprocess", "True",
        "--report_to", "none",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Launcher train v2 (task-decomposed + reasoning, CE gốc)")
    ap.add_argument("--data", required=True, help="thư mục chứa train_3task.json")
    ap.add_argument("--image-folder", default=None,
                    help="thư mục CHỨA images/ (mặc định = --data nếu không truyền)")
    ap.add_argument("--model", default="PULSE-ECG/PULSE-7B")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--execute", action="store_true",
                    help="THỰC SỰ chạy train (mặc định chỉ in lệnh). Dùng trên máy có GPU (Apollo).")
    args = ap.parse_args()

    image_folder = Path(args.image_folder) if args.image_folder else Path(args.data)
    cmd = build_command(Path(args.data), image_folder, args.model, Path(args.output),
                        args.epochs, args.lr, args.lora_r)
    print("=== v2 train command (CE gốc PULSE, KHÔNG ASL) ===")
    print(" ".join(cmd))
    print(f"\nPULSE train entry: {PULSE_TRAIN_ENTRY} (read-only, không sửa)")
    print("Ablation class-balanced weighting: dùng khoaluan/code/weighted_ce.py như hook ở trainer (tuỳ chọn).")
    if args.execute:
        import subprocess
        print("\n--- EXECUTE: launching training ---")
        sys.exit(subprocess.run(cmd).returncode)
    print("\n(DRY RUN — thêm --execute để chạy thật)")


if __name__ == "__main__":
    main()
