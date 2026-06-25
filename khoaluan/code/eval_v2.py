"""Eval teacher-forced cho v2 — macro-AUC RIÊNG từng task (method-spec §6).

KHÔNG generate, KHÔNG parse: với mỗi (ảnh, task) ta dựng chuỗi answer mẫu
"[REASONING]\\n\\n[LABELS]\\nCODE: 0\\n..." và một forward pass đọc P("1") tại vị trí
chữ số đã biết (teacher_forced.score_task). Vì vị trí do ta đặt -> luôn đủ n_codes,
không bao giờ malformed (vá lỗi 73% malformed của free-generation).

Chạy trên máy GPU (Windows/Apollo):
  python -m khoaluan.code.eval_v2 \
      --adapter khoaluan/runs/v2-run1/checkpoints/checkpoint-XXX \
      --base   C:/.../model \
      --vocab  khoaluan/runs/v2-run1/data/scp_3task_vocab.json \
      --val    khoaluan/runs/v2-run1/data/val_3task.json \
      --images data/pulse_ptbxl_stage2/images \
      --n 200 --load-4bit --dtype float16
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from khoaluan.code.prompts import build_prompt
from khoaluan.code.teacher_forced import (
    build_answer_positions, label_token_ids, score_task,
)

IMAGE_TOKEN = "<image>"


def _build_prompt_ids(tokenizer, L, task: str, device):
    conv = L["conv_templates"]["llava_v1"].copy()
    conv.append_message(conv.roles[0], f"{IMAGE_TOKEN}\n{build_prompt(task)}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return (
        L["tokenizer_image_token"](prompt, tokenizer, L["IMAGE_TOKEN_INDEX"], return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )


def _image_tensor(L, image, image_processor, model, device):
    import torch
    dtype = next(model.parameters()).dtype
    t = L["process_images"]([image], image_processor, model.config)[0]
    return t.unsqueeze(0).to(device=device, dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval teacher-forced (macro-AUC per task) cho v2")
    ap.add_argument("--adapter", default="", help="LoRA adapter dir; để trống = eval BASE PULSE-7B zero-shot")
    ap.add_argument("--base", required=True)
    ap.add_argument("--vocab", required=True, help="scp_3task_vocab.json {diag,rhythm,form}")
    ap.add_argument("--val", required=True, help="val_3task.json")
    ap.add_argument("--images", required=True)
    ap.add_argument("--tasks", default="diag,rhythm,form")
    ap.add_argument("--n", type=int, default=200, help="số mẫu MỖI task (0=tất cả)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default=None, help="dump JSON scores/gt (tùy chọn)")
    ap.add_argument("--thresholds", default="", help="JSON eval fold-9 để NẠP per_class_threshold; "
                    "nếu có, áp ngưỡng val lên tập hiện tại thay vì dò lại (tránh rò rỉ test)")
    ap.add_argument("--image-mode", default="real", choices=["real", "blank", "shuffle"],
                    help="ablation kiểm chứng model DÙNG ảnh ECG: real=ảnh đúng; "
                         "blank=ảnh xám đồng nhất (xoá tín hiệu); shuffle=gán ảnh của mẫu KHÁC "
                         "(phá khớp ảnh-nhãn). Nếu AUC tụt về ~0.5 ở blank/shuffle => model thật sự đọc ảnh.")
    ap.add_argument("--placeholder", default="zero", choices=["zero", "one"],
                    help="token điền slot nhãn (mặc định zero). KIỂM SOÁT leakage: đặt 'one' và "
                         "kiểm tra AUC gần như không đổi => điểm là logit dự đoán, nhãn thật không vào context.")
    args = ap.parse_args()

    from PIL import Image
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score
    from khoaluan.code.infer_aslgen import load_model  # tái dùng loader đã kiểm chứng

    vocabs = json.loads(Path(args.vocab).read_text())
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    # Ngưỡng nạp từ eval fold-9 (nếu có): {task: {code: thr}} -> áp lên tập hiện tại.
    loaded_thr = {}
    if args.thresholds:
        prev = json.loads(Path(args.thresholds).read_text())
        loaded_thr = {t: prev[t].get("per_class_threshold", {}) for t in prev}
        print(f"Loaded per-class thresholds from {args.thresholds} (no re-tuning on this set).\n")

    tokenizer, model, image_processor, L = load_model(
        args.base, args.adapter, args.device, args.dtype, args.load_4bit, args.merge)
    pos_id, neg_id = label_token_ids(tokenizer)
    placeholder_id = pos_id if args.placeholder == "one" else neg_id
    print(f"Label tokens: '1'={pos_id} '0'={neg_id}")
    print(f"image-mode={args.image_mode}  placeholder={args.placeholder}({placeholder_id})\n")

    samples = json.loads(Path(args.val).read_text())
    by_task = defaultdict(list)
    for s in samples:
        if s.get("task") in tasks:
            by_task[s["task"]].append(s)

    dump = {}
    for task in tasks:
        vocab = vocabs[task]
        answer_ids, digit_pos = build_answer_positions(
            tokenizer, vocab, neg_id, placeholder_id=placeholder_id)
        rows = by_task[task][: args.n] if args.n else by_task[task]
        # shuffle: gán ảnh của mẫu KHÁC (lệch nửa danh sách -> luôn khớp sai), nhãn giữ nguyên.
        shift = len(rows) // 2 if args.image_mode == "shuffle" else 0
        print(f"=== task={task}  codes={len(vocab)}  samples={len(rows)} ===")

        score_rows, gt_rows = [], []
        for k, s in enumerate(rows, 1):
            src = rows[(k - 1 + shift) % len(rows)] if shift else s
            img_path = Path(args.images) / Path(src["image"]).name
            image = Image.open(str(img_path)).convert("RGB")
            if args.image_mode == "blank":
                image = Image.new("RGB", image.size, (128, 128, 128))  # xoá tín hiệu, giữ kích thước
            prompt_ids = _build_prompt_ids(tokenizer, L, task, args.device)
            images = _image_tensor(L, image, image_processor, model, args.device)
            p1 = score_task(model, tokenizer, prompt_ids, images, [image.size],
                            answer_ids, digit_pos, pos_id, neg_id, args.device)
            score_rows.append(p1)
            gt_rows.append([int(v) for v in s["label_vector"]])
            if k % 25 == 0 or k == len(rows):
                print(f"  [{k}/{len(rows)}] maxP={max(p1):.2f}", flush=True)

        scores = np.array(score_rows)
        gt = np.array(gt_rows)
        per_class = {}
        for c in range(len(vocab)):
            col = gt[:, c]
            if col.min() == 0 and col.max() == 1:
                per_class[vocab[c]] = roc_auc_score(col, scores[:, c])
        macro_auc = float(np.mean(list(per_class.values()))) if per_class else float("nan")

        # Ngưỡng micro toàn cục (1 ngưỡng chung) — để đối chiếu.
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.02, 0.6, 30):
            f1 = f1_score(gt.ravel(), (scores >= t).astype(int).ravel(), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        f1_05 = f1_score(gt.ravel(), (scores >= 0.5).astype(int).ravel(), zero_division=0)

        # HIỆU CHUẨN NGƯỠNG THEO LỚP: mỗi lớp quét ngưỡng tối đa F1 của RIÊNG lớp đó
        # (chuẩn cho multi-label mất cân bằng). Báo macro-F1 với ngưỡng per-class +
        # lưu ngưỡng để dùng lúc infer.
        # Nếu nạp ngưỡng từ fold-9: ÁP ngưỡng đó (không dò lại) -> F1 test trung thực.
        # Ngược lại: dò ngưỡng tối đa F1/lớp trên chính tập này (chỉ dùng cho VAL).
        task_thr = loaded_thr.get(task, {})
        thr_mode = "val-frozen" if task_thr else "self-tuned"
        per_class_thr, per_class_f1 = {}, []
        for c in range(len(vocab)):
            col = gt[:, c]
            if col.min() == 0 and col.max() == 1:
                if task_thr:
                    bt_c = float(task_thr.get(vocab[c], 0.5))
                    bf_c = f1_score(col, (scores[:, c] >= bt_c).astype(int), zero_division=0)
                else:
                    bt_c, bf_c = 0.5, -1.0
                    for t in np.linspace(0.02, 0.8, 40):
                        f1c = f1_score(col, (scores[:, c] >= t).astype(int), zero_division=0)
                        if f1c > bf_c:
                            bf_c, bt_c = f1c, float(t)
                per_class_thr[vocab[c]] = round(bt_c, 3)
                per_class_f1.append(bf_c)
        macro_f1_cal = float(np.mean(per_class_f1)) if per_class_f1 else float("nan")

        print(f"  macro-AUC ({task})       : {macro_auc:.4f}   <-- headline (threshold-free)")
        print(f"  classes with AUC         : {len(per_class)}/{len(vocab)}")
        print(f"  macro-F1 @per-class-thr  : {macro_f1_cal:.4f}   <-- ngưỡng/lớp ({thr_mode})")
        print(f"  micro-F1 @best({best_t:.2f})  : {best_f1:.4f}   |  @0.50: {f1_05:.4f}")
        print(f"  mean P(1) on pos/neg     : {scores[gt==1].mean():.3f} / {scores[gt==0].mean():.3f}")
        ranked = sorted(per_class.items(), key=lambda kv: kv[1])
        print(f"  worst5: {[f'{k}={v:.2f}' for k,v in ranked[:5]]}")
        print(f"  best5 : {[f'{k}={v:.2f}' for k,v in ranked[-5:]]}\n")
        dump[task] = {"codes": vocab, "scores": score_rows, "gt": gt_rows,
                      "macro_auc": macro_auc, "macro_f1_calibrated": macro_f1_cal,
                      "per_class_auc": per_class, "per_class_threshold": per_class_thr,
                      "threshold_mode": thr_mode, "global_best_threshold": best_t}

    if args.out:
        Path(args.out).write_text(json.dumps(dump, indent=2))
        print(f"Dumped -> {args.out}")


if __name__ == "__main__":
    main()
