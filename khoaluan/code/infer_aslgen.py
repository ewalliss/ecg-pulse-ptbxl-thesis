#!/usr/bin/env python3
"""
Inference + quick evaluation for an ASL-Gen LoRA checkpoint (e.g. checkpoint-159).

This MUST mirror the exact training contract of build_stage2_dataset.py:
  - Prompt  = SYSTEM_PROMPT verbatim ("<image>\\nYou are an expert cardiologist...
              ...inside an <asl_labels> block.")
  - Output  = "Interpretation: ...\\n\\nPositive SCP-ECG labels: ...\\n\\n
              <asl_labels>\\nNORM 0\\nIMI 1\\n...(71 lines)\\n</asl_labels>"
  - Image   = anyres tiling (model.config.image_aspect_ratio == 'anyres')

The repo's existing src/pulse_ptbxl/pulse_inference.py targets a DIFFERENT
output scheme ("Rationale / Labels (0.XX)") and would mis-parse this checkpoint,
so this is a dedicated faithful runner.

Loading: the checkpoint dir holds only a LoRA adapter (adapter_config.json +
adapter_model.safetensors), no config.json — so we load the BASE PULSE-7B
(LlavaLlamaForCausalLM) and attach the adapter with PEFT directly. The LLaVA
builder's LoRA path is not used (it expects a full config in the adapter dir).

Colab usage
-----------
  !python testing_inference/infer_aslgen.py \\
      --adapter "/content/.../checkpoints/pulse-ptbxl-qlora/checkpoint-159" \\
      --base    "/content/.../model/PULSE-7B" \\
      --val     "/content/.../outputs/pulse_ptbxl_stage2/val.json" \\
      --images  "/content/.../outputs/pulse_ptbxl_stage2/images" \\
      --vocab   "/content/.../outputs/pulse_ptbxl_stage2/scp_diag_vocab.json" \\
      --n 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent   # script lives in testing_inference/
LLAVA_ROOT = REPO_ROOT / "source" / "LLaVA"

# Must match build_stage2_dataset.SYSTEM_PROMPT exactly (training distribution).
SYSTEM_PROMPT = (
    "<image>\nYou are an expert cardiologist. Analyze this 12-lead ECG. "
    "First, provide a detailed clinical interpretation pinpointing specific waveform abnormalities. "
    "Then, provide the final diagnosis using the 44 SCP-ECG diagnostic labels inside an <asl_labels> block."
)


def _load_llava():
    if str(LLAVA_ROOT) not in sys.path:
        sys.path.insert(0, str(LLAVA_ROOT))
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    from llava.utils import disable_torch_init
    return {
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "conv_templates": conv_templates,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "LlavaLlamaForCausalLM": LlavaLlamaForCausalLM,
        "disable_torch_init": disable_torch_init,
    }


_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_model(base: str, adapter: str, device: str = "cuda",
               dtype_str: str = "float16", load_4bit: bool = False, do_merge: bool = False):
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    L = _load_llava()
    L["disable_torch_init"]()

    dtype = _DTYPES[dtype_str] if device == "cuda" else torch.float32

    load_kwargs = {"low_cpu_mem_usage": True}
    if load_4bit:
        # 4-bit base (~5 GB) — fits 16 GB Tesla. Cannot merge a quantized base,
        # so the LoRA adapter stays attached (PeftModel). Results are identical.
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = {"": 0} if device == "cuda" else {"": device}
        print(f"Loading base PULSE-7B ({base}) in 4-bit NF4 (compute {dtype}) ...")
    else:
        load_kwargs["torch_dtype"] = dtype
        print(f"Loading base PULSE-7B ({base}) in {dtype} ...")

    model = L["LlavaLlamaForCausalLM"].from_pretrained(base, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)

    # Vision tower loads lazily on some builds; ensure it is materialised.
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=dtype)
    image_processor = vision_tower.image_processor

    if adapter:
        print(f"Attaching LoRA adapter ({adapter}) ...")
        model = PeftModel.from_pretrained(model, adapter)
        if do_merge and not load_4bit:
            print("Merging LoRA into base (faster, needs more VRAM) ...")
            model = model.merge_and_unload()
    else:
        print("No adapter given — evaluating BASE PULSE-7B (zero-shot, no QLoRA).")
    if not load_4bit:
        model = model.to(device=device, dtype=dtype)   # 4-bit is already placed by device_map
    model.eval()
    print("Model ready.\n")
    return tokenizer, model, image_processor, L


def build_input(tokenizer, model, image_processor, L, image: Image.Image, device: str):
    conv = L["conv_templates"]["llava_v1"].copy()
    conv.append_message(conv.roles[0], SYSTEM_PROMPT)   # human turn = exact training prompt
    conv.append_message(conv.roles[1], None)            # assistant turn to be generated
    prompt = conv.get_prompt()

    input_ids = (
        L["tokenizer_image_token"](prompt, tokenizer, L["IMAGE_TOKEN_INDEX"], return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )
    dtype = next(model.parameters()).dtype
    image_tensor = L["process_images"]([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).to(device=device, dtype=dtype)
    return input_ids, image_tensor


@torch.inference_mode()
def generate(tokenizer, model, image_processor, L, image_path: str, device: str, max_new_tokens: int):
    image = Image.open(image_path).convert("RGB")
    input_ids, image_tensor = build_input(tokenizer, model, image_processor, L, image, device)
    out = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=False,                 # greedy → deterministic, matches eval intent
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    # LLaVA's generate() runs on inputs_embeds, so `out` already contains ONLY the
    # newly generated tokens — there is no prompt prefix to strip. Slicing by the
    # text prompt length would chop off the start of the answer. Decode directly.
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def parse_asl_labels(text: str, codes: list[str]) -> list[int]:
    """Parse the <asl_labels> block into a 71-vector aligned to `codes`.

    Robust to a missing closing tag or truncated output: any code line not seen
    defaults to 0. Lines look like 'NORM 0' / 'IMI 1'.
    """
    code_to_idx = {c: i for i, c in enumerate(codes)}
    vec = [0] * len(codes)
    in_block = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == "<asl_labels>":
            in_block = True
            continue
        if line == "</asl_labels>":
            break
        if not in_block:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] in code_to_idx:
            try:
                vec[code_to_idx[parts[0]]] = 1 if int(parts[-1]) == 1 else 0
            except ValueError:
                pass
    return vec


def prf(pred: list[int], true: list[int]) -> tuple[int, int, int, float, float, float]:
    tp = sum(p and t for p, t in zip(pred, true))
    fp = sum(p and not t for p, t in zip(pred, true))
    fn = sum((not p) and t for p, t in zip(pred, true))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return tp, fp, fn, prec, rec, f1


def names(vec: list[int], codes: list[str]) -> list[str]:
    return [c for c, b in zip(codes, vec) if b == 1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Inference/eval for ASL-Gen LoRA checkpoint")
    ap.add_argument("--adapter", required=True, help="LoRA checkpoint dir (checkpoint-159)")
    ap.add_argument("--base", required=True, help="Base PULSE-7B dir")
    ap.add_argument("--vocab", required=True, help="scp_diag_vocab.json (for label order)")
    ap.add_argument("--image", help="Single ECG image to run on")
    ap.add_argument("--val", help="val.json to sample from")
    ap.add_argument("--images", help="images dir for val samples")
    ap.add_argument("--n", type=int, default=10, help="num val samples to evaluate")
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="float16 for Tesla T4/V100 (no bf16 support); bfloat16 for A100/Ampere+")
    ap.add_argument("--load-4bit", action="store_true",
                    help="4-bit NF4 base (~5GB) — use on 16GB GPUs; needs bitsandbytes")
    ap.add_argument("--merge", action="store_true",
                    help="merge LoRA into base (faster, needs ~28GB VRAM; skip on small GPUs)")
    ap.add_argument("--show-raw", action="store_true", help="print raw generation")
    args = ap.parse_args()

    vocab = json.loads(Path(args.vocab).read_text())
    codes = vocab["labels"] if isinstance(vocab, dict) else vocab
    assert len(codes) > 0, "empty vocab"
    print(f"Label set: {len(codes)} codes (expected 44 diagnostic)")

    tokenizer, model, image_processor, L = load_model(
        args.base, args.adapter, args.device,
        dtype_str=args.dtype, load_4bit=args.load_4bit, do_merge=args.merge,
    )

    # --- single image mode ---
    if args.image:
        raw = generate(tokenizer, model, image_processor, L, args.image, args.device, args.max_new_tokens)
        pred = parse_asl_labels(raw, codes)
        print("=" * 70)
        print("RAW OUTPUT:\n" + raw)
        print("=" * 70)
        print("Predicted positives:", names(pred, codes) or "(none)")
        return

    # --- val evaluation mode ---
    if not (args.val and args.images):
        ap.error("provide --image, or both --val and --images")

    samples = json.loads(Path(args.val).read_text())[: args.n]
    agg_tp = agg_fp = agg_fn = 0
    print(f"Evaluating {len(samples)} val samples\n" + "=" * 70)
    for i, s in enumerate(samples):
        img_name = Path(s["image"]).name
        img_path = str(Path(args.images) / img_name)
        true = [int(x) for x in s["scp_vector"]]
        raw = generate(tokenizer, model, image_processor, L, img_path, args.device, args.max_new_tokens)
        pred = parse_asl_labels(raw, codes)
        tp, fp, fn, p, r, f1 = prf(pred, true)
        agg_tp += tp; agg_fp += fp; agg_fn += fn
        print(f"[{i+1:02d}] {s.get('id', img_name)}")
        print(f"     GT  : {names(true, codes) or '(none)'}")
        print(f"     PRED: {names(pred, codes) or '(none)'}")
        print(f"     TP={tp} FP={fp} FN={fn} | P={p:.2f} R={r:.2f} F1={f1:.2f}")
        if args.show_raw:
            print("     ---\n" + "\n".join("     " + ln for ln in raw.splitlines()))
        print()

    mp = agg_tp / (agg_tp + agg_fp) if agg_tp + agg_fp else 0.0
    mr = agg_tp / (agg_tp + agg_fn) if agg_tp + agg_fn else 0.0
    mf1 = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    print("=" * 70)
    print(f"MICRO over {len(samples)} samples: TP={agg_tp} FP={agg_fp} FN={agg_fn}")
    print(f"  Micro-Precision={mp:.3f}  Micro-Recall={mr:.3f}  Micro-F1={mf1:.3f}")
    print("Note: chỉ là kiểm tra nhanh trên N mẫu — KHÔNG phải đánh giá chính thức (fold 10).")


if __name__ == "__main__":
    main()
