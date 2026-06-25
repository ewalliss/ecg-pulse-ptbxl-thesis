# Windows + Tesla T4 (16 GB) — Setup & Run Guide

End-to-end on a Windows machine with an NVIDIA **Tesla T4 (16 GB)** GPU.
Preprocessing **and** training both run here. The base PULSE-7B model and the
raw PTB-XL dataset are assumed to already exist on this machine.

> Why `train_t4.py` and not `train.py`: T4 is Turing (sm_75) — **no bf16, no
> FlashAttention-2**. `train.py` uses both and will crash. `train_t4.py` swaps
> them for `fp16` + `sdpa`. Nothing in the PULSE core model is changed.

---

## 0. Prerequisites

- NVIDIA driver + CUDA runtime working (`nvidia-smi` shows the T4).
- Python 3.10/3.11 (conda or venv recommended).
- Raw PTB-XL already on disk: `ptbxl_database.csv`, `scp_statements.csv`,
  `records500/`.
- Base model already on disk (a local `PULSE-7B/` folder or HF cache).

---

## 1. Pull the code

```powershell
git pull origin code
```

## 2. Install dependencies

```powershell
pip install "transformers>=4.36.0,<4.42" "peft>=0.9.0,<0.12" "accelerate>=0.27.0,<0.35" `
    "bitsandbytes>=0.43.0" "pandas<3.0" scikit-learn wfdb matplotlib Pillow tqdm einops
pip install -e .\source\LLaVA
```

> **Do NOT install flash-attn** — it does not build on Turing.
> **bitsandbytes on Windows**: use `>=0.43.0` (official Windows wheels with CUDA).
> Older versions lack a Windows CUDA binary and 4-bit QLoRA will fail to load.

## 3. Point the pipeline at your local PTB-XL

Set the dataset root (the folder that directly contains `ptbxl_database.csv`,
`scp_statements.csv`, `records500/`). This is the only path that must change
per machine:

```powershell
$env:ECGLLAVA_DATASET_ROOT = "D:\data\ptb-xl"
```

Generated artifacts (images, JSON, vocab) land in `.\data\` inside the repo by
default — override with `$env:ECGLLAVA_OUTPUT_ROOT` if you want them elsewhere.

## 4. Build the Stage-2 dataset (preprocessing happens here)

This single command filters to the **44 diagnostic SCP codes**, renders ECG
images, translates reports, and writes `train.json` / `val.json` /
`scp_diag_vocab.json`.

```powershell
# Smoke test first — 200 records, verify the pipeline runs clean:
python build_stage2_dataset.py --limit 200

# Full build — uses ALL CPU cores by default to render images in parallel:
python build_stage2_dataset.py

# Cap the core count if RAM is tight (each worker ~150 MB):
python build_stage2_dataset.py --workers 8
```

> **Image rendering uses CPU, not GPU.** matplotlib (Agg backend) is a CPU
> rasteriser — there is no GPU line-plot path, and rewriting the renderer would
> change the images PULSE was trained on. Instead, rendering runs one process
> per CPU core (`--workers`, default = all cores), turning a ~9 h serial build
> into ~1 h on a typical workstation. The **report translation** step *does* use
> the GPU automatically (MarianMT → CUDA when available).

Output goes to `.\data\pulse_ptbxl_stage2\` (train.json, val.json, images/,
scp_diag_vocab.json).

## 5. Train on the T4

```powershell
python train_t4.py `
    --data   .\data\pulse_ptbxl_stage2 `
    --model  D:\models\PULSE-7B `
    --output .\checkpoints\pulse-ptbxl-qlora-t4
```

- `--model` → your local base-model folder (or `PULSE-ECG/PULSE-7B` to pull from HF).
- Auto-resumes from the latest `checkpoint-*` in `--output` if interrupted.

### Recommended first run (verify VRAM fits before committing hours)

```powershell
python build_stage2_dataset.py --limit 200
python train_t4.py --data .\data\pulse_ptbxl_stage2 --model D:\models\PULSE-7B `
    --output .\checkpoints\smoke --epochs 1
```

Watch peak VRAM in a second terminal: `nvidia-smi -l 2`. Expect ~10–13 GB.

### If you hit CUDA out-of-memory

Lower the sequence length (1728 AnyRes visual tokens dominate the budget):

```powershell
python train_t4.py ... --max-length 2048
```

`2048` is the safe floor that still fits the 1728 image tokens plus the label
block; very long interpretation text may be truncated.

---

## Notes

- T4 is slow: roughly **6–9 h/epoch** on the full ~11k-sample set. Plan for
  long runs and rely on auto-resume.
- `data/`, `model/`, `PULSE-7B/`, `checkpoint(s)/` are git-ignored — they never
  travel through git; you regenerate the dataset locally from raw PTB-XL.
- For A100 / L4 / V100 use `train.py` instead (bf16 + flash-attn-2).
