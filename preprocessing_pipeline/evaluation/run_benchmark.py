"""
Zero-shot benchmark: run PULSE-7B inference on PTB-XL fold 10 and compute
all 71-class evaluation metrics.

Usage
-----
    python -m src.evaluation.run_benchmark \\
        --split test \\
        --model-path model/models--PULSE-ECG--PULSE-7B \\
        --output outputs/benchmark_zeroshot.json

Options
-------
--split       Which split CSV to use: train / val / test  (default: test)
--model-path  HuggingFace model id or local path (default: local cache)
--image-dir   Override default ECG image directory
--output      Where to write the JSON results
--limit       Process only first N records (smoke test)
--load-4bit   Enable 4-bit quantization (needed if GPU VRAM < 16 GB)
--device      Torch device (default: cuda, falls back to mps/cpu)
--threshold   Global threshold override (default: per-label optimized on val)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing_pipeline.config.preprocessing_config import (
    ECG_IMAGE_DIR,
    SPLITS_DIR,
    SCP_ALL_CODES,
    NUM_CLASSES,
)
from preprocessing_pipeline.evaluation.metrics import (
    compute_full_report,
    optimize_thresholds,
    apply_thresholds,
    DEFAULT_THRESHOLD,
)
from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[2] / "model" / "models--PULSE-ECG--PULSE-7B"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot PULSE-7B benchmark on PTB-XL 71-class"
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--model-path", default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output", default="outputs/benchmark_zeroshot.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Global threshold (skips val-set optimization)")
    parser.add_argument("--optimize-thresholds", action="store_true",
                        help="Optimize per-label thresholds on val set before scoring")
    return parser.parse_args()


def load_split(split: str, image_dir: Path, limit: int | None) -> pd.DataFrame:
    split_csv = SPLITS_DIR / f"{split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {split_csv}\n"
            "Run preprocessing first: python -m src.run_preprocessing"
        )
    df = pd.read_csv(split_csv)
    df["image_path"] = df["ecg_id"].apply(lambda x: image_dir / f"{int(x):05d}.png")
    missing = df[~df["image_path"].apply(lambda p: Path(p).exists())]
    if len(missing) > 0:
        log.warning("%d images missing — they will be skipped.", len(missing))
        df = df[df["image_path"].apply(lambda p: Path(p).exists())]
    if limit:
        df = df.head(limit)
    log.info("Loaded %s split: %d records", split, len(df))
    return df.reset_index(drop=True)


def run_inference(
    df: pd.DataFrame,
    model_path: str,
    device: str,
    load_4bit: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run PULSE inference on all records, return (y_true, y_prob).

    Returns
    -------
    y_true : np.ndarray shape (N, 71) — binary ground truth from split CSV
    y_prob : np.ndarray shape (N, 71) — predicted probabilities from PULSE
    """
    from preprocessing_pipeline.pulse_ptbxl.pulse_inference import load_pulse_classifier

    log.info("Loading PULSE model from %s on %s ...", model_path, device)
    classifier = load_pulse_classifier(
        model_path=model_path,
        device=device,
        load_4bit=load_4bit,
    )
    log.info("Model loaded. Running inference on %d records ...", len(df))

    y_true_rows = []
    y_prob_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference", unit="rec"):
        # Ground truth from split CSV columns
        gt_vector = np.array([int(row[code]) for code in SCP_ALL_CODES], dtype=np.int8)
        y_true_rows.append(gt_vector)

        # PULSE inference
        pred = classifier.classify_image(
            image_file=row["image_path"],
            temperature=0.0,
            max_new_tokens=256,
        )

        # Convert binary_vector to float probabilities
        # confidence_scores override binary where available
        prob_vector = np.array(pred.binary_vector, dtype=np.float32)
        if pred.confidence_scores:
            for code, score in pred.confidence_scores.items():
                if code in SCP_ALL_CODES:
                    idx = SCP_ALL_CODES.index(code)
                    prob_vector[idx] = float(score)

        y_prob_rows.append(prob_vector)

    y_true = np.stack(y_true_rows)   # (N, 71)
    y_prob = np.stack(y_prob_rows)   # (N, 71)
    return y_true, y_prob


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir) if args.image_dir else ECG_IMAGE_DIR
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_split(args.split, image_dir, args.limit)

    y_true, y_prob = run_inference(
        df,
        model_path=args.model_path,
        device=args.device,
        load_4bit=args.load_4bit,
    )

    # Threshold strategy
    if args.threshold is not None:
        thresholds = np.full(NUM_CLASSES, args.threshold)
        log.info("Using global threshold %.2f", args.threshold)
    elif args.optimize_thresholds:
        log.info("Optimizing thresholds on val set ...")
        val_df = load_split("val", image_dir, limit=None)
        y_true_val, y_prob_val = run_inference(
            val_df, args.model_path, args.device, args.load_4bit
        )
        thresholds = optimize_thresholds(y_true_val, y_prob_val)
    else:
        thresholds = np.full(NUM_CLASSES, DEFAULT_THRESHOLD)
        log.info("Using default threshold %.2f", DEFAULT_THRESHOLD)

    log.info("Computing metrics ...")
    report = compute_full_report(y_true, y_prob, thresholds=thresholds)
    report["split"] = args.split
    report["model_path"] = args.model_path
    report["limit"] = args.limit

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Results saved → %s", output_path)

    # Print summary
    print("\n=== Benchmark Results ===")
    print(f"  Split:       {args.split}  (N={report['n_samples']})")
    print(f"  Macro AUC:   {report['macro_auc']:.4f}  (PULSE zero-shot baseline: 0.824)")
    print(f"  Macro F1:    {report['macro_f1']:.4f}")
    print(f"  Hamming:     {report['hamming_loss']:.6f}")
    print("\nPer-superclass AUC:")
    for group, auc in sorted(report["per_superclass_auc"].items()):
        print(f"  {group:20s}: {auc:.4f}")


if __name__ == "__main__":
    main()
