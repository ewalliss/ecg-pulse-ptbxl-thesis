from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from preprocessing_pipeline.config.preprocessing_config import ECG_IMAGE_DIR, OUTPUT_ROOT, SCP_ALL_CODES
from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_STAGE2_DIR = OUTPUT_ROOT / "pulse_ptbxl_stage2"
DEFAULT_STAGE2_IMAGE_DIR = DEFAULT_STAGE2_DIR / "images"
DEFAULT_STAGE2_LABEL_VOCAB = DEFAULT_STAGE2_DIR / "scp_diag_vocab.json"

SYSTEM_PROMPT = (
    "<image>\nYou are an expert cardiologist. Analyze this 12-lead ECG. "
    "First, provide a detailed clinical interpretation pinpointing specific waveform abnormalities. "
    "Then, provide the final diagnosis using the 44 SCP-ECG diagnostic labels inside an <asl_labels> block."
)

# Fallback only when a record has no usable report at all (rare). The real
# per-record interpretation is the English-translated PTB-XL ``report`` column.
_INTERPRETATION_FALLBACK = "No automated interpretation available for this record."


def format_binary_answer(
    label_codes: list[str],
    scp_vector: Iterable[int],
    interpretation: str = "",
) -> str:
    vector = [int(v) for v in scp_vector]
    positives = [code for code, bit in zip(label_codes, vector) if bit == 1]
    positive_text = ", ".join(positives) if positives else "NORM"
    asl_lines = "\n".join(f"{code} {bit}" for code, bit in zip(label_codes, vector))
    interp = interpretation.strip() if interpretation and interpretation.strip() else _INTERPRETATION_FALLBACK
    return (
        f"Interpretation: {interp}\n\n"
        f"Positive SCP-ECG labels: {positive_text}\n\n"
        f"<asl_labels>\n{asl_lines}\n</asl_labels>"
    )


def build_stage2_sample(
    ecg_id: int,
    fold: int,
    image_name: str,
    label_codes: list[str],
    scp_vector: Iterable[int],
    interpretation: str = "",
) -> dict:
    vector = [int(value) for value in scp_vector]
    scp_codes = [code for code, value in zip(label_codes, vector) if value == 1]
    return {
        "id": f"ptbxl_{ecg_id:05d}",
        "image": f"images/{image_name}",
        "conversations": [
            {"from": "human", "value": SYSTEM_PROMPT},
            {"from": "gpt", "value": format_binary_answer(label_codes, vector, interpretation)},
        ],
        "scp_codes": scp_codes,
        "scp_vector": vector,
        "scp_vocab_version": "ptbxl_scp_44diag",
        "fold": int(fold),
        "ecg_id": int(ecg_id),
    }


def write_label_vocab(path: Path, label_codes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scp_vocab_version": "ptbxl_scp_44diag",
        "num_labels": len(label_codes),
        "labels": label_codes,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _render_image_core(ecg_id: int, filename_hr: str, image_dir: Path, reuse_existing: bool) -> str | None:
    image_name = f"{ecg_id:05d}.png"
    output_path = image_dir / image_name
    if reuse_existing and output_path.exists():
        return image_name

    existing_path = ECG_IMAGE_DIR / image_name
    if reuse_existing and existing_path.exists():
        image_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(existing_path.read_bytes())
        return image_name

    # Imports are local so render worker processes (spawn on Windows) only pull
    # matplotlib/scipy/wfdb when a record actually needs rendering.
    from preprocessing_pipeline.preprocessing.image.ecg_renderer import render_ecg_image
    from preprocessing_pipeline.preprocessing.signal.filter import apply_bandpass_filter
    from preprocessing_pipeline.preprocessing.signal.loader import load_wfdb_record_safe

    signal = load_wfdb_record_safe(filename_hr)
    if signal is None:
        return None
    render_ecg_image(apply_bandpass_filter(signal), output_path, ecg_id=ecg_id)
    return image_name


def _render_worker(task: tuple[int, str, str, bool]) -> tuple[int, str | None]:
    """Top-level (picklable) worker for ProcessPoolExecutor."""
    ecg_id, filename_hr, image_dir_str, reuse_existing = task
    return ecg_id, _render_image_core(ecg_id, filename_hr, Path(image_dir_str), reuse_existing)


def render_or_reuse_image(row, image_dir: Path, reuse_existing: bool) -> str | None:
    return _render_image_core(int(row["ecg_id"]), str(row["filename_hr"]).strip(), image_dir, reuse_existing)


def build_split_json(
    split_name: str,
    split_df,
    image_dir: Path,
    output_dir: Path,
    reuse_existing_images: bool,
    label_codes: list[str],
    report_map: dict[int, str] | None = None,
    workers: int = 1,
) -> list[dict]:
    from tqdm import tqdm

    report_map = report_map or {}
    df = split_df.reset_index()
    image_dir.mkdir(parents=True, exist_ok=True)

    # ECG rendering is CPU-bound (matplotlib Agg) and embarrassingly parallel:
    # one process per CPU core. GPU does not accelerate line-plot rasterisation.
    tasks = [
        (int(r["ecg_id"]), str(r["filename_hr"]).strip(), str(image_dir), reuse_existing_images)
        for _, r in df.iterrows()
    ]
    rendered: dict[int, str | None] = {}
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for ecg_id, image_name in tqdm(
                pool.map(_render_worker, tasks),
                total=len(tasks), desc=f"Rendering {split_name} (x{workers})",
            ):
                rendered[ecg_id] = image_name
    else:
        for ecg_id, filename_hr, image_dir_str, reuse in tqdm(tasks, desc=f"Rendering {split_name}"):
            rendered[ecg_id] = _render_image_core(ecg_id, filename_hr, Path(image_dir_str), reuse)

    samples = []
    for _, row in df.iterrows():
        ecg_id = int(row["ecg_id"])
        image_name = rendered.get(ecg_id)
        if image_name is None:
            continue
        vector = [int(row[code]) for code in label_codes]
        samples.append(
            build_stage2_sample(
                ecg_id=ecg_id,
                fold=int(row["strat_fold"]),
                image_name=image_name,
                label_codes=label_codes,
                scp_vector=vector,
                interpretation=report_map.get(ecg_id, ""),
            )
        )

    output_path = output_dir / f"{split_name}.json"
    output_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    log.info("Saved %s samples: %d -> %s", split_name, len(samples), output_path)
    return samples


def build_stage2_dataset(
    output_dir: Path = DEFAULT_STAGE2_DIR,
    image_dir: Path = DEFAULT_STAGE2_IMAGE_DIR,
    limit: int | None = None,
    reuse_existing_images: bool = True,
    workers: int = 1,
) -> dict[str, list[dict]]:
    from preprocessing_pipeline.preprocessing.label.multilabel_encoder import build_label_dataframe
    from preprocessing_pipeline.preprocessing.label.report_translator import build_translated_report_map
    from preprocessing_pipeline.preprocessing.split.fold_splitter import build_split_manifests
    from preprocessing_pipeline.run_preprocessing import load_and_filter_metadata

    output_dir = Path(output_dir)
    image_dir = Path(image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_and_filter_metadata(limit=limit)
    label_df = build_label_dataframe(metadata_df)
    splits = build_split_manifests(metadata_df, label_df, output_dir=output_dir / "splits")
    write_label_vocab(output_dir / DEFAULT_STAGE2_LABEL_VOCAB.name, SCP_ALL_CODES)

    # Translate German PTB-XL reports → English (cached) to use as the
    # per-record Interpretation prefix. Built from metadata_df directly so the
    # split manifests (which drop the report column) do not need modifying.
    report_map = build_translated_report_map(
        metadata_df, cache_path=output_dir / "report_en_cache.json"
    )

    return {
        name: build_split_json(
            split_name=name,
            split_df=df,
            image_dir=image_dir,
            output_dir=output_dir,
            reuse_existing_images=reuse_existing_images,
            label_codes=SCP_ALL_CODES,
            report_map=report_map,
            workers=workers,
        )
        for name, df in splits.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PULSE Stage-2 PTB-XL 71-label JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STAGE2_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_STAGE2_IMAGE_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-reuse-existing-images", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1,
        help="parallel image-render processes (default: all CPU cores; set 1 to disable)",
    )
    args = parser.parse_args()

    build_stage2_dataset(
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        limit=args.limit,
        reuse_existing_images=not args.no_reuse_existing_images,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
