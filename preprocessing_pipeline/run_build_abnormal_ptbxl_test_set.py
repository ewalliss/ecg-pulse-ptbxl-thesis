from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from preprocessing_pipeline.config.preprocessing_config import OUTPUT_ROOT, SCP_STATEMENTS_CSV
from preprocessing_pipeline.preprocessing.image.ecg_renderer import render_ecg_image
from preprocessing_pipeline.preprocessing.label.scp_parser import (
    extract_positive_scp_codes,
    parse_scp_codes_column,
)
from preprocessing_pipeline.preprocessing.signal.filter import apply_bandpass_filter
from preprocessing_pipeline.preprocessing.signal.loader import load_wfdb_record_safe
from preprocessing_pipeline.run_preprocessing import load_and_filter_metadata

DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "ptbxl_abnormal_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render seven PTB-XL abnormal ECG images with distinct target SCP codes"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where rendered images and manifest.json will be written",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=7,
        help="Number of distinct abnormal target codes to render",
    )
    return parser.parse_args()


def _load_abnormal_diagnostic_codes() -> dict[str, str]:
    with SCP_STATEMENTS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            row[""].strip(): row["description"].strip()
            for row in reader
            if row["diagnostic"].strip() == "1.0" and row[""].strip() != "NORM"
        }


def _build_candidate_pools(metadata_df, abnormal_codes: set[str]) -> tuple[Counter, dict[str, list[dict]]]:
    counts: Counter[str] = Counter()
    pools = {code: [] for code in abnormal_codes}

    for row in metadata_df.to_dict("records"):
        scp_scores = parse_scp_codes_column(row["scp_codes"])
        positive_codes = [
            code
            for code in extract_positive_scp_codes(scp_scores)
            if code in abnormal_codes and code != "NORM"
        ]
        if not positive_codes:
            continue

        positive_confidences = {
            code: float(scp_scores[code])
            for code in positive_codes
        }
        for code in positive_codes:
            counts[code] += 1
            pools[code].append(
                {
                    "ecg_id": int(row["ecg_id"]),
                    "filename_hr": str(row["filename_hr"]).strip(),
                    "positive_codes": positive_codes,
                    "positive_confidences": positive_confidences,
                    "target_confidence": float(scp_scores[code]),
                }
            )

    for code, candidates in pools.items():
        candidates.sort(
            key=lambda candidate: (
                len(candidate["positive_codes"]),
                -candidate["target_confidence"],
                candidate["ecg_id"],
            )
        )

    return counts, pools


def build_abnormal_test_set(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    count: int = 7,
) -> list[dict]:
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_and_filter_metadata()
    abnormal_descriptions = _load_abnormal_diagnostic_codes()
    counts, candidate_pools = _build_candidate_pools(
        metadata_df,
        set(abnormal_descriptions),
    )

    manifest: list[dict] = []
    used_ecg_ids: set[int] = set()

    for code, _ in counts.most_common():
        if len(manifest) >= count:
            break

        for candidate in candidate_pools[code]:
            ecg_id = candidate["ecg_id"]
            if ecg_id in used_ecg_ids:
                continue

            signal = load_wfdb_record_safe(candidate["filename_hr"])
            if signal is None:
                continue

            filtered_signal = apply_bandpass_filter(signal)
            image_name = f"{len(manifest) + 1:02d}_{code}_{ecg_id:05d}.png"
            render_ecg_image(filtered_signal, image_dir / image_name, ecg_id=ecg_id)

            manifest.append(
                {
                    "ecg_id": ecg_id,
                    "target_code": code,
                    "target_description": abnormal_descriptions[code],
                    "image": f"images/{image_name}",
                    "positive_codes": candidate["positive_codes"],
                    "positive_confidences": candidate["positive_confidences"],
                    "record_path": candidate["filename_hr"],
                }
            )
            used_ecg_ids.add(ecg_id)
            break

    if len(manifest) < count:
        raise RuntimeError(
            f"Only rendered {len(manifest)} abnormal ECGs with distinct target codes; requested {count}."
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_abnormal_test_set(output_dir=args.output_dir, count=args.count)
    print(f"Rendered {len(manifest)} abnormal ECG images to {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")
    for sample in manifest:
        print(
            f"- {sample['target_code']}: ecg_id={sample['ecg_id']} -> {sample['image']}"
        )


if __name__ == "__main__":
    main()
