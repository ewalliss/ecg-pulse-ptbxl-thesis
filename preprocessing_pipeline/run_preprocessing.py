"""
PTB-XL preprocessing pipeline — main entry point.

Execution order
---------------
1. Load ``ptbxl_database.csv`` and filter to human-validated records.
2. Build multi-label binary vectors for all 71 SCP-ECG codes.
3. Build train / val / test manifests (based on ``strat_fold``).
4. For each record: load WFDB signal → bandpass filter → save as .npy.
5. For each record: render filtered signal as ECG paper PNG (no normalisation).

Usage
-----
    cd /Users/dangnguyen/Workspace/graduation_thesis
    python -m src.run_preprocessing [--limit N] [--skip-images] [--skip-signals]

Options
-------
--limit N        Process only the first N records (useful for smoke tests).
--skip-images    Skip ECG image rendering.
--skip-signals   Skip signal .npy export.
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing_pipeline.config.preprocessing_config import (
    PTBXL_DB_CSV,
    SIGNAL_NPY_DIR,
    ECG_IMAGE_DIR,
    REQUIRE_HUMAN_VALIDATION,
    SCP_CONFIDENCE_THRESHOLD,
)
from preprocessing_pipeline.preprocessing.label.scp_parser import (
    parse_scp_codes_column,
    extract_positive_scp_codes,
)
from preprocessing_pipeline.preprocessing.label.multilabel_encoder import build_label_dataframe
from preprocessing_pipeline.preprocessing.signal.loader import load_wfdb_record_safe
from preprocessing_pipeline.preprocessing.signal.filter import apply_bandpass_filter
from preprocessing_pipeline.preprocessing.image.ecg_renderer import render_ecg_image
from preprocessing_pipeline.preprocessing.split.fold_splitter import build_split_manifests
from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)


def load_and_filter_metadata(limit: int | None = None) -> pd.DataFrame:
    """Load ptbxl_database.csv and apply quality filters.

    Filters applied:
    - ``validated_by_human == True`` (if REQUIRE_HUMAN_VALIDATION is set).
    - At least one diagnostic superclass label passes confidence threshold.

    Parameters
    ----------
    limit:
        If set, keep only the first ``limit`` rows after filtering.

    Returns
    -------
    Filtered DataFrame with ``ecg_id`` as a plain column (not index).
    """
    log.info("Loading metadata from %s", PTBXL_DB_CSV)
    # skipinitialspace=True is required: some fields (e.g. static_noise) have
    # a leading space before the opening quote, which prevents the C parser
    # from recognising the field as quoted.  Column names also need stripping.
    df = pd.read_csv(PTBXL_DB_CSV, index_col="ecg_id", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df.index.name = "ecg_id"
    df = df.reset_index()   # bring ecg_id back as a column
    df["ecg_id"] = df["ecg_id"].astype(int)
    log.info("Total records in database: %d", len(df))

    if REQUIRE_HUMAN_VALIDATION:
        # The column value has trailing whitespace: strip before comparison.
        validated = df["validated_by_human"].astype(str).str.strip()
        df = df[validated == "True"]
        log.info("After human-validation filter: %d records", len(df))

    # Pre-filter: keep records with at least one positive SCP code (≥ threshold).
    def has_any_positive_label(scp_raw: str) -> bool:
        codes = parse_scp_codes_column(scp_raw)
        return len(extract_positive_scp_codes(codes)) > 0

    df = df[df["scp_codes"].apply(has_any_positive_label)]
    log.info("After SCP label filter: %d records", len(df))

    if limit:
        # Sample across all strat_fold values so smoke tests include diverse
        # classes. head(limit) only returns early ecg_ids which are nearly
        # all NORM (ecg_id 1-209 contain no IMI records at all).
        df = df.groupby("strat_fold", group_keys=False).apply(
            lambda g: g.sample(
                n=max(1, limit // df["strat_fold"].nunique()),
                random_state=42,
            )
        ).head(limit)
        log.info("Limiting to %d records via stratified sample (--limit flag)", len(df))

    return df.reset_index(drop=True)


def process_signals(
    metadata_df: pd.DataFrame,
    skip_images: bool = False,
    skip_signals: bool = False,
) -> None:
    """Load, filter, normalise, and export signals and images.

    Parameters
    ----------
    metadata_df:
        Filtered metadata DataFrame with columns ``ecg_id`` and
        ``filename_hr``.
    skip_images:
        If True, skip PNG rendering.
    skip_signals:
        If True, skip .npy export.
    """
    SIGNAL_NPY_DIR.mkdir(parents=True, exist_ok=True)
    ECG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    n_total = len(metadata_df)
    n_ok = 0
    n_fail = 0

    for i, row in tqdm(metadata_df.iterrows(), total=n_total, desc="Processing ECG records", unit="rec"):
        ecg_id = int(row["ecg_id"])
        record_path = str(row["filename_hr"]).strip()

        # ---- Load WFDB record -------------------------------------------
        raw_signal = load_wfdb_record_safe(record_path)
        if raw_signal is None:
            n_fail += 1
            continue

        # ---- Bandpass filter (0.5–40 Hz) --------------------------------
        filtered_signal = apply_bandpass_filter(raw_signal)

        # ---- Save .npy (filtered, not normalised — normalise at train time)
        if not skip_signals:
            npy_path = SIGNAL_NPY_DIR / f"{ecg_id:05d}.npy"
            np.save(npy_path, filtered_signal)

        # ---- Save ECG paper image (filtered, original mV scale) ----------
        if not skip_images:
            img_path = ECG_IMAGE_DIR / f"{ecg_id:05d}.png"
            render_ecg_image(filtered_signal, img_path, ecg_id=ecg_id)

        n_ok += 1

    log.info(
        "Signal processing complete: %d OK, %d failed out of %d total.",
        n_ok, n_fail, n_total,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PTB-XL preprocessing pipeline")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N records (smoke test)")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip ECG image rendering")
    parser.add_argument("--skip-signals", action="store_true",
                        help="Skip signal .npy export")
    args = parser.parse_args()

    log.info("=== PTB-XL Preprocessing Pipeline ===")

    # Step 1: Load & filter metadata
    metadata_df = load_and_filter_metadata(limit=args.limit)

    # Step 2: Encode multi-labels
    label_df = build_label_dataframe(metadata_df)

    # Step 3: Build split manifests
    build_split_manifests(metadata_df, label_df)

    # Steps 4 & 5: Process signals and render images
    process_signals(
        metadata_df,
        skip_images=args.skip_images,
        skip_signals=args.skip_signals,
    )

    log.info("=== Preprocessing pipeline finished ===")


if __name__ == "__main__":
    main()
