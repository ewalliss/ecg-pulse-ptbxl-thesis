"""
Train / validation / test split builder for PTB-XL.

PTB-XL ships with 10 pre-computed stratified folds stored in the
``strat_fold`` column of ``ptbxl_database.csv``.  Following the evaluation
protocol of Strodthoff et al. (2021) @strodthoff2021ptbxl:

    fold 10  → test   set  (held out, never touched during training)
    fold  9  → validation set
    folds 1–8 → training set

This module builds and saves three CSV manifests:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv

Each manifest contains: ecg_id, filename_hr, strat_fold, and the five
binary label columns (NORM, MI, STTC, CD, HYP).
"""

from pathlib import Path

import pandas as pd

from preprocessing_pipeline.config.preprocessing_config import (
    SPLITS_DIR,
    TRAIN_FOLDS,
    VAL_FOLD,
    TEST_FOLD,
    SCP_ALL_CODES,
)
from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)


def build_split_manifests(
    metadata_df: pd.DataFrame,
    label_df: pd.DataFrame,
    output_dir: Path = SPLITS_DIR,
) -> dict[str, pd.DataFrame]:
    """Create and save train / val / test manifest CSVs.

    Parameters
    ----------
    metadata_df:
        Full ``ptbxl_database.csv`` loaded as a DataFrame, indexed by
        ``ecg_id``.  Must contain columns ``filename_hr`` and ``strat_fold``.
    label_df:
        Output of :func:`~preprocessing.label.multilabel_encoder.build_label_dataframe`,
        indexed by ``ecg_id``, columns = SCP_ALL_CODES.
    output_dir:
        Directory where the three CSV files are written.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"``, each value a DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge metadata and labels on ecg_id.
    cols_needed = ["ecg_id", "filename_hr", "strat_fold"]
    meta = metadata_df[cols_needed].copy()
    meta["ecg_id"] = meta["ecg_id"].astype(int)
    meta = meta.set_index("ecg_id")

    combined = meta.join(label_df, how="inner")

    # Remove records with no positive label (background / artefact only).
    label_sum = combined[SCP_ALL_CODES].sum(axis=1)
    n_before = len(combined)
    combined = combined[label_sum > 0]
    log.info(
        "Dropped %d records with no diagnostic label (%d remaining).",
        n_before - len(combined),
        len(combined),
    )

    train = combined[combined["strat_fold"].isin(TRAIN_FOLDS)]
    val = combined[combined["strat_fold"] == VAL_FOLD]
    test = combined[combined["strat_fold"] == TEST_FOLD]

    splits = {"train": train, "val": val, "test": test}

    for name, df in splits.items():
        path = output_dir / f"{name}.csv"
        df.reset_index().to_csv(path, index=False)
        log.info(
            "Saved %s split: %d records → %s",
            name, len(df), path,
        )

    _log_label_statistics(splits)
    return splits


def _log_label_statistics(splits: dict[str, pd.DataFrame]) -> None:
    """Print top-10 label prevalence per split as a sanity check."""
    for name, df in splits.items():
        total = len(df)
        if total == 0:
            log.info("Label prevalence — %s: (empty)", name)
            continue
        counts = df[SCP_ALL_CODES].sum().nlargest(10)
        lines = [f"  {c}: {int(counts[c])}/{total} ({100*counts[c]/total:.1f}%)" for c in counts.index]
        log.info("Label prevalence top-10 — %s:\n%s", name, "\n".join(lines))
