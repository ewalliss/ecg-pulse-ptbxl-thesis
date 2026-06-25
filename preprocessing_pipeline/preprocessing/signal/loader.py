"""
WFDB record loader for PTB-XL ECG signals.

PTB-XL stores signals in WFDB format (.dat + .hea pairs).  This module
provides a thin wrapper around ``wfdb.rdsamp`` that returns a normalised
numpy array ready for downstream filtering.

Output shape: (n_leads, n_samples) = (12, 5000) at 500 Hz / 10 s.
"""

from pathlib import Path

import numpy as np
import wfdb

from preprocessing_pipeline.config.preprocessing_config import (
    DATASET_ROOT,
    SIGNAL_SAMPLING_RATE,
)
from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)

EXPECTED_LEADS = 12
EXPECTED_SAMPLES = SIGNAL_SAMPLING_RATE * 10   # 10-second recording → 5 000


def load_wfdb_record(record_path: str | Path) -> np.ndarray:
    """Load a single PTB-XL WFDB record as a float32 array.

    Parameters
    ----------
    record_path:
        Path to the record *without* extension, relative to ``DATASET_ROOT``
        or absolute.  Example: ``"records500/00000/00001_hr"``.

    Returns
    -------
    np.ndarray of shape (12, 5000), dtype float32, units millivolts (mV).

    Raises
    ------
    FileNotFoundError
        If the .dat or .hea file does not exist.
    ValueError
        If the loaded signal has unexpected shape.
    """
    path = Path(record_path)
    if not path.is_absolute():
        path = DATASET_ROOT / path

    # wfdb expects the path without extension.
    record_str = str(path.with_suffix(""))

    signal, meta = wfdb.rdsamp(record_str)
    # signal: (n_samples, n_leads), physical units (mV)

    if signal.shape != (EXPECTED_SAMPLES, EXPECTED_LEADS):
        raise ValueError(
            f"Unexpected signal shape {signal.shape} for record {record_str}. "
            f"Expected ({EXPECTED_SAMPLES}, {EXPECTED_LEADS})."
        )

    # Transpose → (n_leads, n_samples) = (12, 5000)
    return signal.T.astype(np.float32)


def load_wfdb_record_safe(
    record_path: str | Path,
) -> np.ndarray | None:
    """Like :func:`load_wfdb_record` but returns ``None`` on any error."""
    try:
        return load_wfdb_record(record_path)
    except Exception as exc:
        log.warning("Failed to load record %s: %s", record_path, exc)
        return None
