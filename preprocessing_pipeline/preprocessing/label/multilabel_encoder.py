"""
Multi-label binary encoder for all 71 PTB-XL SCP-ECG codes.

For each ECG record, builds a fixed-length binary vector of length 71,
one position per SCP code in ``SCP_ALL_CODES`` (config order).
Index i = 1 if the SCP code at position i has confidence ≥ threshold.

Why 71 codes, not 5 superclasses
---------------------------------
The 5-superclass (NORM / MI / STTC / CD / HYP) grouping collapses
clinically distinct conditions into the same bucket, losing the granularity
needed for fine-grained ECG interpretation.  For example, CRBBB (complete
right bundle branch block) and CLBBB (complete left bundle branch block)
are both in the CD class but require entirely different clinical management.
This thesis targets the full 71-label space, consistent with the ASL-Gen
loss formulation and the MERL zero-shot evaluation protocol (§1, §3.2).
"""

import numpy as np
import pandas as pd

from preprocessing_pipeline.config.preprocessing_config import SCP_ALL_CODES
from preprocessing_pipeline.preprocessing.label.scp_parser import (
    parse_scp_codes_column,
    extract_positive_scp_codes,
)
from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)

# Stable index lookup: SCP code → column position in the binary vector.
SCP_CODE_TO_INDEX: dict[str, int] = {
    code: idx for idx, code in enumerate(SCP_ALL_CODES)
}


def encode_scp_codes_to_binary_vector(
    positive_codes: list[str],
) -> np.ndarray:
    """Convert a list of positive SCP codes to a binary numpy vector.

    Parameters
    ----------
    positive_codes:
        Output of :func:`extract_positive_scp_codes` — codes in vocabulary
        order.

    Returns
    -------
    np.ndarray of shape (71,) and dtype int8.
    """
    vec = np.zeros(len(SCP_ALL_CODES), dtype=np.int8)
    for code in positive_codes:
        if code in SCP_CODE_TO_INDEX:
            vec[SCP_CODE_TO_INDEX[code]] = 1
    return vec


def build_label_dataframe(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame with one binary column per SCP code (71 total).

    Parameters
    ----------
    metadata_df:
        The ``ptbxl_database.csv`` loaded as a DataFrame (or filtered subset).
        Must contain the ``scp_codes`` column and ``ecg_id`` column.

    Returns
    -------
    DataFrame indexed by ``ecg_id`` with 71 columns (one per SCP code),
    values 0/1 (int8).
    """
    rows: list[dict] = []
    for _, row in metadata_df.iterrows():
        scp_codes = parse_scp_codes_column(row["scp_codes"])
        positives = extract_positive_scp_codes(scp_codes)
        vec = encode_scp_codes_to_binary_vector(positives)
        entry = {"ecg_id": int(row["ecg_id"])}
        for code, val in zip(SCP_ALL_CODES, vec):
            entry[code] = int(val)
        rows.append(entry)

    label_df = pd.DataFrame(rows).set_index("ecg_id")
    label_counts = label_df.sum()
    top10 = label_counts.nlargest(10)
    log.info(
        "Encoded %d-label vectors for %d records. Top-10 label counts:\n%s",
        len(SCP_ALL_CODES),
        len(label_df),
        top10.to_string(),
    )

    # Warn about codes that are permanently zero in this dataset — they appear
    # in scp_codes entries but PTB-XL always annotates them with confidence=0,
    # so they can never pass the threshold and will have no positive training signal.
    zero_codes = [c for c in SCP_ALL_CODES if label_counts[c] == 0]
    if zero_codes:
        log.warning(
            "%d SCP codes have ZERO positive labels in this dataset (always confidence=0 "
            "in PTB-XL — these codes carry no training signal):\n  %s",
            len(zero_codes),
            ", ".join(zero_codes),
        )

    return label_df
