"""
SCP code parser for PTB-XL.

Responsibilities
----------------
1. Parse the `scp_codes` column of `ptbxl_database.csv` (stored as a Python
   dict literal string, e.g. ``"{'NORM': 100.0, 'LVOLT': 0.0}"``).
2. Return, for each record, the set of SCP codes whose confidence score
   meets the configured threshold — across all 71 codes in the vocabulary,
   not limited to the 5 diagnostic superclasses.
"""

import ast

from preprocessing_pipeline.config.preprocessing_config import (
    SCP_ALL_CODES,
    SCP_CONFIDENCE_THRESHOLD,
)
from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)

# Build a fast lookup set from the canonical vocabulary.
_SCP_VOCABULARY: frozenset[str] = frozenset(SCP_ALL_CODES)


def parse_scp_codes_column(raw: str) -> dict[str, float]:
    """Parse the scp_codes string stored in ptbxl_database.csv.

    The column contains a Python dict literal such as::

        {'NORM': 100.0, 'LVOLT': 0.0}

    Parameters
    ----------
    raw:
        Raw string value from the CSV cell.

    Returns
    -------
    dict mapping SCP code → confidence score (0–100).
    """
    try:
        parsed = ast.literal_eval(str(raw).strip())
        return {str(k).strip(): float(v) for k, v in parsed.items()}
    except (ValueError, SyntaxError):
        return {}


def extract_positive_scp_codes(
    scp_codes: dict[str, float],
    threshold: float = SCP_CONFIDENCE_THRESHOLD,
) -> list[str]:
    """Return the SCP codes that are positively labelled in this record.

    A code is *positive* if its confidence score ≥ ``threshold`` AND
    the code belongs to the 71-code SCP vocabulary.

    Parameters
    ----------
    scp_codes:
        Output of :func:`parse_scp_codes_column`.
    threshold:
        Minimum confidence (%) to consider a code positive.

    Returns
    -------
    List of positive SCP code strings, in vocabulary order.
    """
    positive: set[str] = set()
    for code, confidence in scp_codes.items():
        if confidence >= threshold and code in _SCP_VOCABULARY:
            positive.add(code)
    # Return in canonical vocabulary order for reproducibility.
    return [c for c in SCP_ALL_CODES if c in positive]
