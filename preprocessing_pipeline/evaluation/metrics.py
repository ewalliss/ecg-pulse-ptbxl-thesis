"""
Evaluation metrics for multi-label ECG classification on PTB-XL fold 10.

All functions operate on NumPy arrays of shape (N, 71) for predicted
probabilities and binary ground-truth labels.

Metrics implemented:
- Macro AUC (roc_auc_score, macro-averaged across 71 labels)
- Macro F1 with per-label threshold optimization on val set
- Hamming loss
- Per-superclass AUC (diagnostic / form / rhythm groups from scp_statements.csv)

Baselines from thesis Table 2.1:
    xresnet1d101:  AUC 0.925  (signal-based upper bound)
    PULSE zero-shot: AUC 0.824
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss

from preprocessing_pipeline.config.preprocessing_config import SCP_ALL_CODES, SCP_STATEMENTS_CSV
from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)

NUM_LABELS = len(SCP_ALL_CODES)
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_macro_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Compute macro-averaged ROC AUC across all 71 labels.

    Labels with no positive examples in y_true are skipped (sklearn default).

    Parameters
    ----------
    y_true:
        Binary ground-truth, shape (N, 71).
    y_prob:
        Predicted probabilities in [0, 1], shape (N, 71).

    Returns
    -------
    float — macro AUC score.
    """
    _validate_shapes(y_true, y_prob)
    valid_cols = np.where(y_true.sum(axis=0) > 0)[0]
    if len(valid_cols) == 0:
        raise ValueError("y_true has no positive labels — cannot compute AUC.")
    if len(valid_cols) < NUM_LABELS:
        log.warning(
            "Skipping %d all-zero label columns for AUC (no positive examples).",
            NUM_LABELS - len(valid_cols),
        )
    auc = roc_auc_score(y_true[:, valid_cols], y_prob[:, valid_cols], average="macro")
    log.info("Macro AUC: %.4f  (over %d labels)", auc, len(valid_cols))
    return float(auc)


def compute_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute macro-averaged F1 from binary predictions.

    Parameters
    ----------
    y_true:
        Binary ground-truth, shape (N, 71).
    y_pred:
        Binary predictions (0/1), shape (N, 71).

    Returns
    -------
    float — macro F1 score.
    """
    _validate_shapes(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    log.info("Macro F1: %.4f", f1)
    return float(f1)


def compute_hamming_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute Hamming loss (fraction of incorrectly predicted labels).

    Parameters
    ----------
    y_true:
        Binary ground-truth, shape (N, 71).
    y_pred:
        Binary predictions (0/1), shape (N, 71).

    Returns
    -------
    float — Hamming loss in [0, 1].
    """
    _validate_shapes(y_true, y_pred)
    hl = hamming_loss(y_true, y_pred)
    log.info("Hamming loss: %.6f", hl)
    return float(hl)


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------

def optimize_thresholds(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    n_thresholds: int = 50,
) -> np.ndarray:
    """Find per-label classification thresholds maximizing F1 on a val set.

    Searches linearly spaced thresholds in [0.05, 0.95] per label.

    Parameters
    ----------
    y_true_val:
        Binary ground-truth on validation set, shape (N_val, 71).
    y_prob_val:
        Predicted probabilities on validation set, shape (N_val, 71).
    n_thresholds:
        Number of candidate thresholds to search per label.

    Returns
    -------
    np.ndarray of shape (71,) with optimal threshold per label.
    """
    _validate_shapes(y_true_val, y_prob_val)
    candidates = np.linspace(0.05, 0.95, n_thresholds)
    thresholds = np.full(NUM_LABELS, DEFAULT_THRESHOLD)

    for k in range(NUM_LABELS):
        if y_true_val[:, k].sum() == 0:
            continue
        best_f1 = -1.0
        for t in candidates:
            pred = (y_prob_val[:, k] >= t).astype(int)
            f1 = f1_score(y_true_val[:, k], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                thresholds[k] = t

    log.info(
        "Threshold optimization complete. Mean=%.3f  Min=%.3f  Max=%.3f",
        thresholds.mean(), thresholds.min(), thresholds.max(),
    )
    return thresholds


def apply_thresholds(
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Convert probabilities to binary predictions using per-label thresholds.

    Parameters
    ----------
    y_prob:
        Predicted probabilities, shape (N, 71).
    thresholds:
        Per-label thresholds, shape (71,).

    Returns
    -------
    np.ndarray of shape (N, 71) with dtype int8.
    """
    return (y_prob >= thresholds[np.newaxis, :]).astype(np.int8)


# ---------------------------------------------------------------------------
# Per-superclass breakdown
# ---------------------------------------------------------------------------

def load_superclass_groups(
    scp_statements_csv: Path = SCP_STATEMENTS_CSV,
) -> dict[str, list[int]]:
    """Load label index groups by diagnostic_class from scp_statements.csv.

    Returns a dict mapping group name → list of column indices into SCP_ALL_CODES.
    Groups: 'diagnostic', 'form', 'rhythm', and one per diagnostic_class value.

    Parameters
    ----------
    scp_statements_csv:
        Path to PTB-XL's scp_statements.csv.

    Returns
    -------
    dict[str, list[int]] — group name → label column indices.
    """
    code_to_idx = {c: i for i, c in enumerate(SCP_ALL_CODES)}
    groups: dict[str, list[int]] = {"diagnostic": [], "form": [], "rhythm": []}
    subgroups: dict[str, list[int]] = {}

    with open(scp_statements_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            code = row.get("", row.get("Unnamed: 0", "")).strip()
            if code not in code_to_idx:
                continue
            idx = code_to_idx[code]

            diagnostic = row.get("diagnostic", "").strip()
            form = row.get("form", "").strip()
            rhythm = row.get("rhythm", "").strip()
            diag_class = row.get("diagnostic_class", "").strip()

            if diagnostic == "1":
                groups["diagnostic"].append(idx)
            if form == "1":
                groups["form"].append(idx)
            if rhythm == "1":
                groups["rhythm"].append(idx)
            if diag_class:
                subgroups.setdefault(diag_class, []).append(idx)

    groups.update(subgroups)
    log.info(
        "Loaded superclass groups: %s",
        {k: len(v) for k, v in groups.items()},
    )
    return groups


def compute_per_superclass_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: dict[str, list[int]] | None = None,
) -> dict[str, float]:
    """Compute macro AUC per superclass group.

    Parameters
    ----------
    y_true:
        Binary ground-truth, shape (N, 71).
    y_prob:
        Predicted probabilities, shape (N, 71).
    groups:
        Output of :func:`load_superclass_groups`. Loaded from CSV if None.

    Returns
    -------
    dict mapping group name → macro AUC for that group's labels.
    """
    if groups is None:
        groups = load_superclass_groups()

    results: dict[str, float] = {}
    for group_name, indices in groups.items():
        if not indices:
            continue
        cols = np.array(indices)
        valid = np.where(y_true[:, cols].sum(axis=0) > 0)[0]
        if len(valid) == 0:
            log.warning("Group '%s': no positive labels — skipping.", group_name)
            continue
        auc = roc_auc_score(
            y_true[:, cols[valid]],
            y_prob[:, cols[valid]],
            average="macro",
        )
        results[group_name] = float(auc)
        log.info("  AUC [%s]: %.4f  (%d labels)", group_name, auc, len(valid))

    return results


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def compute_full_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> dict:
    """Compute all benchmark metrics and return as a serializable dict.

    Parameters
    ----------
    y_true:
        Binary ground-truth, shape (N, 71).
    y_prob:
        Predicted probabilities, shape (N, 71).
    thresholds:
        Per-label thresholds shape (71,). Uses DEFAULT_THRESHOLD if None.
    label_names:
        SCP code names for per-label breakdown. Uses SCP_ALL_CODES if None.

    Returns
    -------
    dict with keys: macro_auc, macro_f1, hamming_loss, per_label_auc,
    per_superclass_auc, n_samples, n_labels, thresholds_used.
    """
    if thresholds is None:
        thresholds = np.full(NUM_LABELS, DEFAULT_THRESHOLD)
    if label_names is None:
        label_names = SCP_ALL_CODES

    y_pred = apply_thresholds(y_prob, thresholds)
    groups = load_superclass_groups()

    per_label_auc: dict[str, float] = {}
    for k, name in enumerate(label_names):
        if y_true[:, k].sum() > 0:
            per_label_auc[name] = float(
                roc_auc_score(y_true[:, k], y_prob[:, k])
            )

    report = {
        "n_samples": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]),
        "macro_auc": compute_macro_auc(y_true, y_prob),
        "macro_f1": compute_macro_f1(y_true, y_pred),
        "hamming_loss": compute_hamming_loss(y_true, y_pred),
        "per_label_auc": per_label_auc,
        "per_superclass_auc": compute_per_superclass_auc(y_true, y_prob, groups),
        "thresholds_used": thresholds.tolist(),
    }
    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_shapes(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: y_true {a.shape} vs y_pred/prob {b.shape}")
    if a.ndim != 2 or a.shape[1] != NUM_LABELS:
        raise ValueError(f"Expected shape (N, {NUM_LABELS}), got {a.shape}")
