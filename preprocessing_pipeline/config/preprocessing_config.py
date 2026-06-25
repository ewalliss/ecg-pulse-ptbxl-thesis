"""
Preprocessing configuration for the PTB-XL multi-label ECG classification pipeline.

All tunable constants live here so that run_preprocessing.py and sub-modules
import from a single source of truth.
"""

from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Project root — auto-detected or overridden via environment variable
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(
    os.environ.get(
        "ECGLLAVA_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

# Override on each machine via the ECGLLAVA_DATASET_ROOT env var (Windows/local);
# the default keeps Colab Drive working unchanged.
#   Windows PowerShell:  $env:ECGLLAVA_DATASET_ROOT = "D:\path\to\ptb-xl"
DATASET_ROOT = Path(
    os.environ.get(
        "ECGLLAVA_DATASET_ROOT",
        "/content/drive/MyDrive/Colab Notebooks/NHOM5/graduation_thesis/data/ptb-xl",
    )
)

PTBXL_DB_CSV = DATASET_ROOT / "ptbxl_database.csv"
SCP_STATEMENTS_CSV = DATASET_ROOT / "scp_statements.csv"

# Use 500 Hz records (filename_hr) for best temporal resolution.
SIGNAL_SAMPLING_RATE = 500          # Hz
SIGNAL_RECORD_SUBDIR = "records500"

# Output root — all generated artefacts land here.
OUTPUT_ROOT = Path(os.environ.get("ECGLLAVA_OUTPUT_ROOT", str(_PROJECT_ROOT / "data")))
SIGNAL_NPY_DIR = OUTPUT_ROOT / "signals"    # .npy files, shape (12, 5000)
ECG_IMAGE_DIR = OUTPUT_ROOT / "images"      # ECG paper PNGs
SPLITS_DIR = OUTPUT_ROOT / "splits"         # train/val/test CSV manifests

# ---------------------------------------------------------------------------
# Signal pre-processing
# ---------------------------------------------------------------------------

# Butterworth bandpass — removes baseline wander and high-freq noise.
BANDPASS_LOWCUT_HZ = 0.5
# 150 Hz matches the clinical diagnostic bandwidth (AHA/ACC) and is consistent
# with the Nyquist argument in thesis §3.1.1.1: f_max = 150 Hz requires
# f_s ≥ 300 Hz, which is satisfied by the 500 Hz PTB-XL records.
# Cutting at 40 Hz would discard the 40–150 Hz band that contains pacemaker
# spikes and the sharp edges of the QRS complex.
BANDPASS_HIGHCUT_HZ = 150.0
BANDPASS_ORDER = 4

# Per-lead z-score normalisation.
NORMALISE_PER_LEAD = True

# ---------------------------------------------------------------------------
# ECG image rendering
# ---------------------------------------------------------------------------

# Standard ECG paper: 25 mm/s horizontal, 10 mm/mV vertical.
ECG_PAPER_SPEED_MM_PER_S = 25.0    # determines x-axis scale
ECG_PAPER_GAIN_MM_PER_MV = 10.0    # determines y-axis scale

# Output pixel target: 1344 × 672 px (2:1 aspect ratio).
# At 200 DPI: 1344 / 200 = 6.72 in wide, 672 / 200 = 3.36 in tall.
# The spatial sampling rate ≈ 196.9 px/s satisfies Nyquist for ECG ≤ 100 Hz
# (thesis §3.1.1.1).
ECG_IMAGE_DPI = 200
ECG_IMAGE_WIDTH_INCHES = 6.72      # → 1344 px at 200 DPI
ECG_IMAGE_HEIGHT_INCHES = 3.36     # →  672 px at 200 DPI
ECG_IMAGE_OUTPUT_WIDTH_PX = 1344
ECG_IMAGE_OUTPUT_HEIGHT_PX = 672

# Lead layout: 4 columns × 3 rows (standard 12-lead view) + rhythm strip.
ECG_LEAD_NAMES = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6",
]
ECG_RHYTHM_LEAD_IDX = 1     # Lead II as rhythm strip

# ---------------------------------------------------------------------------
# Multi-label target
# ---------------------------------------------------------------------------

# Load the 44 DIAGNOSTIC SCP-ECG codes from scp_statements.csv at config import
# time (codes whose `diagnostic` flag is set). Form (19) and rhythm (12)
# statements are excluded: the thesis scopes to the diagnostic task, following
# the standard PTB-XL `diag` benchmark (Strodthoff et al. 2020). These 44 codes
# also map to the 5 diagnostic superclasses / 23 subclasses for hierarchical
# reporting (columns `diagnostic_class` / `diagnostic_subclass`).
#
# Order is determined by the row order in the CSV — fixed and reproducible.
# Never sort or shuffle: the positional index is the label column index in
# every binary vector produced by multilabel_encoder.py.
def _load_scp_codes_from_csv(csv_path: Path, diagnostic_only: bool = True) -> list[str]:
    """Load SCP-ECG codes from scp_statements.csv.

    With ``diagnostic_only=True`` (default) return only the 44 codes flagged as
    diagnostic; pass ``diagnostic_only=False`` to recover the full 71-code
    vocabulary (diagnostic + form + rhythm).
    """
    import csv as _csv
    codes: list[str] = []
    with open(csv_path, newline="") as fh:
        reader = _csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        diag_idx = header.index("diagnostic")
        for row in reader:
            if not (row and row[0].strip()):
                continue
            if diagnostic_only and not row[diag_idx].strip():
                continue
            codes.append(row[0].strip())
    return codes

# Active label set for the thesis: 44 diagnostic SCP-ECG codes.
SCP_ALL_CODES: list[str] = _load_scp_codes_from_csv(SCP_STATEMENTS_CSV, diagnostic_only=True)
NUM_CLASSES: int = len(SCP_ALL_CODES)   # 44 diagnostic codes

# Minimum confidence threshold for a code to be considered a positive label.
SCP_CONFIDENCE_THRESHOLD = 50.0    # percent (≥ 50 % → label is positive)

# Only keep records where at least one label passed the threshold AND
# the record was validated by a human cardiologist.
REQUIRE_HUMAN_VALIDATION = True

# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------

# PTB-XL ships with 10 stratified folds.
# Convention from Strodthoff et al. (2021): fold 10 → test, fold 9 → val,
# folds 1-8 → train.
TEST_FOLD = 10
VAL_FOLD = 9
TRAIN_FOLDS = list(range(1, 9))    # [1, 2, …, 8]
