"""
ECG signal → ECG paper image renderer.

Produces a standard 12-lead ECG paper PNG image at 200 DPI.

Layout
------
The standard clinical layout places 12 leads in a 4-column × 3-row grid,
each 2.5 s wide.  A full-length Lead-II rhythm strip spans the bottom.

    Col 0   Col 1   Col 2   Col 3
Row 0:  I      aVR     V1      V4
Row 1: II      aVL     V2      V5
Row 2: III     aVF     V3      V6
Rhythm:       II  (full 10 s)

Paper scale
-----------
- Horizontal: 25 mm/s → each column is 62.5 mm wide.
- Vertical:   10 mm/mV → grid squares are 1 mV = 10 mm.
- At 200 DPI the spatial Nyquist frequency ≈ 196.9 px/s, satisfying the
  Nyquist criterion for ECG signals ≤ 100 Hz (thesis §3.1.1.1).

Reference: AHA/ANSI/AAMI EC11:2007 standard for ECG recorders.
"""

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from PIL import Image

from preprocessing_pipeline.config.preprocessing_config import (
    ECG_IMAGE_DPI,
    ECG_IMAGE_WIDTH_INCHES,
    ECG_IMAGE_HEIGHT_INCHES,
    ECG_IMAGE_OUTPUT_WIDTH_PX,
    ECG_IMAGE_OUTPUT_HEIGHT_PX,
    ECG_LEAD_NAMES,
    ECG_PAPER_SPEED_MM_PER_S,
    ECG_PAPER_GAIN_MM_PER_MV,
    SIGNAL_SAMPLING_RATE,
)
from preprocessing_pipeline.utils.logger import get_logger

matplotlib.use("Agg")   # headless rendering — no display required

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Layout constants (derived from config)
# ---------------------------------------------------------------------------

_N_COLS = 4
_N_ROWS = 3
_SAMPLES_PER_SEGMENT = SIGNAL_SAMPLING_RATE * 10 // _N_COLS  # 2.5 s = 1 250 samp.
_TIME_AXIS = np.linspace(0, 10 / _N_COLS, _SAMPLES_PER_SEGMENT)   # seconds

# Indices of the 12 leads in the standard column-major grid order:
#   Col 0: I(0), II(1), III(2)   Col 1: aVR(3), aVL(4), aVF(5)
#   Col 2: V1(6), V2(7), V3(8)  Col 3: V4(9), V5(10), V6(11)
_GRID_LEAD_INDICES = [
    [0, 3, 6, 9],   # row 0
    [1, 4, 7, 10],  # row 1
    [2, 5, 8, 11],  # row 2
]

# ECG paper background colour and grid colour.
_PAPER_BG = "#fff8f0"
_GRID_MAJOR = "#e8a0a0"
_GRID_MINOR = "#f2c8c8"
_SIGNAL_COLOR = "#000000"
_SIGNAL_LINEWIDTH = 0.7


def _draw_ecg_grid(ax: plt.Axes, x_end: float, y_center: float, amplitude_mv: float) -> None:
    """Draw ECG graph-paper grid on a single subplot axes."""
    # Major grid: 5 mm = 0.2 s horizontal, 0.5 mV vertical
    ax.set_facecolor(_PAPER_BG)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(which="major", color=_GRID_MAJOR, linewidth=0.5, zorder=1)
    ax.grid(which="minor", color=_GRID_MINOR, linewidth=0.2, zorder=1)
    ax.set_xlim(0, x_end)
    ax.set_ylim(y_center - amplitude_mv / 2, y_center + amplitude_mv / 2)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_ecg_image(
    signal: np.ndarray,
    output_path: Path,
    ecg_id: int | None = None,
) -> Path:
    """Render a 12-lead ECG paper image and save as PNG.

    Parameters
    ----------
    signal:
        Pre-processed ECG array of shape (12, 5000), float32, in mV.
        The signal should already be filtered; normalisation is NOT applied
        here so that the amplitude grid remains clinically meaningful.
    output_path:
        Destination .png file.  Parent directories are created automatically.
    ecg_id:
        Optional record identifier. Kept for the caller's filename/bookkeeping
        only; it is NOT drawn on the image (no title) to avoid a spurious cue.

    Returns
    -------
    Path to the saved PNG file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(
        figsize=(ECG_IMAGE_WIDTH_INCHES, ECG_IMAGE_HEIGHT_INCHES),
        dpi=ECG_IMAGE_DPI,
        facecolor=_PAPER_BG,
    )
    try:
        # ---- 12-lead grid (3 rows x 4 cols) --------------------------------
        n_total_rows = _N_ROWS + 1   # +1 for rhythm strip
        gs = fig.add_gridspec(
            n_total_rows, _N_COLS,
            hspace=0.05, wspace=0.02,
            top=0.93, bottom=0.04, left=0.02, right=0.99,
        )

        for row in range(_N_ROWS):
            for col in range(_N_COLS):
                lead_idx = _GRID_LEAD_INDICES[row][col]
                segment = signal[lead_idx, col * _SAMPLES_PER_SEGMENT: (col + 1) * _SAMPLES_PER_SEGMENT]

                ax = fig.add_subplot(gs[row, col])
                _draw_ecg_grid(ax, x_end=_TIME_AXIS[-1], y_center=0.0, amplitude_mv=3.0)
                ax.plot(_TIME_AXIS, segment, color=_SIGNAL_COLOR,
                        linewidth=_SIGNAL_LINEWIDTH, zorder=2)
                ax.text(
                    0.01, 0.92, ECG_LEAD_NAMES[lead_idx],
                    transform=ax.transAxes,
                    fontsize=6, fontweight="bold", color="#333333",
                    verticalalignment="top",
                )

        # ---- Rhythm strip (full 10 s Lead II) --------------------------------
        rhythm_ax = fig.add_subplot(gs[_N_ROWS, :])
        rhythm_time = np.linspace(0, 10.0, signal.shape[1])
        _draw_ecg_grid(rhythm_ax, x_end=10.0, y_center=0.0, amplitude_mv=3.0)
        rhythm_ax.plot(
            rhythm_time, signal[1],   # Lead II
            color=_SIGNAL_COLOR, linewidth=_SIGNAL_LINEWIDTH, zorder=2,
        )
        rhythm_ax.text(
            0.002, 0.92, "II",
            transform=rhythm_ax.transAxes,
            fontsize=6, fontweight="bold", color="#333333",
            verticalalignment="top",
        )

        # ---- No figure title -------------------------------------------------
        # The per-record "ECG-ID {id}" title is deliberately NOT drawn: it is a
        # non-clinical identifier that the model could memorise as a spurious cue
        # (an image-side confound). Renders stay free of any record identifier.

        fig.savefig(output_path, dpi=ECG_IMAGE_DPI)
    finally:
        plt.close(fig)

    # ---- PIL resize guard (cross-platform safety) -------------------------
    # Matplotlib on different backends may produce images off by 1-2 pixels.
    # Guarantee exact target dimensions with Lanczos resampling.
    with Image.open(output_path) as img:
        if img.size != (ECG_IMAGE_OUTPUT_WIDTH_PX, ECG_IMAGE_OUTPUT_HEIGHT_PX):
            log.debug(
                "Resizing ECG image %s from %s to (%d, %d)",
                output_path.name, img.size,
                ECG_IMAGE_OUTPUT_WIDTH_PX, ECG_IMAGE_OUTPUT_HEIGHT_PX,
            )
            img = img.resize(
                (ECG_IMAGE_OUTPUT_WIDTH_PX, ECG_IMAGE_OUTPUT_HEIGHT_PX),
                Image.LANCZOS,
            )
            img.save(output_path)

    return output_path
