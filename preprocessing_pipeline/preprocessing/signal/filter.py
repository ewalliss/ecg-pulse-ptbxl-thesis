"""
ECG signal filtering.

Applies a Butterworth bandpass filter to remove:
  - Baseline wander (< 0.5 Hz) caused by respiration and electrode movement.
  - Very high-frequency noise (> 150 Hz) beyond clinical diagnostic bandwidth.

The upper cutoff is set to 150 Hz (not 40 Hz) to preserve the full clinical
diagnostic bandwidth:
  - P-wave: 0.5–12 Hz
  - QRS complex: 10–40 Hz (main energy), with higher harmonics up to 150 Hz
  - Pacemaker spikes: impulsive, containing energy up to 150+ Hz

This is consistent with the Nyquist argument in thesis §3.1.1.1: the 500 Hz
PTB-XL sampling rate satisfies f_s ≥ 2 × 150 Hz = 300 Hz.  Cutting at 40 Hz
would contradict that Nyquist bound and discard diagnostically meaningful
high-frequency content (e.g. sharp QRS edges, pacemaker spikes).

Reference: AHA/ANSI/AAMI EC11:2007; Sörnmo & Laguna (2005) §3.3.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from preprocessing_pipeline.config.preprocessing_config import (
    BANDPASS_LOWCUT_HZ,
    BANDPASS_HIGHCUT_HZ,
    BANDPASS_ORDER,
    SIGNAL_SAMPLING_RATE,
)
from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)


def _build_butterworth_bandpass_sos(
    lowcut: float = BANDPASS_LOWCUT_HZ,
    highcut: float = BANDPASS_HIGHCUT_HZ,
    order: int = BANDPASS_ORDER,
    fs: float = SIGNAL_SAMPLING_RATE,
) -> np.ndarray:
    """Compute second-order sections (SOS) for the Butterworth bandpass.

    SOS representation is numerically more stable than the direct-form
    transfer function (b, a) for higher-order filters.

    Parameters
    ----------
    lowcut, highcut:
        Passband edges in Hz.
    order:
        Filter order.  Applied per half-band, so the effective order of the
        bandpass is 2 × order.
    fs:
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray — SOS matrix of shape (n_sections, 6).
    """
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


# Build once at import time — reused for every record.
_BANDPASS_SOS = _build_butterworth_bandpass_sos()


def apply_bandpass_filter(signal: np.ndarray) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass to an ECG signal.

    Uses :func:`scipy.signal.sosfiltfilt` (forward–backward pass) to achieve
    zero phase distortion — critical for preserving QRS timing.

    Parameters
    ----------
    signal:
        ECG array of shape (n_leads, n_samples), dtype float32.

    Returns
    -------
    Filtered array of same shape and dtype.
    """
    filtered = sosfiltfilt(_BANDPASS_SOS, signal, axis=-1)
    return filtered.astype(np.float32)
