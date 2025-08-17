# src/audio/postfx.py
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt

def _butter_filter(x: np.ndarray, sr: int, cutoff: float, btype: str, order: int = 4) -> np.ndarray:
    ny = 0.5 * sr
    wc = np.clip(cutoff / ny, 1e-6, 0.999999)
    b, a = butter(order, wc, btype=btype)
    # filtfilt for zero-phase
    return filtfilt(b, a, x).astype(np.float32)

def highpass(x: np.ndarray, sr: int, cutoff_hz: int) -> np.ndarray:
    if cutoff_hz <= 0:
        return x
    return _butter_filter(x, sr, cutoff_hz, btype="highpass")

def lowpass(x: np.ndarray, sr: int, cutoff_hz: int) -> np.ndarray:
    if cutoff_hz <= 0:
        return x
    return _butter_filter(x, sr, cutoff_hz, btype="lowpass")

def tanh_limiter(x: np.ndarray, drive: float = 1.6) -> np.ndarray:
    # Soft saturation; robust to outliers
    y = np.tanh(np.clip(drive, 0.1, 10.0) * x)
    return y.astype(np.float32)

def normalize_peak(x: np.ndarray, target: float = 0.90) -> np.ndarray:
    target = float(np.clip(target, 0.0, 1.0))
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak == 0.0 or target == 0.0:
        return x
    return (x * (target / peak)).astype(np.float32)
