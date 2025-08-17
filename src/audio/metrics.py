# src/audio/metrics.py
from __future__ import annotations
import numpy as np

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))

def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))

def crest_db(x: np.ndarray) -> float:
    r = rms(x)
    p = peak(x)
    if r <= 1e-12:
        return 0.0
    return float(20.0 * np.log10((p + 1e-12) / (r + 1e-12)))

def zcr(x: np.ndarray) -> float:
    s = np.signbit(x)
    return float(np.mean(s[:-1] != s[1:])) if x.size > 1 else 0.0

def spectral_centroid_hz(x: np.ndarray, sr: int) -> float:
    mag = np.abs(np.fft.rfft(x))
    if mag.sum() <= 1e-12:
        return 0.0
    freqs = np.fft.rfftfreq(x.size, d=1.0/sr)
    return float((freqs * mag).sum() / (mag.sum() + 1e-12))

def hf_ratio(x: np.ndarray, sr: int, thresh_hz: int = 6000) -> float:
    mag = np.abs(np.fft.rfft(x))
    if mag.sum() <= 1e-12:
        return 0.0
    freqs = np.fft.rfftfreq(x.size, d=1.0/sr)
    hi = mag[freqs >= thresh_hz].sum()
    tot = mag.sum()
    return float(hi / (tot + 1e-12))

def spectral_flatness(x: np.ndarray) -> float:
    mag = np.abs(np.fft.rfft(x)) + 1e-12
    geo = np.exp(np.mean(np.log(mag)))
    arith = np.mean(mag)
    return float(geo / (arith + 1e-12))

def compute_all_metrics(x: np.ndarray, sr: int) -> dict:
    return {
        "rms": rms(x),
        "peak": peak(x),
        "crest_db": crest_db(x),
        "zcr": zcr(x),
        "centroid_hz": spectral_centroid_hz(x, sr),
        "hf_ratio": hf_ratio(x, sr, 6000),
        "flatness": spectral_flatness(x),
    }
