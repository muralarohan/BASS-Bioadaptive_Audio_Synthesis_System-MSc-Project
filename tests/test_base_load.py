# tests
import numpy as np
import soundfile as sf
from pathlib import Path

from src.gen.engine import render_one_clip
from src.audio.metrics import rms

def test_base_load(tmp_path: Path):
    out_wav = tmp_path / "base_smoke.wav"
    metrics_csv = tmp_path / "metrics.csv"

    ok = render_one_clip(
        keyword="mellow acoustic guitar, relaxed tempo, soft dynamics",
        adapter_name="base",
        adapter_path=None,
        alpha=0.0,
        duration_sec=2,
        temperature=0.95,
        top_k=150,
        top_p=0.95,
        cfg_coef=1.6,
        seed=4242,
        highpass_hz=120,
        lowpass_hz=10000,
        limiter_drive=1.6,
        peak_target=0.90,
        out_wav=str(out_wav),
        metrics_csv=str(metrics_csv),
        log_level="INFO",
    )
    assert ok is True
    assert out_wav.exists(), "WAV was not created"

    audio, sr = sf.read(str(out_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    assert rms(audio) > 0.01, "RMS too low; base render likely failed"
