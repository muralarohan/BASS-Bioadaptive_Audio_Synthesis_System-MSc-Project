# tests/test_adapter_blend.py
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from src.gen.engine import _get_base_model, render_one_clip
from src.adapters.blend import apply_adapter_blend
from src.audio.metrics import rms, peak

ADAPTER_PATH = Path("adapters/neutral/merged_state_dict.pt")

@pytest.mark.skipif(not ADAPTER_PATH.exists(), reason="neutral adapter file not found")
def test_adapter_blend_and_render(tmp_path: Path):
    # 1) Ensure we can blend into LM and get matched > 0
    model, pristine_lm, sr = _get_base_model()
    model.lm.load_state_dict(pristine_lm, strict=True)  # reset
    matched, _ = apply_adapter_blend(model.lm, str(ADAPTER_PATH), alpha=0.01)
    assert matched > 0, "Adapter blending matched zero tensors (wrong keys/base?)"

    # Restore pristine to avoid compounding for the render that follows
    model.lm.load_state_dict(pristine_lm, strict=True)

    # 2) Render a short clip with adapter enabled
    out_wav = tmp_path / "neutral_smoke.wav"
    metrics_csv = tmp_path / "metrics.csv"
    ok = render_one_clip(
        keyword="neutral piano arpeggio, slow tempo, no drums, soft reverb",
        adapter_name="neutral",
        adapter_path=str(ADAPTER_PATH),
        alpha=0.01,
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

    # 3) Basic sanity on output amplitude
    audio, sr = sf.read(str(out_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    assert rms(audio) > 0.005, "RMS too low after adapter blend"
    assert peak(audio) <= 1.0, "Peak exceeds 1.0 (post-FX normalization failed)"
