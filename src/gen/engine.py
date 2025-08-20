
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import csv
import math
import random
import numpy as np
import torch

from audiocraft.models import MusicGen

from src.adapters.blend import apply_adapter_blend
from src.audio.postfx import highpass, lowpass, tanh_limiter, normalize_peak
from src.audio.metrics import compute_all_metrics

_SR_DEFAULT = 32000


_BASE_CACHE: Dict[str, Any] = {
    "model": None,      
    "pristine": None,   
    "sr": _SR_DEFAULT,
}

def _deep_clone_state_dict(sd: dict) -> dict:
    cloned = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            cloned[k] = v.detach().clone()
        else:
            cloned[k] = v
    return cloned

def _get_base_model() -> tuple[MusicGen, dict, int]:
    """
    Returns (model, pristine_lm_state_dict, sample_rate). Model runs on CPU by default.
    We cache the LM (transformer) weights only, not the wrapper.
    """
    global _BASE_CACHE
    if _BASE_CACHE["model"] is None:
        model = MusicGen.get_pretrained("facebook/musicgen-small")


        model.set_generation_params(use_sampling=True)  # will override per-call
        
        pristine_lm = _deep_clone_state_dict(model.lm.state_dict())
        sr = getattr(model, "sample_rate", _SR_DEFAULT)
        _BASE_CACHE.update({"model": model, "pristine": pristine_lm, "sr": sr})
    return _BASE_CACHE["model"], _BASE_CACHE["pristine"], _BASE_CACHE["sr"]


def render_one_clip(
    *,
    keyword: str,
    adapter_name: str,                # "base" | "neutral" | "calm"
    adapter_path: Optional[str],      # None for base
    alpha: float,                     # adapter blend
    duration_sec: int,
    temperature: float,
    top_k: int,
    top_p: float,
    cfg_coef: float,
    seed: int,
    highpass_hz: int,
    lowpass_hz: int,
    limiter_drive: float,
    peak_target: float,
    out_wav: str,
    metrics_csv: str,
    log_level: str = "INFO",
) -> bool:
    # seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if adapter_name != "base":
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0 (got {alpha})")
        if alpha > 0.05:
            raise ValueError(f"alpha={alpha:.3f} is too high; refuse to proceed (>0.05).")
        if alpha > 0.02:
            print(f"[WARN] alpha={alpha:.3f} is above recommended range (0.01–0.015). Proceed with caution.")

   
    model, pristine_lm, sr = _get_base_model()
    model.lm.load_state_dict(pristine_lm, strict=True)  # full reset of LM

   
    if adapter_name != "base":
        if not adapter_path:
            raise FileNotFoundError(f"Adapter '{adapter_name}' requested, but adapter_path is empty.")
        scope = "lm"
        matched, _ = apply_adapter_blend(model.lm, adapter_path, alpha)
        if matched == 0:
            
            scope = "model"
            matched, _ = apply_adapter_blend(model, adapter_path, alpha)
        print(f"keys matched and blended: {matched} (scope={scope})")
        if matched == 0:
            raise RuntimeError("Adapter apply resulted in matched=0 — incompatible or wrong base/config.")


    model.set_generation_params(
        use_sampling=True,
        duration=duration_sec,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_coef=cfg_coef,
    )


    wavs = model.generate([keyword], progress=False)
    wav = wavs[0].detach().cpu().numpy().astype(np.float32)
    if wav.ndim == 2:
        wav = wav[0]
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

    if highpass_hz and highpass_hz > 0:
        wav = highpass(wav, sr, highpass_hz)
    if lowpass_hz and lowpass_hz > 0:
        wav = lowpass(wav, sr, lowpass_hz)
    wav = tanh_limiter(wav, drive=limiter_drive)
    wav = normalize_peak(wav, target=peak_target)


    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(out_path, wav, sr)

    m = compute_all_metrics(wav, sr)
    row = {
        "adapter": adapter_name,
        "alpha": float(alpha if adapter_name != "base" else 0.0),
        "seed": int(seed),
        "keyword": keyword,
        "duration_sec": int(duration_sec),
        **m,
        "sr": int(sr),
        "file": str(out_path),
    }
    _append_metrics_csv(metrics_csv, row)

    return True


# ---- Helpers ----------------------------------------------------------------
def _write_wav(path: Path, audio: np.ndarray, sr: int):
    import soundfile as sf
    # Keep float32 container; many evaluation tools prefer that
    sf.write(str(path), audio, sr, subtype="PCM_16")

def _append_metrics_csv(csv_path: str, row: dict):
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    field_order = [
        "adapter", "alpha", "seed", "keyword", "duration_sec",
        "rms", "peak", "crest_db", "zcr",
        "centroid_hz", "hf_ratio", "flatness",
        "sr", "file",
    ]
    write_header = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in field_order})
