# src/adapters/blend.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import torch

def _is_float_tensor(t: torch.Tensor) -> bool:
    return isinstance(t, torch.Tensor) and t.is_floating_point()

@torch.no_grad()
def blend_state_dicts_(base_sd: dict, adapter_sd: dict, alpha: float) -> int:
    """
    In-place weight-space blend:
        W_base := (1-α) * W_base + α * W_adapter
    Returns the number of matched (blended) tensors.
    """
    matched = 0
    for k, w_base in base_sd.items():
        w_ad = adapter_sd.get(k, None)
        if w_ad is None:
            continue
        if not (_is_float_tensor(w_base) and _is_float_tensor(w_ad)):
            continue
        if w_base.shape != w_ad.shape:
            continue
        # Ensure same device/dtype
        if w_ad.dtype != w_base.dtype:
            w_ad = w_ad.to(dtype=w_base.dtype)
        if w_ad.device != w_base.device:
            w_ad = w_ad.to(device=w_base.device)
        # Blend in place
        w_base.mul_(1.0 - alpha).add_(w_ad, alpha=alpha)
        matched += 1
    return matched

def load_adapter_state(adapter_path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """
    Loads an adapter 'merged_state_dict' (full model weights after LoRA merge).
    Expected: keys align with the base MusicGen-small model.
    """
    p = Path(adapter_path)
    if not p.exists():
        raise FileNotFoundError(f"Adapter weights not found: {p}")
    sd = torch.load(str(p), map_location=map_location)
    # Some checkpoints store {"state_dict": {...}}
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError(f"Unexpected checkpoint format at {p} (got {type(sd)})")
    return sd

@torch.no_grad()
def apply_adapter_blend(model: torch.nn.Module, adapter_path: str | Path, alpha: float) -> Tuple[int, torch.nn.Module]:
    """
    Loads adapter state, blends into model weights, and returns (matched_count, model).
    """
    adapter_sd = load_adapter_state(adapter_path, map_location="cpu")
    base_sd = model.state_dict()
    matched = blend_state_dicts_(base_sd, adapter_sd, alpha)
    # strict=False to avoid unexpected buffers
    model.load_state_dict(base_sd, strict=False)
    return matched, model
