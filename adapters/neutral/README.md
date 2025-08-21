# BASS Adapter — neutral

- **Base model:** musicgen-small (Audiocraft 1.3.0)
- **Format:** merged_state_dict (LoRA merged to base layout)
- **Intended effect:** stable/clean baseline timbre; reduced harshness/hiss vs. base
- **Blend method:** weight-space blend, not strict overwrite  
  \( W = (1–α)·W_base + α·W_adapter \)
- **α sweet spot:** 0.010–0.015 (start 0.0125)
- **Compatibility:** must match musicgen-small config; use blend loader (not `strict=True`)
- **Checksum (SHA256):**

## Loader sanity check
- Expect: `keys matched and blended` (matched > 0).  
- If `matched=0`, payload likely from a different base/config → do not force-load.

## Quick usage (examples)
- Gentle test (6s):  
  `python ui/cli.py --only neutral --adapter-strength 0.0125 --keyword "neutral piano arpeggio, slow tempo, no drums, soft reverb" --duration 6`
- With post-FX defaults: HPF 120 Hz, LPF 10 kHz, limiter 1.6, peak 0.90.

## Do / Don’t
- Reset to pristine base before blending.
- Keep seed fixed for A/B comparisons.
- Don’t chain adapters in memory.
- Don’t ignore loader warnings (missing/unexpected keys).

## Changelog
- `YYYY-MM-DD` — added neutral `merged_state_dict.pt` (origin/notes: `<fill>`).
