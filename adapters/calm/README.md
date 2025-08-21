# BASS Adapter — calm

- **Base model:** musicgen-small (Audiocraft 1.3.0)
- **Format:** merged_state_dict (LoRA merged to base layout)
- **Intended effect:** smoother timbre, lower brightness, gentler dynamics
- **Blend method:** weight-space blend, not strict overwrite  
  \( W = (1–α)·W_base + α·W_adapter \)
- **α sweet spot:** 0.010–0.015 (start 0.0125)
- **Compatibility:** must match musicgen-small config; use blend loader (not `strict=True`)
- **Checksum (SHA256):** 

## Loader sanity check
- Loader should log: `keys matched and blended` (matched > 0).  
- If `matched=0`, this payload is incompatible → do **not** force-load.

## Quick usage (examples)
- Gentle test (6s):  
  `python ui/cli.py --only calm --adapter-strength 0.0125 --keyword "calm ambient pads, slow tempo, no drums, soft reverb" --duration 6`
- With post-FX defaults: HPF 120 Hz, LPF 10 kHz, limiter 1.6, peak 0.90.

## Do / Don’t
- Restore pristine base before every render (no stacking).
- Try α in {0.01, 0.0125, 0.015}; listen before going higher.
- Don’t push α > 0.03 without A/B (artifacts rise fast).
- Don’t rely on adapter to rescue vague prompts—use musically specific language.

## Changelog
- `YYYY-MM-DD` — added calm `merged_state_dict.pt` (origin/notes: `<fill>`).
