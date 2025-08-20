# BASS: Bioadaptive Audio Synthesis System

BASS is a closed-loop system that listens to your body (via heart rate signals) and generates adaptive music using AI. It combines wearable biometric data with text-to-music generation to create emotionally congruent music for stress reduction, relaxation, or activation (wellbeing in general).

# System Overview
- **Biometric Input:** Polar Verity Sense (HR at 1s resolution) with a 3-minute baseline.
- **Emotion Detection:** Rule-based thresholds derived from WESAD (+15 bpm = stress, –10 bpm = under-arousal).
- **Prompt Bank:** Curated JSON/CSV of ~20 structured prompts (genre + mood + instruments + tempo).
- **Music Generation:** MusicGen-small with LoRA adapters (neutral, calm). Blending in weight space with α ≈ 0.01–0.015.
- **Post-FX:** High-pass 120 Hz, low-pass 10 kHz, soft tanh limiter (drive 1.6, peak target 0.90).
- **Controller:** 20–30 s segments, 2-clip lookahead buffer, 1.5 s crossfades, neutral fallback.
- **Evaluation:** Logs audio metrics (RMS, crest, centroid, etc.) and supports listening tests.

# Run/Quickstart Guide (powershell)

# Activate venv first
.\.venv\Scripts\Activate.ps1

# Live 4-stage session mix
.\scripts\run_live.ps1

# Quick single renders
.\scripts\run_base.ps1
.\scripts\run_neutral.ps1
.\scripts\run_calm.ps1

**Logs**
- Outputs saved under outputs/audio
- Metrics appended to outputs/metrics.csv


## Repository Layout
bass_project/
├─ adapters/ # calm and neutral LoRA adapters + README
├─ prompts/ # emotion_prompt_bank.json / .csv
├─ src/ # core code (engine, adapters, postfx, metrics, state mapper)
├─ ui/cli.py # main entrypoint for sessions
├─ tests/ # smoke tests (base load, adapter blend)
├─ configs/ # defaults.yaml, paths.yaml
├─ scripts/ # helper scripts for runs and sweeps
├─ outputs/ # generated audio, logs, metrics.csv
├─ requirements.txt # pinned dependencies
├─ environment.md # setup guide
└─ README.md



3. **Logs**
- Outputs saved under outputs/audio
- Metrics appended to outputs/metrics.csv

## Emotion Rules (from WESAD)
- Stress: HR > baseline + 15 bpm (≥ 30 s)
- Under-aroused: HR < baseline – 10 bpm (≥ 20 s)
- Neutral: within ±10 bpm of baseline

## Adapters Trained (from DEAM)
- calm/neutral adapters trained with 200 audios of each category


## Future Work

- **True streaming playback:** fork Audiocraft to expose token/PCM chunks during generation (incremental EnCodec decode + audio device streaming). Requires changes in `MusicGen.generate()` to yield frames and a small audio streamer.  
- **Real-time GPU pipeline:** enable Triton/xFormers on the GPU build; keep LM on CUDA with AMP/FP16; measure and target render_time < segment_sec with prebuffer ≥ 2.  
- **Async renderer + player:** background render queue with N-lookahead and a small playback ring buffer; resilient to late renders and device hiccups.  
- **Adaptive segment sizing:** dynamically shorten/lengthen segment duration based on measured render speed to maintain gapless playback without overloading the GPU.  
- **Advanced HR fusion:** incorporate HRV/EDA (when available) and Bayesian smoothing over 30–60 s windows for more robust state estimates.

## Credits
- META AI MusicGen
- WESAD dataset - HR calibration
- DEAM dataset - Adapter training
- Polar Verity Sense - biometric Input

python ui\live.py --device "24:AC:AC:06:E8:39" `
>>   --baseline-bpm 75 `                                
>>   --segments 4 --segment-sec 20 --crossfade-sec 2 `
>>   --log-level INFO