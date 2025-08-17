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


## Quickstart

1. **Environment**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

2. **Run a neutral Adapter Test**
python ui/cli.py --only neutral `
  --adapter-strength 0.0125 `
  --keyword "neutral piano arpeggio, slow tempo, no drums, soft reverb" `
  --duration 6 --highpass 120 --lowpass 10000 --peak 0.90 --seed 4242

3. **Logs**
- Outputs saved under outputs/audio
- Metrics appended to outputs/metrics.csv

## Emotion Rules (from WESAD)
- Stress: HR > baseline + 15 bpm (≥ 30 s)
- Under-aroused: HR < baseline – 10 bpm (≥ 20 s)
- Neutral: within ±10 bpm of baseline

## Adapters Trained (from DEAM)
- calm/neutral adapters trained with 200 audios of each category
- 

## Credits
- META AI MusicGen
- WESAD dataset - HR calibration
- DEAM dataset - Adapter training
- Polar Verity Sense - biometric Input