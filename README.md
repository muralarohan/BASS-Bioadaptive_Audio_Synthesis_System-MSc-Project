# BASS — Bioadaptive Audio Synthesis System (Proof of Concept)

BASS is a closed-loop music generation system that senses physiology (heart rate) and renders adaptive, emotionally congruent music with a text-to-music model. The aim is to support stress reduction, relaxation, or activation through targeted musical states and smooth, gapless transitions.

Note: The project was created in multiple environments, the codes might have minor inconsistencies but the product works)

---

## 1. Overview

### Signal flow
```
Polar Verity Sense HR → HR state mapping (stress / neutral / under)
→ stage plan (energy / neutral / calm)
→ prompt generator (locked base + varying texture/fx)
→ MusicGen-small (+ adapters)
→ post-FX
→ crossfaded session mix
```

### Key properties
- **Physiology-aware control.** A rolling **30 s** HR window with WESAD-derived thresholds drives musical state.
- **Adaptive with ISO fallback.** If live HR is fresh, choose the stage adaptively; if HR lapses, follow a fixed **4-stage ISO** plan derived from the warm-up classification.
- **Consistent timbre, varied content.** MusicGen model seed is fixed at **4241** for timbral stability; prompts vary per session (one locked “base”; per-stage texture/fx sampled without replacement).
- **Smooth playback.** **20 s** segments (default) with **1.5 s** equal-power crossfades and a **peak-normalized** session mix.

---

## 2. Method

### 2.1 Heart-rate mapping
- **Window:** 30 s rolling average; sustained requirement ≈ 30 s.
- **Thresholds (WESAD-inspired):**
  - **Stress:** HR ≥ baseline **+15 bpm** (sustained).
  - **Under-aroused:** HR ≤ baseline **−10 bpm** (sustained).
  - **Neutral:** otherwise.

### 2.2 Stage selection (adaptive + ISO fallback)

**ISO plan from first 30 s warm-up:**
- Stress → `[energy, neutral, calm, calm]`
- Neutral → `[neutral, neutral, calm, calm]`
- Under → `[calm, neutral, calm, calm]`

**During playback (per segment):**
- If HR is **fresh** (newest sample age ≤ segment duration **and** window span ≥ 30 s):
  - If **stress:** choose **neutral**; escalate to **calm** if stress persists for ≥ 2 windows **or** if HR ≥ baseline **+20**.
  - If **neutral:** choose **neutral**.
  - If **under:** choose **energy**.
- If HR is **stale/missing:** fallback to the ISO plan’s stage for that segment.

> This guarantees continuity when HR data drops while still reacting rapidly when data is available.

### 2.3 Prompt generation
- **Banks:** `prompts/energy_prompts.csv`, `prompts/neutral_prompts.csv`, `prompts/calm_prompts.csv` (columns: `base, texture, fx`).
- **Locked base per session:** one base phrase is selected once and reused (stylistic cohesion).
- **Per-stage variation, no repeats:** for each requested stage, texture/fx are drawn **without replacement** (deterministic per-stage shuffle) to avoid within-session repetition.
- **Tempo wording (fixed):**
  - energy → “fast tempo”
  - neutral → “moderate tempo”
  - calm → “slow tempo”
- **Prompt form:** `"{base}, {tempo}, {texture}, {fx}"`
- **Determinism & variety:** Model seed is fixed (**4241**). Prompt selection uses an internal seed mixed with a time-based nonce so each run varies even with the same model seed. (If exact replay is needed, a CLI nonce can be exposed later.)

### 2.4 Generation & post-FX
- **Model:** MusicGen-small (+ **LoRA** adapters).
- **Adapters / blend α (weight-space):**
  - energy: α = **0.0** (base only)
  - neutral: α = **0.005**
  - calm: α = **0.005**
- **Sampler defaults:** temperature **0.95**, top-k **150**, top-p **0.95**, CFG **1.6**.
- **Filters:** HPF **120 Hz**; LPF by stage → energy **12 kHz**, neutral **10 kHz**, calm **9.5 kHz**.
- **Limiter:** soft saturation (drive **1.6**) and peak target **0.90**.
- **Mixing:** equal-power crossfades (**1.5 s**) and final peak normalization.

---

## 3. Installation (Windows, CPU)

```powershell
# 1) Create & activate a venv
cd "your directory"
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 2) Upgrade tooling
python -m pip install --upgrade pip setuptools wheel

# 3) Install requirements (CPU wheels for torch/torchaudio)
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

**`requirements.txt` (expected)**
```text
# Core
torch==2.1.0+cpu
torchaudio==2.1.0+cpu
transformers==4.55.2
numpy==1.26.4
scipy==1.16.1
soundfile==0.13.1
tqdm>=4.66.4
pyyaml>=6.0.1

# BLE (Windows backend for Polar)
bleak[dotnet]>=0.22.2
```

> Optional: Install **FFmpeg** and add it to `PATH` if you plan to transcode formats.

---

## 4. Running a live session

```powershell
. .\.venv\Scripts\Activate.ps1
python ui\live.py --device "Polar Verity Sense" --baseline-bpm 75 `
  --segments 4 --segment-sec 20 --crossfade-sec 1.5 --log-level INFO
```

- You can pass the device **MAC/BLE address** instead of the name.
- Audio is written to `outputs/audio/`; metrics append to `outputs/metrics.csv`.

---

## 5. Repository layout
```
bass_project/
├─ adapters/                 # LoRA adapters (neutral, calm) + READMEs
├─ configs/                  # defaults.yaml, paths.yaml
├─ prompts/                  # energy_prompts.csv, neutral_prompts.csv, calm_prompts.csv
├─ scripts/                  # optional helper scripts
├─ src/
│  ├─ adapters/              # weight-space blending
│  ├─ audio/                 # postfx, metrics
│  ├─ control/               # prompt_iso.py, state_mapper.py
│  ├─ gen/                   # engine: MusicGen wrapper
│  └─ io/                    # Polar bridge (bleak, rolling HR window)
├─ tests/                    # smoke tests (base load, adapter blend)
├─ ui/
│  ├─ cli.py                 # optional CLI
│  └─ live.py                # main live session runner
├─ outputs/                  # audio, logs, metrics.csv   (git-ignored)
├─ requirements.txt
├─ environment.md            # environment details & setup report
└─ README.md
```

`.gitignore` excludes Python caches, virtual envs, model weights, audio artifacts, logs, and outputs/.

---

## 6. Evaluation & logging
- **Per-segment metrics** (e.g., RMS, crest factor, spectral centroid) → `outputs/metrics.csv`.
- **Session mix:** crossfaded and peak-normalized → `*_session_mix.wav`.
- **Console logs:** HR freshness, mode (`TARGETED` vs `ISO_FALLBACK`), chosen stage, prompt text, adapter blend.

---

## 7. Known limitations
- **CPU inference:** MusicGen-small on CPU renders offline; short segments and prebuffering maintain continuity. A GPU build greatly improves turnaround.
- **BLE stability:** In noisy RF environments, HR samples may drop; ISO fallback guarantees uninterrupted playback.
- **Prompt reproducibility:** Model timbre is fixed (seed **4241**: editable if needed). Prompt sequences vary per run due to a time-based nonce; an optional CLI nonce can be added for exact replay.

---

## 8. Datasets & credits
- **Model:** MusicGen (Meta AI).
- **HR thresholds:** Inspired by WESAD (Schmidt et al., 2018).
- **Adapter training:** DEAM-style emotion categories (Aljanaki et al., 2017).
- **Hardware:** Polar Verity Sense for biometric input.

> Please ensure you have the appropriate licenses and usage rights for models and datasets.

---

## 9. Citation
```bibtex
@misc{BASS_2025,
  title  = {BASS: Bioadaptive Audio Synthesis System},
  author = {Rohan Naveen Murala},
  year   = {2025},
  note   = {https://github.com/your-org/bass_project}
}
```

---

## 10. Troubleshooting

- **No HR data:** Check battery; ensure the Polar sensor is awake. Try the MAC/BLE address. Verify Bluetooth LE is enabled.
- **BLE errors (Windows):** Confirm `bleak[dotnet]` is installed (`pip show bleak`). Pair the device in Windows Settings if needed.
- **PyTorch install issues:**
  ```powershell
  pip install --extra-index-url https://download.pytorch.org/whl/cpu `
    torch==2.1.0+cpu torchaudio==2.1.0+cpu
  ```
- **Bright / HF artifacts:** Confirm HPF/LPF are applied; adjust stage LPFs (energy **12 k**, neutral **10 k**, calm **9.5 k**) or raise HPF slightly for your monitoring chain.

**Implementation reflects:** fixed MusicGen seed (**4241**); adaptive targeted control with ISO fallback; fixed tempo wording; per-stage no-repeat prompt sampling; adapter blends α(neutral)=**0.005** and α(calm)=**0.005**.


© 2025 Rohan Naveen Murala — All rights reserved. No redistribution or reuse.
