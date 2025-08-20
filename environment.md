# Environment

## System
- OS           : Windows-10-10.0.26100-SP0 (x64)
- Python       : 3.11.9 (venv active)

## Core Python packages
- torch        : 2.1.0+cpu
- torchaudio   : 2.1.0+cpu
- transformers : 4.55.2
- numpy        : 1.26.4
- scipy        : 1.16.1
- soundfile    : 0.13.1
- bleak        : INSTALLED (with bleak-winrt)   ← **after fix**
- accelerate   : not installed

## PyTorch / CUDA
- CUDA available : False (CPU-only)
- torch.version.cuda : None
- cuDNN : None

## Audio / media tools
- libsoundfile : bundled by wheel
- ffmpeg : INSTALLED & on PATH   ← **after fix**

## Adapters
- adapters/neutral/merged_state_dict.pt : OK
- adapters/calm/merged_state_dict.pt    : OK

## Prompt banks
- prompts/energy_prompts.csv  : OK
- prompts/neutral_prompts.csv : OK
- prompts/calm_prompts.csv    : OK

## Project settings
- MusicGen seed (fixed) : **4241**
- Adapter alpha         : neutral **0.005**, calm **0.005**
- ISO fallback          : enabled when HR is stale; otherwise TARGETED mode

## Notes
- Windows BLE backend requires `bleak` + `bleak-winrt`.
- Use `python .\scan_ble.py` to confirm Polar address.
