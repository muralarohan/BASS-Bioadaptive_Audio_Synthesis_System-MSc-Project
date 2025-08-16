Environment Setup (Windows, Python 3.11)
Prereqs

OS: Windows 10/11

Python: 3.11.x (64-bit)

GPU (optional): NVIDIA RTX (CUDA 12.x)

Tools: Git, PowerShell

1) Create & activate venv
cd .\bass_project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

2) Install dependencies (CPU-safe)
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

Optional: CUDA builds (RTX)

Install NVIDIA CUDA 12.x + matching cuDNN (or use the PyTorch wheels selector).

Then install CUDA wheels:

pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0


xformers from requirements.txt is kept to satisfy imports. On CPU it will warn; that’s fine.

3) Verify versions
python - << 'PY'
import sys, torch, torchaudio, torchvision
print("Python", sys.version.split()[0])
print("torch", torch.__version__, "CUDA:", torch.cuda.is_available())
print("torchaudio", torchaudio.__version__)
print("torchvision", torchvision.__version__)
PY

4) Project folders expected
adapters/{calm,neutral}/merged_state_dict.pt
prompts/emotion_prompt_bank.(json|csv)
src/...  ui/cli.py  outputs/  configs/

5) Gotchas (Windows)

- NumPy negative strides → torch: always .copy() NumPy arrays before torch.from_numpy(...).

- xformers warnings on CPU: ignore; keep installed so audiocraft imports cleanly.

- Long paths: enable Windows long paths if needed (gpedit or registry).

- FFmpeg (if you preprocess audio): install and add to PATH.

6) Quick smoke check (no audio files needed)
python -c "import torch, audiocraft; print('ok', torch.__version__)"

7) Typical issues

- Mismatch torch/torchvision/torchaudio → reinstall trio with the same build (CPU or same CUDA).

- VC++ runtime missing → install “Microsoft Visual C++ Redistributable” (x64).

- GPU not detected → verify nvidia-smi, driver ≥ CUDA toolkit version, then torch.cuda.is_available().

8) Deactivation
deactivate