# ui/live.py
import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import soundfile as sf

# Ensure project root on sys.path to import src.*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.polar_bridge import PolarVeritySenseHR
from src.control.state_mapper import HRStateMapper, HRRules
from src.control.prompt_selector import load_prompt_bank, get_prompt_for_state
from src.audio.postfx import normalize_peak
from src.gen.engine import render_one_clip

def _nowstamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _equal_power_xfade(a: np.ndarray, b: np.ndarray, overlap: int) -> np.ndarray:
    """
    Equal-power crossfade: last 'overlap' samples of a with first 'overlap' of b.
    Assumes both are 1-D float32 arrays at the same sample rate.
    """
    if overlap <= 0 or len(a) == 0:
        return np.concatenate([a, b])
    overlap = min(overlap, len(a), len(b))
    tail = a[-overlap:]
    head = b[:overlap]
    t = np.linspace(0.0, np.pi/2, overlap, dtype=np.float32)
    w_a = (np.cos(t) ** 2).astype(np.float32)
    w_b = (np.sin(t) ** 2).astype(np.float32)
    cross = w_a * tail + w_b * head
    return np.concatenate([a[:-overlap], cross, b[overlap:]]).astype(np.float32)

def _choose_adapter_for_state(state: str) -> str:
    """
    Simple adapter policy:
      - 'calm'  -> 'calm' adapter
      - 'neutral' or 'stress' -> 'neutral' adapter (more stable, fewer artifacts)
      - otherwise -> 'base'
    """
    s = (state or "").lower()
    if s == "calm":
        return "calm"
    if s in {"neutral", "stress"}:
        return "neutral"
    return "base"

def build_parser():
    p = argparse.ArgumentParser(
        prog="BASS Live",
        description="Live HR-driven generation loop (Polar Verity Sense)."
    )
    p.add_argument("--device", type=str, default="Polar Verity Sense", help="BLE name or address of the Polar device.")
    p.add_argument("--baseline-bpm", type=float, required=True, help="Baseline (resting) heart rate in BPM.")
    p.add_argument("--segment-sec", type=int, default=20, help="Segment duration.")
    p.add_argument("--crossfade-sec", type=float, default=1.5, help="Crossfade between segments.")
    p.add_argument("--lookahead", type=int, default=2, help="Planned look-ahead segments (sequential in CPU mode).")
    p.add_argument("--segments", type=int, default=3, help="How many segments to render (for testing).")
    p.add_argument("--alpha", type=float, default=0.0125, help="Adapter blend alpha.")
    p.add_argument("--temperature", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=150)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--cfg", type=float, default=1.6)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--highpass", type=int, default=120)
    p.add_argument("--lowpass", type=int, default=10000)
    p.add_argument("--peak", type=float, default=0.90)
    p.add_argument("--prompt-json", type=str, default="prompts/emotion_prompt_bank.json")
    p.add_argument("--prompt-csv", type=str, default="prompts/emotion_prompt_bank.csv")
    p.add_argument("--outputs", type=str, default="outputs/audio")
    p.add_argument("--metrics-csv", type=str, default="outputs/metrics.csv")
    p.add_argument("--target-state", type=str, default="calm", help="Where we want to guide the user (iso-principle).")
    p.add_argument("--log-level", choices=["INFO", "DEBUG"], default="INFO")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)

    outputs = Path(args.outputs); outputs.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(args.metrics_csv); metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    prompt_json = Path(args.prompt_json); prompt_csv = Path(args.prompt_csv)

    # Load prompt bank
    entries = load_prompt_bank(prompt_json, prompt_csv)
    if not entries:
        print("[ERROR] Prompt bank appears empty."); sys.exit(2)

    # HR mapper thresholds from earlier defaults
    rules = HRRules(stress_delta_bpm=15, under_delta_bpm=-10, window_sec=10, sustain_sec=30)
    mapper = HRStateMapper(rules)

    # Connect to Polar
    hr = PolarVeritySenseHR(device=args.device)
    print(f"[INFO] Connecting to {args.device} ...")
    try:
        hr.connect()
        print("[INFO] Connected. Waiting for HR notifications ...")
    except Exception as e:
        print(f"[ERROR] Could not connect to Polar device: {e}")
        sys.exit(2)

    # Live generation loop (sequential CPU)
    stamp = _nowstamp()
    session_mix = None
    mix_sr = None
    seed = int(args.seed)
    crossfade_samples = None

    try:
        for i in range(args.segments):
            bpm = hr.get_bpm(default=args.baseline_bpm) or args.baseline_bpm
            state = mapper.map_bpm(args.baseline_bpm, bpm)
            # iso-principle step: move one step toward target if far
            target = (args.target_state or "calm").lower()
            order = ["under", "neutral", "stress"]
            def step_toward(cur: str, tgt: str) -> str:
                if cur not in order or tgt not in order:  # unknown -> no step
                    return cur
                ci, ti = order.index(cur), order.index(tgt)
                if ci == ti:
                    return cur
                return order[ci + 1] if ti > ci else order[ci - 1]

            next_state = step_toward(state, target)
            prompt = get_prompt_for_state(entries, next_state)["prompt"]
            adapter_name = _choose_adapter_for_state(next_state)

            # Adapter path resolution (matches ui/cli.py mapping)
            adapter_path = None
            if adapter_name == "neutral":
                adapter_path = "adapters/neutral/merged_state_dict.pt"
            elif adapter_name == "calm":
                adapter_path = "adapters/calm/merged_state_dict.pt"

            # Output filename per segment
            slug = prompt.lower().replace(" ", "-")[:48]
            out_wav = outputs / f"{stamp}_live_seg{i:02d}_{adapter_name}_{slug}_a{args.alpha:.4f}.wav"

            print(f"\n=== LIVE SEGMENT {i+1}/{args.segments} ===")
            print(f"HR avg bpm  : {bpm:.1f}  (baseline {args.baseline_bpm:.1f}) -> state={state}, next={next_state}")
            print(f"Adapter     : {adapter_name}  alpha={args.alpha}")
            print(f"Prompt      : {prompt}")
            print(f"Out         : {out_wav}")

            # Render one clip (engine handles base reset + blending + metrics)
            ok = render_one_clip(
                keyword=prompt,
                adapter_name=adapter_name if adapter_name in {"neutral","calm"} else "base",
                adapter_path=adapter_path,
                alpha=float(args.alpha if adapter_name in {"neutral","calm"} else 0.0),
                duration_sec=int(args.segment_sec),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                cfg_coef=float(args.cfg),
                seed=int(seed),
                highpass_hz=int(args.highpass),
                lowpass_hz=int(args.lowpass),
                limiter_drive=1.6,
                peak_target=float(args.peak),
                out_wav=str(out_wav),
                metrics_csv=str(metrics_csv),
                log_level=args.log_level,
            )
            seed += 1
            if not ok:
                print("[WARN] Segment render returned falsy result; skipping mix step.")
                continue

            # Load back the WAV and build session mix with crossfade
            audio, sr = sf.read(str(out_wav), dtype="float32")
            if audio.ndim > 1:
                audio = audio[:,0]
            if session_mix is None:
                session_mix = audio.astype(np.float32)
                mix_sr = sr
                crossfade_samples = int(float(args.crossfade_sec) * sr)
            else:
                if sr != mix_sr:
                    print("[WARN] Sample rate changed mid-session; resampling not implemented.")
                session_mix = _equal_power_xfade(session_mix, audio.astype(np.float32), overlap=crossfade_samples)

        # Finalize session mix
        if session_mix is not None:
            # Gentle peak normalize to 0.90 as in offline runs
            from src.audio.postfx import normalize_peak
            session_mix = normalize_peak(session_mix, target=0.90)
            live_out = outputs / f"{stamp}_live_session_mix.wav"
            sf.write(str(live_out), session_mix, mix_sr, subtype="PCM_16")
            print(f"\n[OK] Live session mix written to: {live_out}")
        else:
            print("[WARN] No segments were rendered; nothing to write.")
    finally:
        try:
            hr.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    main()
