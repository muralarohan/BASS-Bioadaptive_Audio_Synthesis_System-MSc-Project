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
        return np.concatenate([a, b]).astype(np.float32)
    overlap = min(overlap, len(a), len(b))
    tail = a[-overlap:].astype(np.float32)
    head = b[:overlap].astype(np.float32)
    t = np.linspace(0.0, np.pi / 2, overlap, dtype=np.float32)
    w_a = (np.cos(t) ** 2).astype(np.float32)
    w_b = (np.sin(t) ** 2).astype(np.float32)
    cross = w_a * tail + w_b * head
    return np.concatenate([a[:-overlap], cross, b[overlap:]]).astype(np.float32)


def _adapter_for_stage(stage: str) -> str:
    """
    Map musical stage to adapter name expected by render_one_clip:
      - 'energy'  -> base (no adapter)
      - 'neutral' -> neutral adapter
      - 'calm'    -> calm adapter
    """
    s = (stage or "").lower()
    if s == "neutral":
        return "neutral"
    if s == "calm":
        return "calm"
    return "base"  # energy uses base model


def build_parser():
    p = argparse.ArgumentParser(
        prog="BASS Live",
        description="Live HR-driven generation loop (Polar Verity Sense). Locked 4-stage schedule (energy/neutral/calm)."
    )
    p.add_argument("--device", type=str, default="Polar Verity Sense",
                   help="BLE name or address of the Polar device.")
    p.add_argument("--baseline-bpm", type=float, required=True,
                   help="Baseline (resting) heart rate in BPM.")

    # Session layout
    p.add_argument("--segments", type=int, default=4,
                   help="How many segments to render (>=4: extra segments will hold the final stage).")
    p.add_argument("--segment-sec", type=int, default=20,
                   help="Segment duration.")
    p.add_argument("--crossfade-sec", type=float, default=1.5,
                   help="Crossfade between segments.")
    p.add_argument("--lookahead", type=int, default=2,
                   help="Planned look-ahead segments (sequential in CPU mode).")

    # Generation / adapters (stage overrides apply at runtime)
    p.add_argument("--alpha", type=float, default=0.01,
                   help="Adapter blend alpha (fixed to 0.01 for neutral/calm; energy has no adapter).")
    p.add_argument("--temperature", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=150)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--cfg", type=float, default=1.6)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--highpass", type=int, default=120)
    p.add_argument("--lowpass", type=int, default=10000)
    p.add_argument("--peak", type=float, default=0.90)

    # Prompt bank
    p.add_argument("--prompt-json", type=str, default="prompts/emotion_prompt_bank.json")
    p.add_argument("--prompt-csv", type=str, default="prompts/emotion_prompt_bank.csv")

    # Outputs
    p.add_argument("--outputs", type=str, default="outputs/audio")
    p.add_argument("--metrics-csv", type=str, default="outputs/metrics.csv")

    p.add_argument("--log-level", choices=["INFO", "DEBUG"], default="INFO")
    return p


def _build_locked_schedule(phys_init: str):
    """
    Build the musical stage plan from the initial physiological state.
    Returns (schedule_id, stages_list) where stages are 'energy'|'neutral'|'calm'.
    """
    pi = (phys_init or "").lower()
    if pi == "stress":
        stages = ["neutral", "neutral", "calm", "calm"]
        sid = "stress->N,N,C,C"
    elif pi == "under":
        stages = ["energy", "neutral", "neutral", "calm"]
        sid = "under->E,N,N,C"
    else:  # treat anything else as neutral
        stages = ["neutral", "neutral", "calm", "calm"]
        sid = "neutral->N,N,C,C"
    return sid, stages


# ---- stage-specific generation tweaks ----
# These override the baseline args per musical stage.
STAGE_PARAMS = {
    "energy":  {"lowpass": 12000, "cfg": 1.7, "temperature": 0.95, "alpha": 0.0},   # no adapter
    "neutral": {"lowpass": 10000, "cfg": 1.6, "temperature": 0.95, "alpha": 0.01},  # neutral adapter @ 0.01
    "calm":    {"lowpass": 9500,  "cfg": 1.5, "temperature": 0.92, "alpha": 0.01},  # calm adapter @ 0.01
}


def main(argv=None):
    args = build_parser().parse_args(argv)

    outputs = Path(args.outputs); outputs.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(args.metrics_csv); metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    prompt_json = Path(args.prompt_json); prompt_csv = Path(args.prompt_csv)

    # Load prompt bank (must include 'state' entries for energy/neutral/calm)
    entries = load_prompt_bank(prompt_json, prompt_csv)
    if not entries:
        print("[ERROR] Prompt bank appears empty."); sys.exit(2)

    # HR mapper thresholds (WESAD-derived). We classify once after 30 s warm-up.
    rules = HRRules(stress_delta_bpm=15, under_delta_bpm=-10, window_sec=30, sustain_sec=30)
    mapper = HRStateMapper(rules)

    # Connect & strict 30 s warm-up
    hr = PolarVeritySenseHR(device=args.device)  # 30 s window inside
    print(f"[INFO] Connecting to {args.device} ...")
    try:
        hr.connect()
        print("[INFO] Connected. Waiting for HR notifications ...")
        cnt, span_sec, warm_bpm, waited = hr.wait_for_window(
            seconds=30.0, timeout=45.0, default=args.baseline_bpm
        )
        print(f"[INFO] HR warm-up: reached window=30.0s in {waited:.1f}s "
              f"(samples={cnt}, span={span_sec:.1f}s) → mean={warm_bpm:.1f} bpm")
    except Exception as e:
        print(f"[ERROR] Could not connect to Polar device: {e}")
        sys.exit(2)

    # Classify once and lock schedule
    phys_init = mapper.map_bpm(args.baseline_bpm, hr.get_bpm(default=args.baseline_bpm) or args.baseline_bpm)
    schedule_id, plan = _build_locked_schedule(phys_init)
    print(f"[INFO] Initial physiological state: {phys_init}")
    print(f"[INFO] Locked 4-stage schedule: {schedule_id}")

    # Live generation loop (sequential CPU, follow locked plan)
    stamp = _nowstamp()
    session_mix = None
    mix_sr = None
    seed = int(args.seed)
    crossfade_samples = None

    try:
        total_segments = int(args.segments)
        for i in range(total_segments):
            # Rolling 30 s avg shown for logging only
            bpm = hr.get_bpm(default=args.baseline_bpm) or args.baseline_bpm
            phys_now = mapper.map_bpm(args.baseline_bpm, bpm)

            # Choose musical stage from locked plan (hold final stage if more than 4 segments)
            stage = plan[i] if i < len(plan) else plan[-1]

            # Stage-specific prompt and adapter
            prompt = get_prompt_for_state(entries, stage)["prompt"]
            adapter_name = _adapter_for_stage(stage)

            # Adapter path resolution
            adapter_path = None
            if adapter_name == "neutral":
                adapter_path = "adapters/neutral/merged_state_dict.pt"
            elif adapter_name == "calm":
                adapter_path = "adapters/calm/merged_state_dict.pt"

            # Stage-specific overrides
            sp = STAGE_PARAMS[stage]
            stage_lowpass = int(sp["lowpass"])
            stage_cfg = float(sp["cfg"])
            stage_temp = float(sp["temperature"])
            stage_alpha = float(sp["alpha"])  # 0.01 for adapter stages, 0.0 for energy

            slug = prompt.lower().replace(" ", "-")[:48]
            # Use stage tag (energy/neutral/calm) in filename for clarity
            out_wav = outputs / f"{stamp}_live_seg{i:02d}_{stage}_{slug}_a{stage_alpha:.4f}.wav"

            print(f"\n=== LIVE SEGMENT {i+1}/{total_segments} ===")
            print(f"HR avg bpm  : {bpm:.1f}  (baseline {args.baseline_bpm:.1f}) "
                  f"phys_now={phys_now} | phys_init={phys_init} | stage={stage.upper()}")
            print(f"Adapter     : {adapter_name}  alpha={stage_alpha}")
            print(f"Sampler     : temp={stage_temp} top_k={args.top_k} top_p={args.top_p} cfg={stage_cfg}")
            print(f"Filters     : HPF={args.highpass}Hz LPF={stage_lowpass}Hz peak={args.peak}")
            print(f"Prompt      : {prompt}")
            print(f"Out         : {out_wav}")

            ok = render_one_clip(
                keyword=prompt,
                adapter_name=adapter_name,           # base for energy; neutral/calm otherwise
                adapter_path=adapter_path,
                alpha=stage_alpha,
                duration_sec=int(args.segment_sec),
                temperature=stage_temp,
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                cfg_coef=stage_cfg,
                seed=int(seed),
                highpass_hz=int(args.highpass),
                lowpass_hz=stage_lowpass,
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
                audio = audio[:, 0]
            if session_mix is None:
                session_mix = audio.astype(np.float32)
                mix_sr = sr
                crossfade_samples = int(float(args.crossfade_sec) * sr)
            else:
                if sr != mix_sr:
                    print("[WARN] Sample rate changed mid-session; resampling not implemented.")
                session_mix = _equal_power_xfade(session_mix, audio.astype(np.float32), overlap=crossfade_samples)

        # Finalize session mix
        if session_mix is not None and mix_sr is not None:
            session_mix = normalize_peak(session_mix, target=0.90)
            live_out = outputs / f"{stamp}_live_session_mix.wav"
            sf.write(str(live_out), session_mix, mix_sr, subtype="PCM_16")
            print(f"\n[OK] Live session mix written to: {live_out}")
        else:
            print("[WARN] No segments were rendered; nothing to write.")

    finally:
        # Cleanly stop notifications, disconnect, and shut down the background loop/thread
        try:
            hr.disconnect()
        except Exception:
            pass
        try:
            hr.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
