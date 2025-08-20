# ui/live.py
import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.polar_bridge import PolarVeritySenseHR
from src.control.state_mapper import HRStateMapper, HRRules
from src.control.prompt_iso import build_prompt_session, ISO_PLAN
from src.audio.postfx import normalize_peak
from src.gen.engine import render_one_clip


MUSICGEN_SEED = 4241


STRESS_ESCALATE_DELTA = 20.0
STRESS_SUSTAIN_WINDOWS = 2


def _nowstamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _equal_power_xfade(a: np.ndarray, b: np.ndarray, overlap: int) -> np.ndarray:
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
    s = (stage or "").lower()
    if s == "neutral":
        return "neutral"
    if s == "calm":
        return "calm"
    return "base"  # energy uses base model


def _safe_slug(prompt: str, maxlen: int = 48) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_ ") else " " for ch in prompt.lower())
    parts = cleaned.split()
    slug = "-".join(parts)
    return slug[:maxlen] if slug else "prompt"


def build_parser():
    p = argparse.ArgumentParser(
        prog="BASS Live",
        description="Live HR-driven generation loop (Polar Verity Sense). Adaptive (TARGETED) with ISO fallback."
    )
    p.add_argument("--device", type=str, default="Polar Verity Sense",
                   help="BLE name or address of the Polar device.")
    p.add_argument("--baseline-bpm", type=float, required=True,
                   help="Baseline (resting) heart rate in BPM.")

    #layout
    p.add_argument("--segments", type=int, default=4,
                   help="How many segments to render (>=4: extra segments will hold the final stage).")
    p.add_argument("--segment-sec", type=int, default=20,
                   help="Segment duration.")
    p.add_argument("--crossfade-sec", type=float, default=1.5,
                   help="Crossfade between segments.")
    p.add_argument("--lookahead", type=int, default=2,
                   help="Planned look-ahead segments (sequential in CPU mode).")

    # Generation / adapters (stage overrides apply at runtime)
    p.add_argument("--alpha", type=float, default=0.005,
                   help="Default adapter blend alpha (stage overrides: neutral=0.005, calm=0.005).")
    p.add_argument("--temperature", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=150)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--cfg", type=float, default=1.6)

    # NOTE: seed controls prompt/session randomness only.
    p.add_argument("--seed", type=int, default=4242,
                   help="Prompt/session seed (affects prompt selection). Model seed is fixed at 4241.")
    p.add_argument("--highpass", type=int, default=120)
    p.add_argument("--lowpass", type=int, default=10000)
    p.add_argument("--peak", type=float, default=0.90)

    # Outputs
    p.add_argument("--outputs", type=str, default="outputs/audio")
    p.add_argument("--metrics-csv", type=str, default="outputs/metrics.csv")

    p.add_argument("--log-level", choices=["INFO", "DEBUG"], default="INFO")
    return p


# stage-specific generation tweaks
STAGE_PARAMS = {
    "energy":  {"lowpass": 12000, "cfg": 1.7, "temperature": 0.95, "alpha": 0.0},      # no adapter
    "neutral": {"lowpass": 10000, "cfg": 1.6, "temperature": 0.95, "alpha": 0.008},    # neutral adapter 
    "calm":    {"lowpass": 9500,  "cfg": 1.5, "temperature": 0.92, "alpha": 0.008},    # calm adapter 
}


def _iso_plan_from_phys(phys_init: str) -> list[str]:
    key = (phys_init or "").lower()
    if key not in ISO_PLAN:
        if key == "stressed":
            key = "stress"
        elif key in {"under_aroused", "underaroused", "low", "low_energy"}:
            key = "under"
        else:
            key = "neutral"
    return ISO_PLAN[key]


def main(argv=None):
    args = build_parser().parse_args(argv)

    outputs = Path(args.outputs); outputs.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(args.metrics_csv); metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    # HR mapper thresholds classify once after 30 s warm-up.
    rules = HRRules(stress_delta_bpm=15, under_delta_bpm=-10, window_sec=30, sustain_sec=30)
    mapper = HRStateMapper(rules)

    # strict 30 s warm-up
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

    #ISO plan (fallback)
    phys_init = mapper.map_bpm(args.baseline_bpm, hr.get_bpm(default=args.baseline_bpm) or args.baseline_bpm)
    iso_plan = _iso_plan_from_phys(phys_init)
    if args.segments > len(iso_plan):
        iso_plan = iso_plan + [iso_plan[-1]] * (args.segments - len(iso_plan))
    else:
        iso_plan = iso_plan[:args.segments]

    # Build a prompt session
    try:
        prompt_sess = build_prompt_session(session_seed=int(args.seed))
    except Exception as e:
        print(f"[ERROR] Prompt session build failed: {e}")
        sys.exit(2)

    plan_str = ",".join(s[0].upper() for s in iso_plan)  # e.g., N,N,C,C
    print(f"[INFO] Initial physiological state: {phys_init}")
    print(f"[INFO] Locked 4-stage ISO fallback: {plan_str}  (base='{prompt_sess.locked_base}')")
    print(f"[INFO] MusicGen model seed: {MUSICGEN_SEED} (fixed)")

    # Adaptive counters
    stress_windows = 0

    # Live generation loop (sequential CPU)
    stamp = _nowstamp()
    session_mix = None
    mix_sr = None
    crossfade_samples = None

    try:
        total_segments = int(args.segments)

        # Freshness thresholds for HR data
        freshness_threshold = max(2.0, float(args.segment_sec) * 0.75)
        min_span = max(10.0, rules.window_sec * 0.5)

        for i in range(total_segments):
            # HR snapshot
            avg_bpm = hr.get_bpm(default=args.baseline_bpm) or args.baseline_bpm
            phys_now = mapper.map_bpm(args.baseline_bpm, avg_bpm)
            age = hr.last_sample_age_sec()
            span = hr.window_span()

            # Decide mode
            is_fresh = (age is not None) and (age <= freshness_threshold) and (span >= min_span)
            if is_fresh:
                # TARGETED: complementary mapping + escalation for sustained stress
                mode = "TARGETED"
                if phys_now == "stress":
                    stress_windows += 1
                    if stress_windows >= STRESS_SUSTAIN_WINDOWS or (avg_bpm >= args.baseline_bpm + STRESS_ESCALATE_DELTA):
                        stage = "calm"
                    else:
                        stage = "neutral"
                elif phys_now == "under":
                    stress_windows = 0
                    stage = "energy"
                else:  # neutral
                    stress_windows = 0
                    stage = "neutral"
            else:
                # ISO FALLBACK
                mode = "ISO_FALLBACK"
                stage = iso_plan[i] if i < len(iso_plan) else iso_plan[-1]
                if phys_now != "stress":
                    stress_windows = 0

            # Prompt for the chosen stage
            piece = prompt_sess.next_prompt_for(stage)
            prompt = piece.text
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
            stage_alpha = float(sp["alpha"])  # 0.005 for neutral/calm, 0.0 for energy

            slug = _safe_slug(prompt, maxlen=48)
            out_wav = outputs / f"{stamp}_live_seg{i:02d}_{stage}_{slug}_a{stage_alpha:.4f}.wav"

            age_txt = f"{age:.1f}s" if age is not None else "None"
            print(f"\n=== LIVE SEGMENT {i+1}/{total_segments} ===")
            print(f"HR avg bpm  : {avg_bpm:.1f}  (baseline {args.baseline_bpm:.1f}) "
                  f"phys_now={phys_now} | mode={mode} | last_age={age_txt} | span={span:.1f}s")
            print(f"Stage       : {stage.upper()}  (stress_windows={stress_windows})")
            print(f"Adapter     : {adapter_name}  alpha={stage_alpha}")
            print(f"Sampler     : temp={stage_temp} top_k={args.top_k} top_p={args.top_p} cfg={stage_cfg}")
            print(f"Filters     : HPF={args.highpass}Hz LPF={stage_lowpass}Hz peak={args.peak}")
            print(f"Prompt      : {prompt}")
            print(f"Out         : {out_wav}")

            ok = render_one_clip(
                keyword=prompt,
                adapter_name=adapter_name,
                adapter_path=adapter_path,
                alpha=stage_alpha,
                duration_sec=int(args.segment_sec),
                temperature=stage_temp,
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                cfg_coef=stage_cfg,
                seed=MUSICGEN_SEED,                   # fixed model seed
                highpass_hz=int(args.highpass),
                lowpass_hz=stage_lowpass,
                limiter_drive=1.6,
                peak_target=float(args.peak),
                out_wav=str(out_wav),
                metrics_csv=str(metrics_csv),
                log_level=args.log_level,
            )
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
