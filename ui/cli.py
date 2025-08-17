# ui/cli.py
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# --- ensure project root on sys.path (so `import src.*` works when running from /ui) ---
ROOT = Path(__file__).resolve().parents[1]  # .../bass_project
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------------------------------------------

# --------- small helpers ---------
def _safe_load_yaml(path):
    """Load YAML if available; else return {} and print a hint."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml  # requires PyYAML
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARN] Could not parse YAML at {p} ({e}). Falling back to defaults.")
        return {}

def _slugify(text, maxlen=48):
    keep = []
    for ch in text.lower():
        if ch.isalnum() or ch in ("-", "_", " "):
            keep.append(ch if ch != " " else "-")
    slug = "".join(keep).strip("-_")
    return slug[:maxlen] if slug else "prompt"

def _nowstamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

# --------- defaults (mirror configs/defaults.yaml) ---------
DEFAULTS = {
    "sampler": {
        "duration_sec": 20,
        "temperature": 0.95,
        "top_k": 150,
        "top_p": 0.95,
        "cfg_coef": 1.6,
        "use_sampling": True,
        "seed": 4242,
    },
    "postfx": {
        "highpass_hz": 120,
        "lowpass_hz": 10000,
        "limiter_drive": 1.6,
        "peak_target": 0.90,
    },
    "adapters": {
        "enabled": True,
        "name": "neutral",
        "alpha": 0.0125,
    },
    "realtime": {
        "segment_sec": 20,
        "crossfade_sec": 1.5,
        "lookahead_segments": 2,
    },
    "emotion_rules": {
        "stress_delta_bpm": 15,
        "under_delta_bpm": -10,
        "window_sec": 10,
        "sustain_sec": 30,
    },
    "logging": {
        "metrics_csv": "outputs/metrics.csv",
        "audio_dir": "outputs/audio",
        "logs_dir": "outputs/logs",
    },
}

DEFAULT_PATHS = {
    "paths": {
        "adapters": {
            "calm": "adapters/calm/merged_state_dict.pt",
            "neutral": "adapters/neutral/merged_state_dict.pt",
        },
        "prompts": {
            "json": "prompts/emotion_prompt_bank.json",
            "csv": "prompts/emotion_prompt_bank.csv",
        },
        "outputs": {
            "audio": "outputs/audio",
            "logs": "outputs/logs",
            "metrics": "outputs/metrics.csv",
        },
    }
}

# --------- CLI ---------
def build_parser():
    p = argparse.ArgumentParser(
        prog="BASS CLI",
        description="Run BASS music generation with optional LoRA adapter blending."
    )
    # Required (can be overridden by --prompt-id/--state)
    p.add_argument("--keyword", type=str, required=False, help="MusicGen text prompt or keyword.")

    # Use prompt bank
    p.add_argument("--prompt-id", type=int, help="Pick a prompt from the bank by ID.")
    p.add_argument("--state", type=str, help="Pick a prompt by state (e.g., calm, neutral, stress, under).")
    p.add_argument("--list-prompts", action="store_true", help="List prompt bank entries and exit.")

    # Adapter selection
    p.add_argument("--only", choices=["base", "neutral", "calm"], help="Select which path to run.")
    p.add_argument("--adapter-strength", type=float, help="Adapter blend alpha (e.g., 0.0125).")
    p.add_argument("--disable-adapter", action="store_true", help="Force base (no adapter).")

    # Sampler
    p.add_argument("--duration", type=int, help="Clip length in seconds.")
    p.add_argument("--temperature", type=float, help="Sampling temperature.")
    p.add_argument("--top-k", type=int, help="Top-K.")
    p.add_argument("--top-p", type=float, help="Top-P.")
    p.add_argument("--cfg", type=float, help="Classifier-free guidance coef.")
    p.add_argument("--seed", type=int, help="Random seed.")

    # Post-FX
    p.add_argument("--highpass", type=int, help="High-pass Hz.")
    p.add_argument("--lowpass", type=int, help="Low-pass Hz (0=off).")
    p.add_argument("--limiter-drive", type=float, help="Soft limiter drive.")
    p.add_argument("--peak", type=float, help="Peak target (0..1).")

    # Realtime/planning (kept for future continuous mode)
    p.add_argument("--segment-sec", type=int, help="Segment length for realtime.")
    p.add_argument("--crossfade-sec", type=float, help="Crossfade length.")
    p.add_argument("--lookahead", type=int, help="Lookahead segments.")

    # Paths & configs
    p.add_argument("--config", default="configs/defaults.yaml", help="Path to defaults.yaml.")
    p.add_argument("--paths", default="configs/paths.yaml", help="Path to paths.yaml.")
    p.add_argument("--audio-dir", help="Override output audio dir.")
    p.add_argument("--metrics-csv", help="Override metrics CSV path.")
    p.add_argument("--logs-dir", help="Override logs dir.")
    p.add_argument("--log-level", choices=["INFO", "DEBUG"], default="INFO")

    return p

def _resolve_config(args):
    # Load YAMLs (if present) and merge with DEFAULTS
    cfg = DEFAULTS.copy()
    user_cfg = _safe_load_yaml(args.config)
    for k, v in (user_cfg or {}).items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    paths = DEFAULT_PATHS.copy()
    user_paths = _safe_load_yaml(args.paths)
    # shallow merge (sufficient for our simple layout)
    if isinstance(user_paths, dict):
        for k, v in user_paths.items():
            if isinstance(v, dict) and k in paths:
                paths[k].update(v)
            else:
                paths[k] = v

    # CLI overrides
    # adapter
    if args.only:
        cfg["adapters"]["name"] = args.only
    if args.adapter_strength is not None:
        cfg["adapters"]["alpha"] = float(args.adapter_strength)
    if args.disable_adapter:
        cfg["adapters"]["enabled"] = False

    # sampler
    s = cfg["sampler"]
    if args.duration is not None: s["duration_sec"] = int(args.duration)
    if args.temperature is not None: s["temperature"] = float(args.temperature)
    if args.top_k is not None: s["top_k"] = int(args.top_k)
    if args.top_p is not None: s["top_p"] = float(args.top_p)
    if args.cfg is not None: s["cfg_coef"] = float(args.cfg)
    if args.seed is not None: s["seed"] = int(args.seed)

    # postfx
    f = cfg["postfx"]
    if args.highpass is not None: f["highpass_hz"] = int(args.highpass)
    if args.lowpass is not None: f["lowpass_hz"] = int(args.lowpass)
    if args.limiter_drive is not None: f["limiter_drive"] = float(args.limiter_drive)
    if args.peak is not None: f["peak_target"] = float(args.peak)

    # realtime
    r = cfg["realtime"]
    if args.segment_sec is not None: r["segment_sec"] = int(args.segment_sec)
    if args.crossfade_sec is not None: r["crossfade_sec"] = float(args.crossfade_sec)
    if args.lookahead is not None: r["lookahead_segments"] = int(args.lookahead)

    # outputs
    log_cfg = cfg["logging"]
    if args.audio_dir: log_cfg["audio_dir"] = args.audio_dir
    if args.metrics_csv: log_cfg["metrics_csv"] = args.metrics_csv
    if args.logs_dir: log_cfg["logs_dir"] = args.logs_dir

    return cfg, paths

def _resolve_paths(adapter_name, paths_cfg):
    # Adapter file for the chosen name
    adapter_map = paths_cfg["paths"]["adapters"]
    adapter_file = None
    if adapter_name in adapter_map:
        adapter_file = Path(adapter_map[adapter_name])
    # Outputs
    out_map = paths_cfg["paths"]["outputs"]
    audio_dir = Path(out_map.get("audio", "outputs/audio"))
    metrics_csv = Path(out_map.get("metrics", "outputs/metrics.csv"))
    logs_dir = Path(out_map.get("logs", "outputs/logs"))
    # Prompts (used if --prompt-id/--state/--list-prompts)
    prompts = paths_cfg["paths"].get("prompts", {})
    prompt_json = Path(prompts.get("json", "prompts/emotion_prompt_bank.json"))
    prompt_csv = Path(prompts.get("csv", "prompts/emotion_prompt_bank.csv"))
    return adapter_file, audio_dir, metrics_csv, logs_dir, prompt_json, prompt_csv

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg, paths_cfg = _resolve_config(args)

    # Resolve paths
    adapter_name = cfg["adapters"].get("name", "neutral")
    adapter_file, audio_dir, metrics_csv, logs_dir, prompt_json, prompt_csv = _resolve_paths(adapter_name, paths_cfg)

    # Validate folders
    audio_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    # Prompt bank usage (list or select) happens BEFORE requiring --keyword
    if args.list_prompts or args.prompt_id is not None or args.state:
        # Load bank
        try:
            from src.control.prompt_selector import load_prompt_bank, list_prompts, get_prompt_by_id, get_prompt_for_state
        except Exception as e:
            print("[ERROR] Failed to import prompt_selector:", e); sys.exit(1)

        if not prompt_json.exists() and not prompt_csv.exists():
            print(f"[ERROR] Prompt bank not found at {prompt_json} or {prompt_csv}."); sys.exit(2)

        entries = load_prompt_bank(prompt_json, prompt_csv)

        if args.list_prompts:
            lines = list_prompts(entries)
            print("Available prompts:")
            for ln in lines:
                print(" ", ln)
            sys.exit(0)

        # Select by ID overrides state
        if args.prompt_id is not None:
            try:
                chosen = get_prompt_by_id(entries, int(args.prompt_id))
            except Exception as e:
                print(f"[ERROR] {e}"); sys.exit(2)
            keyword = chosen["prompt"]
        elif args.state:
            chosen = get_prompt_for_state(entries, args.state)
            keyword = chosen["prompt"]
        else:
            keyword = args.keyword  # fallback if something odd happens
    else:
        # No bank usage → require keyword
        if not args.keyword:
            print("[ERROR] You must pass --keyword or use --prompt-id / --state or --list-prompts.")
            sys.exit(2)
        keyword = args.keyword

    adapter_enabled = cfg["adapters"].get("enabled", True)
    alpha = float(cfg["adapters"].get("alpha", 0.0125))

    # Adapter selection rules
    chosen_adapter = "base"
    adapter_path = None
    if adapter_enabled and adapter_name != "base":
        chosen_adapter = adapter_name
        adapter_path = adapter_file
        if adapter_path is None or not adapter_path.exists():
            print(f"[ERROR] Adapter '{adapter_name}' requested but file not found at path configured.")
            print(f"        Check configs/paths.yaml under paths.adapters.{adapter_name}")
            sys.exit(2)

    # Build output filename
    slug = _slugify(keyword)
    stamp = _nowstamp()
    a_str = f"a{alpha:.4f}" if chosen_adapter != "base" else "a0.0000"
    out_name = f"{stamp}_{chosen_adapter}_{slug}_{a_str}.wav"
    out_path = audio_dir / out_name

    # Echo resolved run plan
    s = cfg["sampler"]; f = cfg["postfx"]; r = cfg["realtime"]
    print("=== BASS Run Plan ===")
    print(f"Prompt           : {keyword}")
    print(f"Adapter          : {chosen_adapter}  (enabled={adapter_enabled}, alpha={alpha})")
    if adapter_path: print(f"Adapter file     : {adapter_path}")
    print(f"Duration (s)     : {s['duration_sec']}   Seed: {s['seed']}")
    print(f"Sampler          : temp={s['temperature']} top_k={s['top_k']} top_p={s['top_p']} cfg={s['cfg_coef']}")
    print(f"Post-FX          : HPF={f['highpass_hz']}Hz LPF={f['lowpass_hz']}Hz drive={f['limiter_drive']} peak={f['peak_target']}")
    print(f"Realtime         : seg={r['segment_sec']}s xfade={r['crossfade_sec']}s lookahead={r['lookahead_segments']}")
    print(f"Outputs          : audio_dir={audio_dir}  metrics_csv={metrics_csv}")
    print(f"Target file      : {out_path}")

    # Import generation engine with proper traceback on failure
    try:
        from src.gen.engine import render_one_clip
    except Exception as e:
        import traceback
        print("[ERROR] Failed to import generation engine:", e)
        traceback.print_exc()
        sys.exit(1)

    # Run generation
    try:
        result = render_one_clip(
            keyword=keyword,
            adapter_name=chosen_adapter,
            adapter_path=str(adapter_path) if adapter_path else None,
            alpha=alpha,
            duration_sec=int(s["duration_sec"]),
            temperature=float(s["temperature"]),
            top_k=int(s["top_k"]),
            top_p=float(s["top_p"]),
            cfg_coef=float(s["cfg_coef"]),
            seed=int(s["seed"]),
            highpass_hz=int(f["highpass_hz"]),
            lowpass_hz=int(f["lowpass_hz"]),
            limiter_drive=float(f["limiter_drive"]),
            peak_target=float(f["peak_target"]),
            out_wav=str(out_path),
            metrics_csv=str(metrics_csv),
            log_level=args.log_level,
        )
        if not result:
            print("[ERROR] render_one_clip returned falsy result.")
            sys.exit(1)
        print("[OK] Render complete.")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
