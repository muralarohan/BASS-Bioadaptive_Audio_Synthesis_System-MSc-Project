#!/usr/bin/env python
"""
Join metrics.csv + session_log.csv and produce a simple stage dashboard.

Outputs:
- outputs/plots/session_dashboard.png
- outputs/plots/summary.csv (joined per-segment table for analysis)

Usage:
  python tools/plot_metrics.py
  # or with custom paths:
  python tools/plot_metrics.py --metrics outputs/metrics.csv --session outputs/session_log.csv --outdir outputs/plots
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="outputs/metrics.csv")
    ap.add_argument("--session", default="outputs/session_log.csv")
    ap.add_argument("--outdir", default="outputs/plots")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    session_path = Path(args.session)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    m = load_csv(metrics_path)  # columns include: file, rms, peak, crest_db, zcr, centroid_hz, hf_ratio, flatness, ...
    s = load_csv(session_path)  # columns include: file, stage_type, phys_init, schedule_id, hr_avg_bpm, ...

    # Join on file (robust join)
    for col in ("file", "stage_type"):
        if col not in s.columns:
            raise ValueError(f"session_log.csv missing required column: {col}")
    if "file" not in m.columns:
        raise ValueError("metrics.csv missing required column: file")

    df = pd.merge(s, m, on="file", how="inner")

    # Save joined table for your analysis
    df.to_csv(outdir / "summary.csv", index=False)

    # Aggregate by stage
    stage_order = ["energy", "neutral", "calm"]
    df["stage_type"] = pd.Categorical(df["stage_type"].str.lower(), categories=stage_order, ordered=True)
    agg = df.groupby("stage_type", observed=True).agg(
        rms_mean=("rms", "mean"),
        crest_mean=("crest_db", "mean"),
        centroid_mean=("centroid_hz", "mean"),
        hf_ratio_mean=("hf_ratio", "mean"),
        flatness_mean=("flatness", "mean"),
        rms_std=("rms", "std"),
        centroid_std=("centroid_hz", "std"),
        hf_ratio_std=("hf_ratio", "std"),
    ).reset_index()

    # Simple panels
    panels = [
        ("Average Spectral Centroid (Hz)", "centroid_mean"),
        ("Average HF Ratio (≥6–8 kHz band)", "hf_ratio_mean"),
        ("Average RMS (linear)", "rms_mean"),
        ("Average Crest (dB)", "crest_mean"),
    ]

    # Make a single dashboard figure with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=120)
    axes = axes.ravel()

    x = np.arange(len(stage_order))
    for ax, (title, col) in zip(axes, panels):
        y = []
        for st in stage_order:
            row = agg[agg["stage_type"] == st]
            y.append(float(row[col]) if not row.empty else np.nan)
        ax.bar(x, y)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_order)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("BASS Session Dashboard (by Stage)", y=0.98)
    fig.tight_layout()
    fig.savefig(outdir / "session_dashboard.png")
    plt.close(fig)

    print(f"[OK] Wrote: {outdir / 'session_dashboard.png'}")
    print(f"[OK] Wrote: {outdir / 'summary.csv'}")

if __name__ == "__main__":
    main()
