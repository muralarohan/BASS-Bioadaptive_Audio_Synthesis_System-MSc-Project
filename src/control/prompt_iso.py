# src/control/prompt_iso.py
"""
ISO-style prompt generator for BASS live sessions.

Terminology aligned with this project:

Physiological state (from HR mapper / live warm-up):
  - "stress" | "neutral" | "under"

Musical stages used in generation:
  - "energy" (base model, no adapter)
  - "neutral" (neutral adapter)
  - "calm"   (calm adapter)

Input CSVs (under prompts/):
  - energy_prompts.csv
  - neutral_prompts.csv
  - calm_prompts.csv

Each CSV must have at least the columns:
  base, texture, fx

Logic:
  • Lock one 'base' phrase once per session (chosen with a PROMPT SEED).
  • Per segment, pick (texture, fx) from the active stage’s CSV using a
    deterministic RNG and **without replacement** within the session.
  • Tempo wording is plain per stage: energy=fast, neutral=moderate, calm=slow.
  • Assemble: "{base}, {tempo}, {texture}, {fx}"

Determinism & variability:
  • Model seed (in live.py) can stay fixed for timbre/behavior stability.
  • Prompt selection uses a *prompt seed* = mix(session_seed, nonce).
    - If you don't pass a nonce, we use time.time_ns() so each run differs.
    - If you want fully reproducible sessions, pass a fixed nonce.
"""

from __future__ import annotations
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Tuple


# ---- default CSV locations (relative to repo root) ---------------------------
PROMPTS_DIR = Path("prompts")
CSV_PATHS_DEFAULT: Dict[str, Path] = {
    "energy": PROMPTS_DIR / "energy_prompts.csv",
    "neutral": PROMPTS_DIR / "neutral_prompts.csv",
    "calm": PROMPTS_DIR / "calm_prompts.csv",
}

# ---- ISO progression per physiological state (project terms) -----------------
ISO_PLAN: Dict[str, List[str]] = {
    "stress":  ["neutral", "neutral", "calm", "calm"],
    "neutral": ["neutral", "neutral", "calm", "calm"],
    "under":   ["energy",  "neutral", "neutral", "calm"],
}


@dataclass(frozen=True)
class PromptPiece:
    segment: int        # 0-based
    stage: str          # "energy" | "neutral" | "calm"
    base: str
    tempo: str
    texture: str
    fx: str

    @property
    def text(self) -> str:
        parts = [self.base]
        if self.tempo:
            parts.append(self.tempo)
        if self.texture:
            parts.append(self.texture)
        if self.fx:
            parts.append(self.fx)
        return ", ".join([p for p in parts if p and p.strip()])


# ---- CSV loading --------------------------------------------------------------
def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    required = {"base", "texture", "fx"}
    missing = [c for c in required if (rows and c not in rows[0])]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing} (needs {sorted(required)})")
    return rows


def _load_all_csvs(csv_paths: Dict[str, Path] | None = None) -> Dict[str, List[Dict[str, str]]]:
    paths = csv_paths or CSV_PATHS_DEFAULT
    return {stage: _load_csv_rows(p) for stage, p in paths.items()}


def _shared_bases(rows_by_stage: Dict[str, List[Dict[str, str]]]) -> List[str]:
    if "neutral" not in rows_by_stage or not rows_by_stage["neutral"]:
        for st in ("energy", "calm"):
            if rows_by_stage.get(st):
                return [row.get("base", "").strip() for row in rows_by_stage[st] if row.get("base", "").strip()]
        return []
    return [row.get("base", "").strip() for row in rows_by_stage["neutral"] if row.get("base", "").strip()]


# ---- seed mixing --------------------------------------------------------------
_MASK64 = (1 << 64) - 1

def _mix_seed(a: int, b: int) -> int:
    """Deterministic 64-bit mix (no reliance on Python's salted hash)."""
    x = (a & _MASK64) ^ ((b * 0x9E3779B97F4A7C15) & _MASK64)
    x ^= (x >> 33)
    x = (x * 0xff51afd7ed558ccd) & _MASK64
    x ^= (x >> 33)
    x = (x * 0xc4ceb9fe1a85ec53) & _MASK64
    x ^= (x >> 33)
    return x & _MASK64


def _pick_locked_base(bases: List[str], prompt_seed: int) -> str:
    if not bases:
        raise ValueError("No 'base' phrases found in prompt CSVs.")
    rng = Random(int(prompt_seed))
    return rng.choice(bases)


# ---- Tempo wording (plain per stage; no ramps) --------------------------------
def _tempo_for_stage(stage: str) -> str:
    return {"energy": "fast tempo", "neutral": "moderate tempo", "calm": "slow tempo"}[stage]


# ---- Per-stage no-repeat sampler ---------------------------------------------
_STAGE_SALT = {"energy": 1, "neutral": 2, "calm": 3}

def _build_stage_orders(rows_by_stage: Dict[str, List[Dict[str, str]]], prompt_seed: int
                        ) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """
    For each stage, build a shuffled order of row indices using a stage-specific RNG.
    Returns (orders, cursors) where orders[stage] is a list of indices, and cursors[stage] tracks next position.
    """
    orders: Dict[str, List[int]] = {}
    cursors: Dict[str, int] = {}
    for stage, rows in rows_by_stage.items():
        n = len(rows)
        if n == 0:
            orders[stage] = []
            cursors[stage] = 0
            continue
        rng = Random(_mix_seed(prompt_seed, _STAGE_SALT.get(stage, 0)))
        idxs = list(range(n))
        rng.shuffle(idxs)
        orders[stage] = idxs
        cursors[stage] = 0
    return orders, cursors


def _draw_row_for_stage(stage: str, rows_by_stage: Dict[str, List[Dict[str, str]]],
                        orders: Dict[str, List[int]], cursors: Dict[str, int]) -> Dict[str, str]:
    rows = rows_by_stage.get(stage, [])
    if not rows:
        return {"texture": "", "fx": ""}
    order = orders[stage]
    cur = cursors[stage]
    if cur >= len(order):
        # exhausted → reshuffle deterministically by advancing a small LCG step
        # (keeps reproducibility while avoiding immediate repeats)
        rng = Random(_mix_seed(sum(order), len(order) * 1103515245 + 12345))
        rng.shuffle(order)
        cursors[stage] = 0
        cur = 0
    row = rows[order[cur]]
    cursors[stage] = cur + 1
    return row


# ---- Public API ---------------------------------------------------------------
def generate_iso_prompts(
    *,
    phys_state: str,
    session_seed: int,
    segments: int = 4,
    csv_paths: Dict[str, Path] | None = None,
    nonce: int | None = None,   # new: mix-in to vary prompts across runs with same session_seed
) -> List[PromptPiece]:
    """
    Generate 'segments' prompts following the locked ISO progression for the given
    physiological state ("stress" | "neutral" | "under").

    Seeds:
      - prompt_seed = mix(session_seed, nonce or time.time_ns())
      - 'base' chosen with Random(prompt_seed)
      - per-stage rows sampled without replacement using prompt_seed & stage salt

    To reproduce a past session exactly, record & pass the 'nonce' you used.
    """
    phys_key = (phys_state or "").strip().lower()
    if phys_key not in ISO_PLAN:
        if phys_key in {"stressed"}:
            phys_key = "stress"
        elif phys_key in {"under_aroused", "underaroused", "low", "low_energy"}:
            phys_key = "under"
        else:
            phys_key = "neutral"

    # Build prompt seed (decoupled from model seed)
    if nonce is None:
        nonce = time.time_ns()
    prompt_seed = _mix_seed(int(session_seed), int(nonce))

    # Load CSVs and pick a locked base
    rows_by_stage = _load_all_csvs(csv_paths)
    bases = _shared_bases(rows_by_stage)
    locked_base = _pick_locked_base(bases, int(prompt_seed))

    # Per-stage deterministic, no-repeat order
    orders, cursors = _build_stage_orders(rows_by_stage, int(prompt_seed))

    # Build the stage plan
    plan = ISO_PLAN[phys_key]
    if segments > len(plan):
        plan = plan + [plan[-1]] * (segments - len(plan))
    else:
        plan = plan[:segments]

    out: List[PromptPiece] = []
    prev_stage: str | None = None  # retained for future use if we reintroduce transition wording

    for seg_idx, stage in enumerate(plan):
        row = _draw_row_for_stage(stage, rows_by_stage, orders, cursors)
        texture = (row.get("texture") or "").strip()
        fx = (row.get("fx") or "").strip()
        tempo = _tempo_for_stage(stage)

        piece = PromptPiece(
            segment=seg_idx,
            stage=stage,
            base=locked_base,
            tempo=tempo,
            texture=texture,
            fx=fx,
        )
        out.append(piece)
        prev_stage = stage

    return out
