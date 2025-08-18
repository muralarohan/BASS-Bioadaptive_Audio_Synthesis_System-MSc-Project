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
  • Lock one 'base' phrase once per session (chosen deterministically from the shared base list).
  • For each segment, pick (texture, fx) from the active stage’s CSV using a segment-seeded RNG.
  • Apply gentle tempo wording per stage and only soften wording when the stage changes.
  • Assemble: "{base}, {tempo}, {texture}, {fx}"
  • Deterministic across runs: Random(session_seed + segment_idx).
"""

from __future__ import annotations
import csv
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
# We accept "stress" | "neutral" | "under" and map to the 4-stage plan.
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
        # Compose final prompt
        parts = [self.base]
        if self.tempo:
            parts.append(self.tempo)
        if self.texture:
            parts.append(self.texture)
        if self.fx:
            parts.append(self.fx)
        # Join with comma+space, avoid double commas/spaces
        return ", ".join([p for p in parts if p and p.strip()])


# ---- CSV loading --------------------------------------------------------------
def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    # Basic column validation
    required = {"base", "texture", "fx"}
    missing = [c for c in required if (rows and c not in rows[0])]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing} (needs {sorted(required)})")
    return rows


def _load_all_csvs(csv_paths: Dict[str, Path] | None = None) -> Dict[str, List[Dict[str, str]]]:
    paths = csv_paths or CSV_PATHS_DEFAULT
    return {stage: _load_csv_rows(p) for stage, p in paths.items()}


def _shared_bases(rows_by_stage: Dict[str, List[Dict[str, str]]]) -> List[str]:
    """
    Assumes the same 'base' list across CSVs (as per plan).
    If they differ, we take the 'neutral' CSV as source of truth.
    """
    if "neutral" not in rows_by_stage or not rows_by_stage["neutral"]:
        # fallback: take the first available non-empty stage
        for st in ("energy", "calm"):
            if rows_by_stage.get(st):
                return [row.get("base", "").strip() for row in rows_by_stage[st] if row.get("base", "").strip()]
        return []
    return [row.get("base", "").strip() for row in rows_by_stage["neutral"] if row.get("base", "").strip()]


def _pick_locked_base(bases: List[str], session_seed: int) -> str:
    if not bases:
        raise ValueError("No 'base' phrases found in prompt CSVs.")
    rng = Random(int(session_seed))
    return rng.choice(bases)


# ---- Tempo wording (gentle transitions when stage changes) -------------------
def _tempo_for_stage(stage: str) -> str:
    # default wording when stage stays the same
    return {"energy": "fast tempo", "neutral": "moderate tempo", "calm": "slow tempo"}[stage]


def _tempo_override(curr_stage: str, prev_stage: str | None) -> str:
    """
    Apply softer transition wording when stage changes between segments.
    """
    if prev_stage is None or prev_stage == curr_stage:
        return _tempo_for_stage(curr_stage)

    # Stage changed → soften directionally
    if curr_stage == "energy":
        # coming up from neutral/calm
        return "slightly faster tempo"
    if curr_stage == "neutral":
        if prev_stage == "energy":
            return "moderately lively"
        if prev_stage == "calm":
            return "slightly slower tempo"
        return "moderate tempo"
    if curr_stage == "calm":
        # coming down from energy/neutral
        return "gradually slowing"
    return _tempo_for_stage(curr_stage)


# ---- Per-segment randomized texture/fx (deterministic by seed) ---------------
def _random_texture_fx(rows: List[Dict[str, str]], rng: Random) -> Tuple[str, str]:
    if not rows:
        return "", ""
    pick = rng.choice(rows)
    texture = (pick.get("texture") or "").strip()
    fx = (pick.get("fx") or "").strip()
    return texture, fx


# ---- Public API ---------------------------------------------------------------
def generate_iso_prompts(
    *,
    phys_state: str,
    session_seed: int,
    segments: int = 4,
    csv_paths: Dict[str, Path] | None = None,
) -> List[PromptPiece]:
    """
    Generate 'segments' prompts following the locked ISO progression for the given
    physiological state ("stress" | "neutral" | "under").

    Determinism:
      - Locked base chosen with Random(session_seed)
      - Per-segment texture/fx chosen with Random(session_seed + segment_idx)

    Returns a list of PromptPiece objects (with .text property).
    """
    phys_key = (phys_state or "").strip().lower()
    if phys_key not in ISO_PLAN:
        # Accept a couple of synonyms just in case
        if phys_key in {"stressed"}:
            phys_key = "stress"
        elif phys_key in {"under_aroused", "underaroused", "low", "low_energy"}:
            phys_key = "under"
        else:
            phys_key = "neutral"

    # Load CSVs and pick a locked base
    rows_by_stage = _load_all_csvs(csv_paths)
    bases = _shared_bases(rows_by_stage)
    locked_base = _pick_locked_base(bases, int(session_seed))

    # Build the stage plan
    plan = ISO_PLAN[phys_key]
    if segments > len(plan):
        # hold the last stage for extra segments
        plan = plan + [plan[-1]] * (segments - len(plan))
    else:
        plan = plan[:segments]

    out: List[PromptPiece] = []
    prev_stage: str | None = None

    for seg_idx, stage in enumerate(plan):
        rng = Random(int(session_seed) + seg_idx)
        rows = rows_by_stage.get(stage, [])
        texture, fx = _random_texture_fx(rows, rng)
        tempo = _tempo_override(stage, prev_stage)

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
