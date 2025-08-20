"""
ISO-style prompt generator + session API for BASS live sessions.

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

Logic (session API):
  • Lock one 'base' phrase once per session (chosen with a PROMPT SEED).
  • Per stage, sample (texture, fx) WITHOUT REPLACEMENT, deterministically.
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
from typing import Dict, List, Tuple, Optional

# ---- default CSV locations (relative to repo root) ---------------------------
PROMPTS_DIR = Path("prompts")
CSV_PATHS_DEFAULT: Dict[str, Path] = {
    "energy": PROMPTS_DIR / "energy_prompts.csv",
    "neutral": PROMPTS_DIR / "neutral_prompts.csv",
    "calm": PROMPTS_DIR / "calm_prompts.csv",
}

# ---- ISO progression per physiological state (project terms) -----------------
ISO_PLAN: Dict[str, List[str]] = {
    "stress":  ["energy", "neutral", "calm", "calm"],
    "neutral": ["neutral", "neutral", "calm", "calm"],
    "under":   ["neutral",  "energy", "neutral", "calm"],
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

def _tempo_for_stage(stage: str) -> str:
    return {"energy": "fast tempo", "neutral": "moderate tempo", "calm": "slow tempo"}[stage]

_STAGE_SALT = {"energy": 1, "neutral": 2, "calm": 3}

def _build_stage_orders(rows_by_stage: Dict[str, List[Dict[str, str]]], prompt_seed: int
                        ) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
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

        rng = Random(_mix_seed(sum(order), len(order) * 1103515245 + 12345))
        rng.shuffle(order)
        cursors[stage] = 0
        cur = 0
    row = rows[order[cur]]
    cursors[stage] = cur + 1
    return row


class PromptSession:
    """
    Holds locked base and per-stage, no-repeat texture/fx samplers.
    Allows requesting a prompt for ANY stage at runtime.
    """
    def __init__(self, *, session_seed: int, csv_paths: Dict[str, Path] | None = None,
                 nonce: Optional[int] = None):
        if nonce is None:
            nonce = time.time_ns()
        prompt_seed = _mix_seed(int(session_seed), int(nonce))

        self.rows_by_stage = _load_all_csvs(csv_paths)
        bases = _shared_bases(self.rows_by_stage)
        if not bases:
            raise ValueError("No 'base' phrases found in prompt CSVs.")
        self._rng = Random(int(prompt_seed))
        self.locked_base = self._rng.choice(bases)

        self.orders, self.cursors = _build_stage_orders(self.rows_by_stage, int(prompt_seed))
        self.seg_idx = 0  # incremented when you call next_prompt

    def next_prompt_for(self, stage: str) -> PromptPiece:
        stage = (stage or "").lower()
        if stage not in ("energy", "neutral", "calm"):
            stage = "neutral"
        row = _draw_row_for_stage(stage, self.rows_by_stage, self.orders, self.cursors)
        texture = (row.get("texture") or "").strip()
        fx = (row.get("fx") or "").strip()
        tempo = _tempo_for_stage(stage)
        piece = PromptPiece(
            segment=self.seg_idx,
            stage=stage,
            base=self.locked_base,
            tempo=tempo,
            texture=texture,
            fx=fx,
        )
        self.seg_idx += 1
        return piece


def build_prompt_session(*, session_seed: int, csv_paths: Dict[str, Path] | None = None,
                         nonce: Optional[int] = None) -> PromptSession:
    return PromptSession(session_seed=session_seed, csv_paths=csv_paths, nonce=nonce)

def generate_iso_prompts(
    *,
    phys_state: str,
    session_seed: int,
    segments: int = 4,
    csv_paths: Dict[str, Path] | None = None,
    nonce: int | None = None,
) -> List[PromptPiece]:
    """
    Backwards-compatible helper: produce a fixed ISO plan’s prompts.
    """
    phys_key = (phys_state or "").strip().lower()
    if phys_key not in ISO_PLAN:
        if phys_key in {"stressed"}:
            phys_key = "stress"
        elif phys_key in {"under_aroused", "underaroused", "low", "low_energy"}:
            phys_key = "under"
        else:
            phys_key = "neutral"

    plan = ISO_PLAN[phys_key]
    if segments > len(plan):
        plan = plan + [plan[-1]] * (segments - len(plan))
    else:
        plan = plan[:segments]

    sess = build_prompt_session(session_seed=session_seed, csv_paths=csv_paths, nonce=nonce)
    out: List[PromptPiece] = []
    for _idx, stage in enumerate(plan):
        out.append(sess.next_prompt_for(stage))
    return out
