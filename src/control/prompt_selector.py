# src/control/prompt_selector.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json, csv

# Heuristics for column names in CSV/JSON entries
PROMPT_COLS = ("prompt", "text", "description", "content")
STATE_COLS  = ("state", "mood", "emotion", "target", "category")
TITLE_COLS  = ("title", "name", "label")
ID_COLS     = ("id", "idx", "index")

def _norm_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip().lower()
    aliases = {
        "calm": "calm",
        "relax": "calm",
        "relaxed": "calm",
        "neutral": "neutral",
        "stress": "stress",
        "stressed": "stress",
        "high": "stress",
        "under": "under",
        "under-aroused": "under",
        "under_aroused": "under",
        "low": "under",
    }
    return aliases.get(s, s)

def _coerce_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _extract_promptish(d: Dict[str, Any]) -> Tuple[Optional[int], str, Optional[str], Optional[str]]:
    """Return (id, prompt, state, title) from a generic dict."""
    pid = None
    for k in ID_COLS:
        if k in d:
            pid = _coerce_int(d[k], None); break

    prompt = None
    for k in PROMPT_COLS:
        if k in d and isinstance(d[k], str) and d[k].strip():
            prompt = d[k].strip(); break

    state = None
    for k in STATE_COLS:
        if k in d and isinstance(d[k], str) and d[k].strip():
            state = _norm_state(d[k]); break

    title = None
    for k in TITLE_COLS:
        if k in d and isinstance(d[k], str) and d[k].strip():
            title = d[k].strip(); break

    return pid, prompt, state, title

def load_prompt_bank(json_path: Path, csv_path: Path) -> List[Dict[str, Any]]:
    """
    Load prompt bank from JSON (preferred) or CSV (fallback).
    Returns a list of normalized dicts: {id, prompt, state, title, raw}
    """
    entries: List[Dict[str, Any]] = []

    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        items = data.get("prompts", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("JSON prompt bank must be a list or have key 'prompts'.")
        for row in items:
            if not isinstance(row, dict):
                continue
            pid, prompt, state, title = _extract_promptish(row)
            if not prompt:
                continue
            entries.append({
                "id": pid,
                "prompt": prompt,
                "state": state,
                "title": title,
                "raw": row
            })
    elif csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                pid, prompt, state, title = _extract_promptish(row)
                if not prompt:
                    # Try concatenating likely columns if prompt missing
                    maybe = [row.get(k, "") for k in PROMPT_COLS if k in row]
                    joined = ", ".join([s for s in maybe if s]).strip()
                    if joined:
                        prompt = joined
                if not prompt:
                    continue
                entries.append({
                    "id": pid,
                    "prompt": prompt,
                    "state": state,
                    "title": title,
                    "raw": row
                })
    else:
        raise FileNotFoundError(f"No prompt bank at {json_path} or {csv_path}")

    # Assign sequential IDs if missing
    next_id = 1
    for e in entries:
        if e["id"] is None:
            e["id"] = next_id
            next_id += 1
    return entries

def list_prompts(entries: List[Dict[str, Any]], limit: Optional[int] = None) -> List[str]:
    """Return formatted strings for display."""
    rows = []
    for e in entries[: limit or len(entries)]:
        pid = e["id"]
        state = e.get("state") or "-"
        title = e.get("title") or "-"
        prompt = e["prompt"].replace("\n", " ")
        if len(prompt) > 80:
            prompt = prompt[:77] + "..."
        rows.append(f"[{pid}] state={state:8s}  title={title:20s}  prompt={prompt}")
    return rows

def get_prompt_by_id(entries: List[Dict[str, Any]], pid: int) -> Dict[str, Any]:
    for e in entries:
        if int(e["id"]) == int(pid):
            return e
    raise KeyError(f"Prompt id {pid} not found.")

def get_prompt_for_state(entries: List[Dict[str, Any]], state: str) -> Dict[str, Any]:
    state = _norm_state(state)
    # 1) exact state match
    for e in entries:
        if _norm_state(e.get("state")) == state:
            return e
    # 2) fuzzy contains in raw fields
    key = state or ""
    key = key.lower()
    for e in entries:
        raw = e.get("raw", {})
        hay = " ".join([str(raw.get(k, "")) for k in STATE_COLS + TITLE_COLS + PROMPT_COLS]).lower()
        if key and key in hay:
            return e
    # 3) fallback: first entry
    return entries[0] if entries else {"id": 0, "prompt": "neutral ambient, slow tempo, soft pads", "state": "neutral", "title": None}
