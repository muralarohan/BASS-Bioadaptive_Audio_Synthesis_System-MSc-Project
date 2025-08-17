# src/control/state_mapper.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class HRRules:
    stress_delta_bpm: int = 15   # baseline +15 => stress
    under_delta_bpm:  int = -10  # baseline -10 => under
    # window/sustain are used in streaming mode; single-shot uses thresholds only
    window_sec: int = 10
    sustain_sec: int = 30

class HRStateMapper:
    def __init__(self, rules: HRRules | None = None):
        self.rules = rules or HRRules()

    def map_bpm(self, baseline_bpm: float, current_bpm: float) -> str:
        """
        Single-shot mapping from baseline & current BPM to state.
        Returns one of: 'stress' | 'neutral' | 'under'
        """
        d = float(current_bpm) - float(baseline_bpm)
        if d >= self.rules.stress_delta_bpm:
            return "stress"
        if d <= self.rules.under_delta_bpm:
            return "under"
        return "neutral"
