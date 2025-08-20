from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Type, Tuple, Optional, List

class HRSource(ABC):
    """Unified interface all HR sources must implement."""

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    @abstractmethod
    def connect(self) -> None:
        """Open connections / auth etc. Non-blocking."""
        ...

    @abstractmethod
    def wait_for_window(self, seconds: float, timeout: float, default: float) -> Tuple[int, float, float, float]:
        """
        Block until we have ~`seconds` of HR samples (or timeout).
        Returns: (sample_count, span_sec, mean_bpm, waited_sec).
        Use `default` if needed.
        """
        ...

    @abstractmethod
    def get_bpm(self, default: float) -> float:
        """Return latest rolling-average BPM (or default)."""
        ...

    def baseline_bpm(self) -> Optional[float]:
        """Optional: return resting/baseline HR if the source provides it."""
        return None

    def disconnect(self) -> None:
        """Optional cleanup."""
        ...

    def close(self) -> None:
        """Optional cleanup."""
        ...


_SOURCE_REGISTRY: Dict[str, Type[HRSource]] = {}

def register_source(name: str):
    """Decorator to register a concrete HRSource under a CLI name."""
    def deco(cls: Type[HRSource]) -> Type[HRSource]:
        _SOURCE_REGISTRY[name.lower()] = cls
        return cls
    return deco

def create_source(name: str, **kwargs) -> HRSource:
    key = (name or "").lower()
    if key not in _SOURCE_REGISTRY:
        raise ValueError(f"Unknown HR source '{name}'. Available: {sorted(_SOURCE_REGISTRY.keys())}")
    return _SOURCE_REGISTRY[key](**kwargs)

def available_sources() -> List[str]:
    return sorted(_SOURCE_REGISTRY.keys())
