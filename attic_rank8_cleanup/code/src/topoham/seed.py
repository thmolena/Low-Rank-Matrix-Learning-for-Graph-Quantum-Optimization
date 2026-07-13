"""Deterministic seeding and run provenance.

Every experiment logs seed, command, package versions, runtime, memory and a
timestamp (see COPILOT_INSTRUCTIONS). Provenance is attached to every results
file so that no claimed number is untraceable to a concrete run.
"""
from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed all relevant RNGs and return a numpy Generator for local use."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "networkx", "sklearn", "matplotlib"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # pragma: no cover - optional deps
            versions[mod] = "absent"
    return versions


def _peak_memory_mb() -> float:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes.
        return usage / (1024 ** 2) if sys.platform == "darwin" else usage / 1024
    except Exception:  # pragma: no cover
        return float("nan")


@dataclass
class RunProvenance:
    """Captures the metadata required for every reproducible run."""

    seed: int
    command: str = field(default_factory=lambda: " ".join(sys.argv))
    platform: str = field(default_factory=platform.platform)
    versions: Dict[str, str] = field(default_factory=_package_versions)
    timestamp_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    runtime_sec: float = 0.0
    peak_memory_mb: float = 0.0
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def finalize(self) -> "RunProvenance":
        self.runtime_sec = round(time.perf_counter() - self._t0, 4)
        self.peak_memory_mb = round(_peak_memory_mb(), 2)
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_t0", None)
        return d
