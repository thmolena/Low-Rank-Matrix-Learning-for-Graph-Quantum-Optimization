"""YAML experiment configuration.

Loads the experiment protocol of the paper (Methods, "Hamiltonian families and
protocol"): the Hamiltonian ``families`` to sweep, system ``sizes`` ``n``, number
of seeded instances per (family, size), evolution ``time`` ``t``, fixed Trotter
``steps`` ``r`` (so the matched gate proxy is ``n_terms * r``), the ``steps_grid``
used for the gate-budget frontier, and the exact ``reference_backend``. ``smoke``
is the fast demo configuration; ``full`` is the reported-scale configuration whose
numbers appear in the manuscript.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    name: str = "smoke"
    seed: int = 0
    families: List[str] = field(default_factory=lambda: [
        "tfim", "heisenberg", "molecular_like",
    ])
    sizes: List[int] = field(default_factory=lambda: [3, 4])
    n_per_family: int = 3
    # Trotter parameters.
    time: float = 1.0
    steps: int = 4                       # the reference (matched-budget) step count
    steps_grid: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8, 12])
    reference_backend: str = "expm"      # "expm" (dense) or "krylov"
    # Scheduling study parameters.
    schedules: List[str] = field(default_factory=lambda: [
        "random", "coefficient", "commutator", "antithetic", "symmetric", "learned",
    ])
    targets: List[float] = field(default_factory=lambda: [0.9, 0.99, 0.999])
    target_ref: float = 0.99             # target the learned router optimises for
    impotence_samples: int = 48          # random orderings per instance (Theorem 1)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)
