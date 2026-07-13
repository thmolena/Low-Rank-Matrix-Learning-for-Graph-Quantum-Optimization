"""Hamiltonian family generators.

A Hamiltonian is represented as a list of ``(coefficient, pauli_string)`` terms,

    H = sum_j  c_j * P_j ,

with real coefficients ``c_j`` and Pauli strings ``P_j`` over ``n`` qubits (see
``pauli.py``). This is the standard local-Pauli decomposition used for
qubit Hamiltonians arising from spin lattices and (Jordan-Wigner / Bravyi-Kitaev
mapped) molecular electronic structure problems.

Three families are provided, spanning the structural regimes that matter for
Trotter ordering:

* :func:`tfim` -- transverse-field Ising, a chain with two non-commuting layers
  (the ``ZZ`` couplings and the ``X`` fields). Few distinct anticommuting pairs.
* :func:`heisenberg` -- ``XX + YY + ZZ`` chain; the three bond operators on a
  shared edge mutually commute but neighbouring bonds interleave.
* :func:`molecular_like` -- random *local* 2-body Pauli terms with random
  coefficients (seeded), a tractable stand-in for the dense, irregular
  anticommutation structure of mapped molecular Hamiltonians.

Each Hamiltonian carries a tiny ``meta`` dict (family, n) for bookkeeping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

Term = Tuple[float, str]


@dataclass
class Hamiltonian:
    """A real-coefficient sum of Pauli-string terms on ``n`` qubits."""

    n: int
    terms: List[Term]
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.terms)

    @property
    def coeffs(self) -> List[float]:
        return [c for c, _ in self.terms]

    @property
    def paulis(self) -> List[str]:
        return [p for _, p in self.terms]


def _site(op: str, i: int, n: int) -> str:
    """Pauli string with single-qubit ``op`` on qubit ``i`` and ``I`` elsewhere."""
    s = ["I"] * n
    s[i] = op
    return "".join(s)


def _bond(op: str, i: int, j: int, n: int) -> str:
    """Pauli string with ``op`` on qubits ``i`` and ``j``, ``I`` elsewhere."""
    s = ["I"] * n
    s[i] = op
    s[j] = op
    return "".join(s)


def tfim(n: int, J: float = 1.0, h: float = 1.0,
         periodic: bool = False) -> Hamiltonian:
    r"""Transverse-field Ising model.

    .. math:: H = -J \sum_i Z_i Z_{i+1} \; -\; h \sum_i X_i .

    The ``ZZ`` couplings all commute with each other, as do the ``X`` fields, but
    a coupling and a field sharing a qubit anticommute -- the canonical
    two-non-commuting-layer structure that first-order Trotter splitting targets.
    """
    terms: List[Term] = []
    last = n if periodic else n - 1
    for i in range(last):
        terms.append((-J, _bond("Z", i, (i + 1) % n, n)))
    for i in range(n):
        terms.append((-h, _site("X", i, n)))
    return Hamiltonian(n, terms, {"family": "tfim", "J": J, "h": h})


def heisenberg(n: int, Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0,
               periodic: bool = False) -> Hamiltonian:
    r"""Heisenberg / XXZ chain.

    .. math:: H = \sum_i ( J_x X_iX_{i+1} + J_y Y_iY_{i+1} + J_z Z_iZ_{i+1} ).

    The three operators on one bond mutually commute; consecutive bonds share a
    qubit and interleave, giving a richer anticommutation graph than the TFIM.
    """
    terms: List[Term] = []
    last = n if periodic else n - 1
    for i in range(last):
        j = (i + 1) % n
        terms.append((Jx, _bond("X", i, j, n)))
        terms.append((Jy, _bond("Y", i, j, n)))
        terms.append((Jz, _bond("Z", i, j, n)))
    return Hamiltonian(n, terms, {"family": "heisenberg",
                                  "Jx": Jx, "Jy": Jy, "Jz": Jz})


def molecular_like(n: int, n_terms: int, rng: np.random.Generator,
                   weight: int = 2, coeff_scale: float = 1.0) -> Hamiltonian:
    r"""Random local 2-body Pauli Hamiltonian (seeded, deduplicated).

    A tractable proxy for a mapped molecular electronic-structure Hamiltonian:
    each term acts on ``weight`` distinct qubits with random non-identity Paulis
    and a Gaussian coefficient. The resulting anticommutation graph is dense and
    irregular -- the regime where term ordering matters most for Trotter error.
    """
    if n < weight:
        raise ValueError("n must be >= weight")
    seen: Dict[str, float] = {}
    paulis = "XYZ"
    # Bound the number of distinct weight-k terms to avoid an infinite loop.
    from math import comb
    max_terms = comb(n, weight) * (3 ** weight)
    target = min(n_terms, max_terms)
    while len(seen) < target:
        sites = rng.choice(n, size=weight, replace=False)
        s = ["I"] * n
        for q in sites:
            s[q] = paulis[int(rng.integers(0, 3))]
        key = "".join(s)
        if key not in seen:
            seen[key] = float(rng.normal(0.0, coeff_scale))
    terms = [(c, p) for p, c in seen.items()]
    return Hamiltonian(n, terms, {"family": "molecular_like", "weight": weight})


# Registry consumed by the runner / config; values are (builder, needs_rng).
FAMILIES = ("tfim", "heisenberg", "molecular_like")


def build(family: str, n: int, rng: np.random.Generator,
          n_terms: int | None = None) -> Hamiltonian:
    """Dispatch a family name to its builder with sensible per-family defaults."""
    if family == "tfim":
        return tfim(n)
    if family == "heisenberg":
        return heisenberg(n)
    if family == "molecular_like":
        nt = n_terms if n_terms is not None else max(4, 2 * n)
        return molecular_like(n, nt, rng)
    raise ValueError(f"unknown family {family!r}")
