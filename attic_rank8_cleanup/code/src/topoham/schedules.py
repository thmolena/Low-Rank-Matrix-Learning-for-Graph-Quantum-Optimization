r"""Term *schedules* for product-formula simulation.

The free degree of freedom in a product formula at a fixed *set* of per-step
rotations is not a single ordering but a **schedule**: the ordering may vary
between Trotter steps, and the step may be *folded* symmetrically. A schedule is
realised as a flat list of ``(term_index, step_fraction)`` rotations; the angle
applied for an entry is :math:`c_j\,(t/r)\,\text{fraction}`. The total rotation
count of the (seam-merged) list is the gate cost on which all schedules are
compared.

Folds
-----
``first_order``  every step applies the ordering :math:`\pi` once (the classic
                 first-order schedule). Cost :math:`L\cdot r`.
``antithetic``   the ordering alternates direction by step,
                 :math:`\pi,\pi^{R},\pi,\pi^{R},\dots`. Each adjacent step-pair
                 :math:`S_{\pi^R}(\Delta)S_\pi(\Delta)` is a *palindrome*, hence a
                 symmetric (second-order) formula whose leading :math:`O(\Delta^2)`
                 commutator error cancels — at the **identical** :math:`L\cdot r`
                 rotation count (no seam merge applied).
``symmetric``    the Strang half-step fold :math:`\prod e^{-i\frac{\Delta}{2}h_{\pi(a)}}
                 \prod e^{-i\frac{\Delta}{2}h_{\pi(L-a)}}` with seam merges; a
                 lower-constant second-order formula at :math:`\approx 2Lr` rotations.

Ordering source
---------------
``coefficient``  terms by :math:`|c_j|` (a strong first-order heuristic);
``clique``       commuting colour-cliques of the commutator graph laid out
                 contiguously (cliques are exact super-terms: their internal
                 rotations commute, so only *inter-clique* commutators survive).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import pauli, policies
from .commutator_graph import greedy_color_groups
from .hamiltonians import Hamiltonian

FlatSchedule = List[Tuple[int, float]]

# Schedules compared in the study (references handled separately by the env).
SCHEDULES = ("random", "coefficient", "commutator", "antithetic", "symmetric", "learned")
# The non-learned schedules the learned policy selects among.
FIXED_SCHEDULES = ("random", "coefficient", "commutator", "antithetic", "symmetric")


# ---------------------------------------------------------------------------
# Ordering sources
# ---------------------------------------------------------------------------
def clique_order(ham: Hamiltonian) -> List[int]:
    """Lay the commuting colour-cliques out contiguously (largest-weight first).

    A proper colouring of the commutator graph partitions the terms into
    mutually-commuting cliques. Placing each clique contiguously makes its
    internal rotations adjacent; since they commute, the schedule is exact within
    a clique and only inter-clique commutators contribute to the error.
    """
    groups = greedy_color_groups(ham)
    coeffs = [abs(c) for c in ham.coeffs]
    groups = sorted(groups, key=lambda g: (-sum(coeffs[j] for j in g), min(g)))
    order: List[int] = []
    for g in groups:
        order.extend(sorted(g, key=lambda j: (-coeffs[j], j)))
    return order


def _source_order(source: str, ham: Hamiltonian, rng: np.random.Generator) -> List[int]:
    if source == "coefficient":
        return policies.coefficient_ordering(ham)
    if source == "clique":
        return clique_order(ham)
    if source == "random":
        return policies.random_ordering(ham, rng)
    raise ValueError(f"unknown ordering source {source!r}")


# ---------------------------------------------------------------------------
# Folds -> flat (term_index, step_fraction) lists
# ---------------------------------------------------------------------------
def first_order_flat(order: Sequence[int], r: int) -> FlatSchedule:
    return [(int(j), 1.0) for _ in range(r) for j in order]


def antithetic_flat(order: Sequence[int], r: int) -> FlatSchedule:
    order = list(order)
    rev = list(reversed(order))
    flat: FlatSchedule = []
    for k in range(r):
        seq = order if k % 2 == 0 else rev
        flat.extend((int(j), 1.0) for j in seq)
    return flat


def symmetric_flat(order: Sequence[int], r: int) -> FlatSchedule:
    order = list(order)
    rev = list(reversed(order))
    flat: FlatSchedule = []
    for _ in range(r):
        flat.extend((int(j), 0.5) for j in order)
        flat.extend((int(j), 0.5) for j in rev)
    return flat


def _merge(flat: FlatSchedule) -> FlatSchedule:
    """Merge adjacent rotations on the same term (seam merge: e^{a}e^{b}=e^{a+b})."""
    merged: List[List[float]] = []
    for j, u in flat:
        if merged and merged[-1][0] == j:
            merged[-1][1] += u
        else:
            merged.append([j, u])
    return [(int(j), float(u)) for j, u in merged]


# ---------------------------------------------------------------------------
# Named schedules: (ordering source, fold, merge policy)
# ---------------------------------------------------------------------------
# merge=False on antithetic keeps the count exactly L*r (the zero-overhead claim);
# merge=True on symmetric gives the minimal Strang rotation count.
_SPEC: Dict[str, Tuple[str, str, bool]] = {
    "random":      ("random",      "first_order", False),
    "coefficient": ("coefficient", "first_order", False),
    "commutator":  ("clique",      "first_order", False),
    "antithetic":  ("clique",      "antithetic",  False),
    "symmetric":   ("clique",      "symmetric",   True),
}


def build_flat(name: str, ham: Hamiltonian, r: int,
               rng: np.random.Generator) -> Tuple[FlatSchedule, bool]:
    """Return the (flat schedule, already-merged?) for a fixed schedule name."""
    source, fold, merge = _SPEC[name]
    order = _source_order(source, ham, rng)
    if fold == "first_order":
        flat = first_order_flat(order, r)
    elif fold == "antithetic":
        flat = antithetic_flat(order, r)
    elif fold == "symmetric":
        flat = symmetric_flat(order, r)
    else:  # pragma: no cover
        raise ValueError(fold)
    if merge:
        flat = _merge(flat)
    return flat, merge


def apply_flat(ham: Hamiltonian, t: float, r: int, flat: FlatSchedule,
               psi0: np.ndarray) -> Tuple[np.ndarray, int]:
    """Execute a flat schedule; return ``(state, n_rotations)``.

    ``n_rotations`` (the gate cost) is the length of the executed list — the
    quantity all schedules are compared at.
    """
    dt = t / r
    psi = np.asarray(psi0, dtype=complex).copy()
    terms = ham.terms
    for j, u in flat:
        c, p = terms[j]
        psi = pauli.apply_pauli_rotation(psi, p, c * dt * u)
    return psi, len(flat)


def gate_count(name: str, ham: Hamiltonian, r: int,
               rng: np.random.Generator | None = None) -> int:
    """Total rotation count of a named schedule (the matched-cost x-axis)."""
    flat, _ = build_flat(name, ham, r, rng or np.random.default_rng(0))
    return len(flat)
