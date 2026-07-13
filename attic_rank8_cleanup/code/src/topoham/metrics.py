"""Trotter-error metrics and summary statistics.

The quality of an approximate time-evolution :math:`|\\psi_{\\mathrm{trot}}\\rangle`
against the exact reference :math:`|\\psi_{\\mathrm{exact}}\\rangle` is measured by:

* :func:`fidelity` -- the state overlap :math:`|\\langle\\psi_{\\mathrm{exact}}|
  \\psi_{\\mathrm{trot}}\\rangle|^2 \\in [0,1]` (1 = perfect);
* :func:`observable_error` -- the absolute error in a physical observable
  (here :math:`\\langle Z_0\\rangle`), the quantity a practitioner actually reads
  off a simulation;
* and the *gate proxy* (``num_terms * r``), the implementation cost held fixed
  when comparing orderings.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import pauli


def fidelity(psi_exact: np.ndarray, psi_trotter: np.ndarray) -> float:
    """State fidelity :math:`|\\langle a|b\\rangle|^2`, clipped to ``[0, 1]``."""
    overlap = np.vdot(psi_exact, psi_trotter)
    f = float(np.abs(overlap) ** 2)
    return min(1.0, max(0.0, f))


def infidelity(psi_exact: np.ndarray, psi_trotter: np.ndarray) -> float:
    return 1.0 - fidelity(psi_exact, psi_trotter)


def observable_error(psi_exact: np.ndarray, psi_trotter: np.ndarray,
                     observable: str | None = None) -> float:
    """Absolute error in :math:`\\langle O\\rangle` between the two states.

    Defaults to :math:`O = Z_0` on the qubit count implied by the statevectors.
    """
    n = int(round(np.log2(psi_exact.shape[0])))
    if observable is None:
        observable = "Z" + "I" * (n - 1)
    ex = pauli.expectation(psi_exact, observable)
    tr = pauli.expectation(psi_trotter, observable)
    return abs(ex - tr)


def gate_proxy(n_terms: int, r: int) -> int:
    """Implementation cost of a first-order Trotter schedule: one rotation per
    term per step. Held fixed across orderings at matched ``(n_terms, r)``."""
    return int(n_terms) * int(r)


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": round(mean, 6), "std": round(std, 6), "ci95": round(ci95, 6), "n": int(arr.size)}
