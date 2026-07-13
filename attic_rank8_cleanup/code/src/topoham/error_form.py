r"""The commutator error form: the exact leading first-order Trotter error
operator, assembled matrix-free as a signed sum of Pauli strings.

Background
----------
For :math:`H=\sum_j h_j` with :math:`h_j=c_jP_j` and an ordering :math:`\pi`, one
first-order Trotter step :math:`S_\pi(\Delta)=\prod_a e^{-i\Delta h_{\pi(a)}}`
has single-step error

.. math::
   e^{-i\Delta H}-S_\pi(\Delta) = -\tfrac{\Delta^2}{2}\,E_\pi + O(\Delta^3),
   \qquad E_\pi=\sum_{a<b}[h_{\pi(a)},h_{\pi(b)}].

``E_\pi`` is the **leading error operator**. For Pauli terms an anticommuting
pair contributes :math:`[c_jP_j,c_kP_k]=2c_jc_k\,P_jP_k` (a phased Pauli) and a
commuting pair contributes zero, so

.. math::
   E_\pi=\sum_{\{j,k\}\in E(\mathcal G)} \varepsilon_{jk}(\pi)\,
          2c_jc_k\,(P_jP_k),

where :math:`\varepsilon_{jk}(\pi)=+1` if :math:`j` precedes :math:`k` in
:math:`\pi` and :math:`-1` otherwise. **Re-ordering only flips the signs
:math:`\varepsilon`** — it never changes which commutators appear (the
*ordering-impotence* phenomenon). Two pairs whose product lands on the same
Pauli string are a *collision*; only the collisions give a sign-dependent (hence
ordering-reducible) contribution to :math:`\lVert E_\pi\rVert`.

Everything here is computed without forming a :math:`2^n\times2^n` matrix; the
dense routines are provided only to *validate* the matrix-free form (they agree
to machine precision) and to report the exact spectral norm on small systems.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from . import pauli
from .hamiltonians import Hamiltonian

# A leading-error form is a mapping {Pauli string R -> complex coefficient}.
ErrorForm = Dict[str, complex]


def leading_error_form(ham: Hamiltonian, ordering: Sequence[int]) -> ErrorForm:
    r"""Matrix-free coefficients of :math:`E_\pi=\sum_{a<b}[h_{\pi(a)},h_{\pi(b)}]`.

    Returns ``{R: coeff}`` so that :math:`E_\pi=\sum_R \mathrm{coeff}(R)\,R`. Each
    ordered anticommuting pair adds :math:`2c_ac_b\,\omega` to the entry of the
    Pauli string :math:`R` with :math:`P_aP_b=\omega R`. Cost :math:`O(L^2 n)`.
    """
    coeffs = ham.coeffs
    paulis = ham.paulis
    order = list(ordering)
    acc: ErrorForm = {}
    L = len(order)
    for a in range(L):
        ja = order[a]
        pa = paulis[ja]
        ca = coeffs[ja]
        for b in range(a + 1, L):
            jb = order[b]
            pb = paulis[jb]
            if pauli.commute(pa, pb):
                continue
            r, omega = pauli.pauli_product(pa, pb)   # P_a P_b = omega * R
            acc[r] = acc.get(r, 0j) + 2.0 * ca * coeffs[jb] * omega
    return acc


def leading_error_hs(ham: Hamiltonian, ordering: Sequence[int]) -> float:
    r"""Hilbert--Schmidt norm of :math:`E_\pi`, normalised by :math:`2^{n/2}`.

    Because distinct Pauli strings are Hilbert--Schmidt orthogonal with
    :math:`\lVert R\rVert_{HS}^2=2^n`, the (per-:math:`\sqrt{2^n}`) HS norm is
    :math:`\sqrt{\sum_R |\mathrm{coeff}(R)|^2}` — a fully matrix-free surrogate
    for the size of the leading error.
    """
    acc = leading_error_form(ham, ordering)
    return float(np.sqrt(sum(abs(v) ** 2 for v in acc.values())))


def dense_leading_error(ham: Hamiltonian, ordering: Sequence[int]) -> np.ndarray:
    r"""Dense :math:`E_\pi` by summing pair commutators (validation/spectral norm).

    Used only on small systems to (a) cross-check :func:`leading_error_form` and
    (b) report the exact spectral norm. Never used on the simulation path.
    """
    coeffs = ham.coeffs
    paulis = ham.paulis
    order = list(ordering)
    dim = 1 << ham.n
    mats = {p: pauli.to_matrix(p) for p in set(paulis)}
    hj = [coeffs[j] * mats[paulis[j]] for j in range(len(ham))]
    E = np.zeros((dim, dim), dtype=complex)
    L = len(order)
    for a in range(L):
        Ha = hj[order[a]]
        for b in range(a + 1, L):
            Hb = hj[order[b]]
            E += Ha @ Hb - Hb @ Ha
    return E


def leading_error_opnorm(ham: Hamiltonian, ordering: Sequence[int]) -> float:
    r"""Exact spectral norm :math:`\lVert E_\pi\rVert` (largest singular value)."""
    if len(ham) < 2:
        return 0.0
    return float(np.linalg.norm(dense_leading_error(ham, ordering), 2))


def form_to_dense(ham: Hamiltonian, form: ErrorForm) -> np.ndarray:
    """Reconstruct the dense operator from a matrix-free error form."""
    dim = 1 << ham.n
    E = np.zeros((dim, dim), dtype=complex)
    for r, v in form.items():
        E += v * pauli.to_matrix(r)
    return E


def collision_stats(ham: Hamiltonian) -> Dict[str, float]:
    r"""Structure of the commutator error form that governs ordering-reducibility.

    ``n_anti_pairs``  number of anticommuting term pairs (edges of the graph);
    ``distinct_paulis`` number of distinct error Pauli strings :math:`R`;
    ``colliding_pairs`` pairs that share an :math:`R` with another pair (the only
                        sign-/ordering-reducible part of the error);
    ``irreducible_frac`` fraction of edges that map to a *unique* :math:`R` and
                        are therefore ordering-invariant in HS norm.
    """
    paulis = ham.paulis
    L = len(ham)
    mult: Dict[str, int] = {}
    for a in range(L):
        for b in range(a + 1, L):
            if pauli.commute(paulis[a], paulis[b]):
                continue
            r, _ = pauli.pauli_product(paulis[a], paulis[b])
            mult[r] = mult.get(r, 0) + 1
    n_edges = int(sum(mult.values()))
    colliding = int(sum(v for v in mult.values() if v >= 2))
    return {
        "n_anti_pairs": float(n_edges),
        "distinct_paulis": float(len(mult)),
        "colliding_pairs": float(colliding),
        "irreducible_frac": round((n_edges - colliding) / n_edges, 6) if n_edges else 1.0,
    }


def _pair_commutators(ham: Hamiltonian) -> Dict:
    """Precompute the dense commutator of every anticommuting ordered pair once,
    so an ordering's :math:`E_\\pi` is assembled by summation (no re-multiplying)."""
    coeffs = ham.coeffs
    paulis = ham.paulis
    mats = {p: pauli.to_matrix(p) for p in set(paulis)}
    hj = [coeffs[j] * mats[paulis[j]] for j in range(len(ham))]
    C: Dict = {}
    L = len(ham)
    for a in range(L):
        for b in range(a + 1, L):
            if pauli.commute(paulis[a], paulis[b]):
                continue
            C[(a, b)] = hj[a] @ hj[b] - hj[b] @ hj[a]   # [h_a, h_b]
    return C


def _opnorm_from_commutators(order: Sequence[int], C: Dict, dim: int) -> float:
    pos = {j: i for i, j in enumerate(order)}
    E = np.zeros((dim, dim), dtype=complex)
    for (a, b), Cab in C.items():
        # +[h_a,h_b] if a precedes b in this ordering, else the reverse sign.
        E += Cab if pos[a] < pos[b] else -Cab
    return float(np.linalg.norm(E, 2))


def ordering_impotence(ham: Hamiltonian, rng: np.random.Generator,
                       n_samples: int = 48) -> Dict[str, float]:
    r"""Quantify how little any reordering can shrink :math:`\lVert E_\pi\rVert`.

    Samples random orderings plus the coefficient ordering and reports the spread
    of the leading-error spectral norm. Reordering only re-signs the precomputed
    pair commutators, so the whole sweep reuses one set of dense commutators.
    ``ratio_coeff_to_min`` close to 1 means a strong fixed heuristic is already
    near the floor (Theorem 1); the matrix-free HS surrogate is reported too.
    """
    from . import policies
    L = len(ham)
    if L < 2:
        return {"op_min": 0.0, "op_median": 0.0, "op_max": 0.0,
                "ratio_coeff_to_min": 1.0, "ratio_median_to_min": 1.0,
                "hs_ratio_max_to_min": 1.0}
    C = _pair_commutators(ham)
    dim = 1 << ham.n
    coeff_order = policies.coefficient_ordering(ham)
    orders: List[List[int]] = [coeff_order]
    for _ in range(n_samples):
        idx = np.arange(L)
        rng.shuffle(idx)
        orders.append(idx.tolist())
    ops = np.array([_opnorm_from_commutators(o, C, dim) for o in orders], dtype=float)
    hss = np.array([leading_error_hs(ham, o) for o in orders], dtype=float)
    op_min = float(ops.min())
    return {
        "op_min": round(op_min, 6),
        "op_median": round(float(np.median(ops)), 6),
        "op_max": round(float(ops.max()), 6),
        "ratio_coeff_to_min": round(float(ops[0] / max(op_min, 1e-12)), 6),
        "ratio_median_to_min": round(float(np.median(ops) / max(op_min, 1e-12)), 6),
        "hs_ratio_max_to_min": round(float(hss.max() / max(hss.min(), 1e-12)), 6),
    }


def verify_error_form(rng: np.random.Generator, tol: float = 1e-9) -> bool:
    """Integrity gate: the matrix-free error form equals the dense pair-commutator
    sum to ``tol`` on random small Hamiltonians (records ``error_form_verified``).
    """
    from .hamiltonians import build, FAMILIES
    ok = True
    for fam in FAMILIES:
        for n in (3, 4):
            ham = build(fam, n, rng)
            order = np.arange(len(ham))
            rng.shuffle(order)
            order = order.tolist()
            E_form = form_to_dense(ham, leading_error_form(ham, order))
            E_dense = dense_leading_error(ham, order)
            if not np.allclose(E_form, E_dense, atol=tol):
                ok = False
    return ok
