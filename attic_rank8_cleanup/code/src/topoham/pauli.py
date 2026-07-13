"""Pauli-string algebra and efficient statevector Pauli rotations.

A Pauli string on ``n`` qubits is a length-``n`` string over ``{I, X, Y, Z}``,
e.g. ``"XIZ"`` is :math:`X_0 \\otimes I_1 \\otimes Z_2`. Two facts make a fast,
exact statevector Trotter simulator possible without ever forming a dense
:math:`2^n \\times 2^n` matrix for the evolution:

1. **Commutation is local.** Two Pauli strings commute iff the number of qubits
   on which their single-qubit factors *anticommute* (i.e. differ and neither is
   ``I``) is even. This is an O(n) parity check -- no matrices required. It is
   the structural fact the commutator graph (``commutator_graph.py``) is built
   from.
2. **A Pauli is an involution.** Every (non-identity) Pauli string :math:`P`
   satisfies :math:`P^2 = I`, so its eigenvalues are :math:`\\pm 1` and

   .. math:: e^{-i\\theta P} = \\cos\\theta\\, I - i\\sin\\theta\\, P .

   Hence one Trotter factor is a single sparse application
   ``cos(theta)*psi - 1j*sin(theta)*(P @ psi)`` (``apply_pauli_rotation``),
   computed in O(2^n) time by permuting and phasing amplitudes -- never by
   building or exponentiating a matrix.

The dense :func:`to_matrix` is provided only for tests and for the small-``n``
exact reference; the simulator path uses :func:`apply_pauli`.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

# Single-qubit Pauli matrices (the algebra's generators).
_I = np.array([[1, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_SINGLE: Dict[str, np.ndarray] = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}

# Whether two single-qubit factors anticommute: same letter or any I -> commute.
_PAIR_ANTICOMMUTE = {
    (a, b): (a != "I" and b != "I" and a != b) for a in "IXYZ" for b in "IXYZ"
}

# Single-qubit Pauli product table: p*q = phase * r, with phase in {1,-1,i,-i}.
# (e.g. X*Y = i Z, Y*X = -i Z); used to form the commutator error operator
# matrix-free (commutator_graph / error_form).
_SINGLE_PRODUCT = {
    ("I", "I"): ("I", 1), ("I", "X"): ("X", 1), ("I", "Y"): ("Y", 1), ("I", "Z"): ("Z", 1),
    ("X", "I"): ("X", 1), ("X", "X"): ("I", 1), ("X", "Y"): ("Z", 1j), ("X", "Z"): ("Y", -1j),
    ("Y", "I"): ("Y", 1), ("Y", "X"): ("Z", -1j), ("Y", "Y"): ("I", 1), ("Y", "Z"): ("X", 1j),
    ("Z", "I"): ("Z", 1), ("Z", "X"): ("Y", 1j), ("Z", "Y"): ("X", -1j), ("Z", "Z"): ("I", 1),
}


def num_qubits(pauli: str) -> int:
    return len(pauli)


def support(pauli: str) -> Tuple[int, ...]:
    """Qubit indices on which ``pauli`` acts non-trivially (its locality set)."""
    return tuple(i for i, p in enumerate(pauli) if p != "I")


def weight(pauli: str) -> int:
    """Pauli weight = number of non-identity factors (the locality of the term)."""
    return len(support(pauli))


def commute(p: str, q: str) -> bool:
    """True iff Pauli strings ``p`` and ``q`` commute.

    Counts the qubits on which the single-qubit factors anticommute; the strings
    commute iff that count is even. O(n), no matrices.
    """
    if len(p) != len(q):
        raise ValueError(f"length mismatch: {p!r} vs {q!r}")
    anti = 0
    for a, b in zip(p, q):
        if _PAIR_ANTICOMMUTE[(a, b)]:
            anti += 1
    return anti % 2 == 0


def anticommute(p: str, q: str) -> bool:
    """True iff ``p`` and ``q`` do NOT commute (an edge in the commutator graph)."""
    return not commute(p, q)


def pauli_product(p: str, q: str) -> Tuple[str, complex]:
    r"""Product of two Pauli strings as a phased Pauli: :math:`PQ = \omega\,R`.

    Returns ``(R, omega)`` where ``R`` is a Pauli string and ``omega`` a unit-
    modulus phase in :math:`\{1,-1,i,-i\}`. Each qubit contributes one entry of
    the single-qubit product table; phases multiply. Runs in :math:`O(n)` with no
    matrices. This is the algebraic kernel behind the *commutator error form*:
    for an anticommuting pair :math:`P_j,P_k` the commutator is
    :math:`[c_jP_j,c_kP_k]=2c_jc_k\,P_jP_k`, a phased Pauli, so the leading
    Trotter error operator is a sum of phased Paulis assembled with this routine.
    """
    if len(p) != len(q):
        raise ValueError(f"length mismatch: {p!r} vs {q!r}")
    phase: complex = 1 + 0j
    out = []
    for a, b in zip(p, q):
        r, ph = _SINGLE_PRODUCT[(a, b)]
        out.append(r)
        phase *= ph
    return "".join(out), phase


@lru_cache(maxsize=4096)
def to_matrix(pauli: str) -> np.ndarray:
    """Dense :math:`2^n \\times 2^n` matrix of a Pauli string (Kronecker product).

    Used by the tests and the small-``n`` exact reference only; the simulator
    never calls this on the full Hamiltonian.
    """
    mat = np.array([[1.0 + 0j]])
    for ch in pauli:
        mat = np.kron(mat, _SINGLE[ch])
    return mat


def apply_pauli(state: np.ndarray, pauli: str) -> np.ndarray:
    r"""Return :math:`P\,|\psi\rangle` without forming the dense matrix of ``P``.

    Each single-qubit factor is applied by reshaping the statevector so the
    target qubit is an axis, then permuting/phasing along that axis:

      * ``X`` swaps the two halves,
      * ``Y`` swaps and applies :math:`\mp i`,
      * ``Z`` phases the ``|1\rangle`` half by :math:`-1`,
      * ``I`` is a no-op.

    Total cost O(n * 2^n), exact, matches ``to_matrix(pauli) @ state``.
    """
    n = len(pauli)
    if state.shape[0] != (1 << n):
        raise ValueError(f"state dim {state.shape[0]} != 2**{n}")
    psi = np.asarray(state, dtype=complex).reshape([2] * n)
    # Qubit q is axis q (big-endian: leftmost letter is qubit 0 = most significant
    # bit). Apply each single-qubit factor along its axis into a fresh array so no
    # in-place aliasing can corrupt later factors.
    for q, ch in enumerate(pauli):
        if ch == "I":
            continue
        a0 = np.take(psi, 0, axis=q)
        a1 = np.take(psi, 1, axis=q)
        out = np.empty_like(psi)
        sl0 = [slice(None)] * n
        sl1 = [slice(None)] * n
        sl0[q] = 0
        sl1[q] = 1
        if ch == "X":
            out[tuple(sl0)] = a1
            out[tuple(sl1)] = a0
        elif ch == "Y":
            out[tuple(sl0)] = -1j * a1
            out[tuple(sl1)] = 1j * a0
        elif ch == "Z":
            out[tuple(sl0)] = a0
            out[tuple(sl1)] = -a1
        psi = out
    return psi.reshape(1 << n)


def apply_pauli_rotation(state: np.ndarray, pauli: str, theta: float) -> np.ndarray:
    r"""Apply one Trotter factor :math:`e^{-i\theta P}` to ``state``.

    Because :math:`P^2 = I`, :math:`e^{-i\theta P} = \cos\theta\,I -
    i\sin\theta\,P`, so this is a single :func:`apply_pauli` plus a scaled add.
    This is the inner kernel of the entire Trotter simulator (``env.py``); the
    in-browser JS demo applies the exact same identity.
    """
    if pauli == "I" * len(pauli):  # global identity term -> pure global phase
        return np.exp(-1j * theta) * np.asarray(state, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    return c * np.asarray(state, dtype=complex) - 1j * s * apply_pauli(state, pauli)


def zero_state(n: int) -> np.ndarray:
    r"""Computational-basis :math:`|0\rangle^{\otimes n}` statevector."""
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    return psi


def plus_state(n: int) -> np.ndarray:
    r"""Uniform superposition :math:`|+\rangle^{\otimes n}` (a generic probe state)."""
    return np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=complex)


def expectation(state: np.ndarray, pauli: str) -> float:
    r"""Real expectation :math:`\langle\psi|P|\psi\rangle` (P is Hermitian)."""
    return float(np.real(np.vdot(state, apply_pauli(state, pauli))))
