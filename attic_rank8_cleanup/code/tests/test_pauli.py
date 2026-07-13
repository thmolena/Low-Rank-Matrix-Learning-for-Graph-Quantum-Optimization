"""Pauli algebra and the fast statevector rotation kernel.

The central correctness guarantee: the O(2^n) rotation
``cos(theta) psi - i sin(theta) (P psi)`` -- the inner loop of the whole Trotter
simulator and of the in-browser demo -- agrees with the unambiguous dense
``expm(-i theta P) @ psi`` to numerical precision.
"""
import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from topoham import pauli


def test_commute_known_pairs():
    # Single-qubit: X and Z anticommute; X and X commute; anything with I commutes.
    assert pauli.commute("X", "X")
    assert not pauli.commute("X", "Z")
    assert not pauli.commute("Y", "Z")
    assert pauli.commute("X", "I")
    # Two-qubit bond operators: XX and ZZ commute (two anticommuting sites -> even).
    assert pauli.commute("XX", "ZZ")
    assert pauli.commute("YY", "ZZ")
    # XX and ZI share exactly one anticommuting site -> anticommute.
    assert not pauli.commute("XX", "ZI")
    assert pauli.anticommute("XX", "ZI")


def test_commute_matches_matrix_definition():
    # Brute-force check commute() against the matrix commutator on all 2-qubit pairs.
    strings = ["".join(s) for s in itertools.product("IXYZ", repeat=2)]
    for p in strings:
        for q in strings:
            A, B = pauli.to_matrix(p), pauli.to_matrix(q)
            comm_zero = np.allclose(A @ B - B @ A, 0.0)
            assert pauli.commute(p, q) == comm_zero, (p, q)


def test_to_matrix_is_involution():
    for p in ["XIZ", "YYX", "ZZI", "XYZ"]:
        M = pauli.to_matrix(p)
        assert np.allclose(M @ M, np.eye(M.shape[0]))            # P^2 = I
        assert np.allclose(M, M.conj().T)                        # Hermitian


@pytest.mark.parametrize("n", [1, 2, 3])
def test_apply_pauli_matches_matrix(n):
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = "".join(rng.choice(list("IXYZ"), size=n))
        psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        assert np.allclose(pauli.apply_pauli(psi, p), pauli.to_matrix(p) @ psi, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_apply_pauli_rotation_matches_expm(n):
    rng = np.random.default_rng(7)
    for _ in range(12):
        p = "".join(rng.choice(list("IXYZ"), size=n))
        theta = float(rng.uniform(-np.pi, np.pi))
        psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        psi /= np.linalg.norm(psi)
        fast = pauli.apply_pauli_rotation(psi, p, theta)
        exact = expm(-1j * theta * pauli.to_matrix(p)) @ psi
        assert np.allclose(fast, exact, atol=1e-9), (p, theta)


def test_rotation_preserves_norm():
    rng = np.random.default_rng(3)
    psi = pauli.plus_state(3)
    out = pauli.apply_pauli_rotation(psi, "XYZ", 0.9)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-12
