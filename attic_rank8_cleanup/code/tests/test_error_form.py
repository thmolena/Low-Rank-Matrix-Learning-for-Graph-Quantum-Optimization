"""The commutator error form: matrix-free == dense, and ordering impotence.

These tests pin the new theoretical object. The matrix-free leading-error form
must equal the dense pair-commutator sum to machine precision (the integrity
gate), and -- because reordering only re-signs each pair's commutator -- the
leading-error norm must be exactly ordering-invariant on families with no
commutator collisions (Theorem 1).
"""
import numpy as np
import pytest

from topoham import hamiltonians as H
from topoham import error_form as ef
from topoham import pauli, policies


def test_pauli_product_table():
    assert pauli.pauli_product("X", "Y") == ("Z", 1j)
    assert pauli.pauli_product("Y", "X") == ("Z", -1j)
    assert pauli.pauli_product("Z", "Z") == ("I", 1)
    R, ph = pauli.pauli_product("XZ", "ZX")
    # (X*Z)(Z*X) = (-iY)(+iY) on the two qubits -> Y Y with phase (-i)(i)=1
    assert R == "YY" and abs(ph - 1) < 1e-12


@pytest.mark.parametrize("family", H.FAMILIES)
def test_matrix_free_form_equals_dense(family):
    rng = np.random.default_rng(3)
    ham = H.build(family, 4, rng)
    order = list(np.random.default_rng(1).permutation(len(ham)))
    E_form = ef.form_to_dense(ham, ef.leading_error_form(ham, order))
    E_dense = ef.dense_leading_error(ham, order)
    assert np.allclose(E_form, E_dense, atol=1e-10)


def test_verify_error_form_gate():
    assert ef.verify_error_form(np.random.default_rng(0)) is True


def test_reordering_only_resigns_no_collision_families():
    # TFIM / Heisenberg have no commutator collisions, so the HS norm of the
    # leading error is exactly ordering-invariant.
    for ham in (H.tfim(5), H.heisenberg(5)):
        stats = ef.collision_stats(ham)
        assert stats["colliding_pairs"] == 0
        assert stats["irreducible_frac"] == 1.0
        a = ef.leading_error_hs(ham, policies.coefficient_ordering(ham))
        b = ef.leading_error_hs(ham, list(reversed(range(len(ham)))))
        c = ef.leading_error_hs(ham, list(np.random.default_rng(9).permutation(len(ham))))
        assert abs(a - b) < 1e-9 and abs(a - c) < 1e-9


def test_ordering_impotence_ratio_near_one():
    # No reordering can shrink ||E|| far below a strong heuristic's value.
    rng = np.random.default_rng(0)
    ham = H.molecular_like(6, 14, rng)
    imp = ef.ordering_impotence(ham, np.random.default_rng(1), n_samples=32)
    assert imp["ratio_median_to_min"] < 1.6
    assert imp["op_min"] > 0
