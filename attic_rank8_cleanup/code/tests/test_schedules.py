"""Term schedules: the zero-overhead order upgrade and clique exactness.

Pins the two central claims of the scheduling primitive:
  * the antithetic fold uses *exactly* L*r rotations (the same as first-order) yet
    converges at second order (its infidelity slope is ~4, not ~2);
  * a commuting-clique schedule is exact within each clique (Heisenberg becomes
    essentially exact at a small step count).
"""
import math

import numpy as np
import pytest

from topoham import hamiltonians as H
from topoham import pauli, schedules
from topoham.env import reference_state


def _infid(ham, name, t, r):
    psi0 = pauli.plus_state(ham.n)
    ref = reference_state(ham, t, psi0)
    flat, _ = schedules.build_flat(name, ham, r, np.random.default_rng(0))
    psi, _ = schedules.apply_flat(ham, t, r, flat, psi0)
    return 1.0 - abs(np.vdot(ref, psi)) ** 2


def test_antithetic_gate_count_is_exactly_Lr():
    ham = H.molecular_like(6, 14, np.random.default_rng(0))
    L = len(ham)
    for r in (2, 4, 8):
        assert schedules.gate_count("antithetic", ham, r) == L * r
        # first-order schedules share the identical budget
        assert schedules.gate_count("coefficient", ham, r) == L * r


def test_symmetric_is_second_order_two_terms_per_step():
    ham = H.tfim(5)
    L = len(ham)
    # Strang fold applies each term twice per step with seam merges.
    assert schedules.gate_count("symmetric", ham, 4) <= 2 * L * 4
    assert schedules.gate_count("symmetric", ham, 4) > L * 4


@pytest.mark.parametrize("family", ["tfim", "molecular_like"])
def test_antithetic_converges_at_second_order(family):
    rng = np.random.default_rng(2)
    ham = H.build(family, 6, rng)
    t = 1.0
    rs = [8, 16, 32]
    first = [_infid(ham, "coefficient", t, r) for r in rs]
    anti = [_infid(ham, "antithetic", t, r) for r in rs]

    def slope(vals):
        x = np.log(rs); y = np.log(vals)
        return float(-np.polyfit(x, y, 1)[0])

    assert 1.6 < slope(first) < 2.6          # first order ~ 2
    assert slope(anti) > 3.2                  # antithetic ~ 4 (second order)
    assert anti[-1] < first[-1]               # and lower error at fine r


def test_clique_schedule_is_exact_on_heisenberg():
    ham = H.heisenberg(6)
    # commuting-clique (commutator) first-order schedule is near-exact at small r
    assert _infid(ham, "commutator", 2.0, 2) < 1e-6


def test_apply_flat_gate_count_matches_len():
    ham = H.tfim(4)
    for name in schedules.FIXED_SCHEDULES:
        flat, _ = schedules.build_flat(name, ham, 3, np.random.default_rng(0))
        psi0 = pauli.plus_state(ham.n)
        _, gates = schedules.apply_flat(ham, 1.0, 3, flat, psi0)
        assert gates == len(flat)
