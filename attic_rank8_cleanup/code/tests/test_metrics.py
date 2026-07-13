"""Metric sanity: fidelity bounds, observable error, summaries."""
import numpy as np
import pytest

from topoham import hamiltonians as H
from topoham import metrics, pauli
from topoham.env import reference_state, trotter_state


def test_fidelity_in_unit_interval():
    rng = np.random.default_rng(0)
    ham = H.molecular_like(4, 8, rng)
    psi0 = pauli.plus_state(ham.n)
    ref = reference_state(ham, 1.0, psi0)
    for r in (1, 2, 3, 8):
        psi = trotter_state(ham, 1.0, r, list(range(len(ham))), psi0)
        f = metrics.fidelity(ref, psi)
        assert 0.0 <= f <= 1.0


def test_fidelity_self_is_one():
    psi = pauli.plus_state(3)
    assert metrics.fidelity(psi, psi) == pytest.approx(1.0)


def test_observable_error_nonnegative_and_zero_for_identical():
    psi = pauli.zero_state(3)
    assert metrics.observable_error(psi, psi) == pytest.approx(0.0)
    other = pauli.apply_pauli(psi, "XII")  # flips qubit 0 -> <Z0> changes sign
    assert metrics.observable_error(psi, other) > 0.0


def test_gate_proxy_and_summarize():
    assert metrics.gate_proxy(5, 3) == 15
    s = metrics.summarize([0.9, 0.8, 0.95, 0.85])
    assert 0.0 <= s["mean"] <= 1.0
    assert s["n"] == 4
    assert s["ci95"] >= 0.0
    empty = metrics.summarize([])
    assert empty["n"] == 0
