"""Trotter convergence and ordering invariants."""
import numpy as np
import pytest

from topoham import hamiltonians as H
from topoham import metrics, pauli
from topoham.env import TrotterEnv, reference_state, trotter_state


def test_trotter_converges_to_exact():
    # As r -> large, first-order Trotter fidelity -> 1 for any fixed ordering.
    rng = np.random.default_rng(1)
    ham = H.tfim(4)
    t, psi0 = 1.5, pauli.plus_state(ham.n)
    ref = reference_state(ham, t, psi0)
    order = list(range(len(ham)))
    fids = [metrics.fidelity(ref, trotter_state(ham, t, r, order, psi0))
            for r in (1, 2, 4, 8, 32, 128)]
    # monotone-ish increase and convergence to 1
    assert fids[-1] > 0.999
    assert fids[-1] >= fids[0]


@pytest.mark.parametrize("family", H.FAMILIES)
def test_reference_backends_agree(family):
    rng = np.random.default_rng(5)
    ham = H.build(family, 4, rng)
    psi0 = pauli.plus_state(ham.n)
    a = reference_state(ham, 1.0, psi0, backend="expm")
    b = reference_state(ham, 1.0, psi0, backend="krylov")
    assert np.allclose(a, b, atol=1e-8)


def test_exact_reference_is_ordering_independent():
    # The *exact* evolution does not depend on a term ordering; only the Trotter
    # approximation does. Reference fidelity to itself is 1 regardless of order.
    rng = np.random.default_rng(2)
    ham = H.molecular_like(4, 8, rng)
    psi0 = pauli.plus_state(ham.n)
    ref = reference_state(ham, 1.0, psi0)
    assert metrics.fidelity(ref, ref) == pytest.approx(1.0)
    # And at very large r every ordering reaches the same (near-1) fidelity.
    o1 = list(range(len(ham)))
    o2 = list(reversed(o1))
    env = TrotterEnv.from_hamiltonian(ham, 1.0, 256, psi0)
    f1 = env.evaluate(o1)["fidelity"]
    f2 = env.evaluate(o2)["fidelity"]
    assert abs(f1 - f2) < 1e-3 and f1 > 0.999


def test_gate_proxy_is_ordering_independent():
    rng = np.random.default_rng(9)
    ham = H.heisenberg(4)
    env = TrotterEnv.from_hamiltonian(ham, 1.0, 3, pauli.plus_state(ham.n))
    a = env.evaluate(list(range(len(ham))))
    b = env.evaluate(list(reversed(range(len(ham)))))
    assert a["gate_proxy"] == b["gate_proxy"] == len(ham) * 3
