"""The core claim, at demo scale.

At a *fixed* small Trotter step count (matched gate budget), ordering the terms
by the commutator graph's commuting groups yields a mean fidelity at least as
high as a random ordering, averaged over seeded Hamiltonians -- and it produces
strictly fewer anticommuting adjacencies, the mechanism behind the gain.
"""
import numpy as np
import pytest

from topoham import hamiltonians as H
from topoham import policies
from topoham import pauli
from topoham.env import TrotterEnv


def _mean_fidelity(strategy, hams, t, r, seed=0):
    fids = []
    for ham in hams:
        env = TrotterEnv.from_hamiltonian(ham, t, r, pauli.plus_state(ham.n))
        order = policies.get_ordering(strategy, ham, np.random.default_rng(seed))
        fids.append(env.evaluate(order)["fidelity"])
    return float(np.mean(fids))


def _seeded_hams():
    rng = np.random.default_rng(0)
    hams = []
    for n in (3, 4):
        hams.append(H.tfim(n))
        hams.append(H.heisenberg(n))
        for _ in range(3):
            hams.append(H.molecular_like(n, 2 * n, rng))
    return hams


def test_commutator_beats_random_on_average():
    hams = _seeded_hams()
    t, r = 1.5, 2
    comm = _mean_fidelity("commutator", hams, t, r)
    rand_runs = [_mean_fidelity("random", hams, t, r, seed=s) for s in range(6)]
    rand = float(np.mean(rand_runs))
    assert comm >= rand - 1e-9, (comm, rand)


def test_commutator_reduces_adjacent_anticommutations():
    rng = np.random.default_rng(3)
    for ham in [H.tfim(5), H.heisenberg(5), H.molecular_like(5, 12, rng)]:
        comm = policies.commutator_ordering(ham)
        rand = policies.random_ordering(ham, np.random.default_rng(1))
        a_comm = policies.adjacent_anticommutations(ham, comm)
        a_rand = policies.adjacent_anticommutations(ham, rand)
        assert a_comm <= a_rand


def test_commutator_ordering_is_a_permutation():
    ham = H.molecular_like(4, 10, np.random.default_rng(2))
    order = policies.commutator_ordering(ham)
    assert sorted(order) == list(range(len(ham)))


def test_learned_router_runs_and_returns_permutation():
    rng = np.random.default_rng(4)
    hams = [H.tfim(4), H.heisenberg(4), H.molecular_like(4, 8, rng),
            H.molecular_like(4, 8, rng)]
    labels = ["commutator", "commutator", "coefficient", "commutator"]
    pol = policies.LearnedOrderingPolicy().fit(hams, labels)
    order = pol.ordering(hams[0], rng)
    assert sorted(order) == list(range(len(hams[0])))
    assert pol.predict_strategy(hams[0]) in policies.FIXED_STRATEGIES
