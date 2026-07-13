"""Term-ordering policies for first-order Trotterization.

Every policy is a map ``Hamiltonian -> permutation of its term indices``. The
permutation is the order in which the per-term rotations are applied within a
Trotter step (``env.trotter_state``). Since the gate proxy ``n_terms * r`` does
not depend on the order, **all policies are compared at identical cost** -- the
only thing that changes is the Trotter error.

Policies
--------
``random``       a reference baseline: a seeded shuffle.
``coefficient``  terms sorted by :math:`|c_j|` descending -- a chemistry folklore
                 heuristic (apply the large-norm rotations in a fixed place).
``locality``     terms sorted by their qubit support, so spatially-overlapping
                 (and thus often anticommuting) terms are kept adjacent/grouped.
``commutator``   the contribution: order the terms by **greedy commuting-group
                 colouring of the commutator graph**, emitting one mutually-
                 commuting colour class after another. Within a class every
                 adjacency commutes (zero local Trotter error), so the schedule's
                 count of anticommuting adjacencies -- the source of the leading
                 :math:`[P_a,P_b]` error -- is minimised at no extra gate cost.
``learned``      a tiny scikit-learn classifier that reads the commutator-graph
                 features and predicts which of the above fixed strategies will
                 give the highest fidelity, then returns that strategy's ordering.

The two *reference* evolutions (``exact`` dense ``expm`` and ``krylov``
``expm_multiply``) are not orderings; they live in ``env.reference_state`` and
define the fidelity target.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from . import commutator_graph as cg
from . import pauli
from .hamiltonians import Hamiltonian

Ordering = List[int]

# Fixed (non-learned) strategies the learned policy chooses among.
FIXED_STRATEGIES = ("random", "coefficient", "locality", "commutator")


def random_ordering(ham: Hamiltonian, rng: np.random.Generator) -> Ordering:
    """A seeded random permutation of the term indices (the baseline)."""
    idx = np.arange(len(ham))
    rng.shuffle(idx)
    return idx.tolist()


def coefficient_ordering(ham: Hamiltonian) -> Ordering:
    """Terms by :math:`|c_j|` descending (ties broken by index for determinism)."""
    return sorted(range(len(ham)), key=lambda j: (-abs(ham.coeffs[j]), j))


def locality_ordering(ham: Hamiltonian) -> Ordering:
    """Terms by qubit support (lexicographic), then weight -- spatially local
    terms, which tend to share qubits and anticommute, are kept together."""
    paulis = ham.paulis
    return sorted(range(len(ham)),
                  key=lambda j: (pauli.support(paulis[j]), pauli.weight(paulis[j]), j))


def commutator_ordering(ham: Hamiltonian) -> Ordering:
    r"""Greedy commutator-graph ordering (the contribution).

    The leading per-pair Trotter error between two terms scales with
    :math:`|c_a|\,|c_b|\,\lVert[P_a,P_b]\rVert`, which is **zero whenever the
    terms commute** -- i.e. whenever they are *non-adjacent* in the commutator
    graph. We therefore build the schedule as a greedy walk that, from the
    largest-coefficient term, repeatedly appends the unused term that

      1. **commutes** with the current tail if possible (no graph edge -> no
         first-order error at that junction); failing that, the anticommuting
         term with the smallest coefficient product :math:`|c_{\mathrm{tail}}|
         |c_j|` (cheapest unavoidable commutator);
      2. among equals, the one sharing the most qubit support with the tail
         (keeps spatially-local, often-commuting terms clustered) and then the
         larger coefficient.

    This directly minimises the weighted count of anticommuting adjacencies in
    the Trotter product -- the schedule-level surrogate for the error -- at no
    change to the gate proxy. ``greedy_color_groups`` in ``commutator_graph.py``
    is the dual (commuting-group) view of the same object.
    """
    paulis = ham.paulis
    coeffs = [abs(c) for c in ham.coeffs]
    m = len(ham)
    if m == 0:
        return []
    remaining = set(range(m))
    start = max(range(m), key=lambda j: (coeffs[j], -j))
    order: Ordering = [start]
    remaining.discard(start)
    while remaining:
        tail = order[-1]
        tail_support = set(pauli.support(paulis[tail]))

        def cost(j: int):
            anti = pauli.anticommute(paulis[tail], paulis[j])
            weighted = coeffs[tail] * coeffs[j] if anti else 0.0
            shared = len(tail_support & set(pauli.support(paulis[j])))
            return (1 if anti else 0, weighted, -shared, -coeffs[j], j)

        nxt = min(remaining, key=cost)
        order.append(nxt)
        remaining.discard(nxt)
    return order


def adjacent_anticommutations(ham: Hamiltonian, ordering: Sequence[int]) -> int:
    """Count consecutive pairs in ``ordering`` whose Pauli terms anticommute.

    This is the schedule-level surrogate the commutator policy minimises; a lower
    value correlates with lower first-order Trotter error.
    """
    paulis = ham.paulis
    return sum(
        1 for a, b in zip(ordering, ordering[1:])
        if pauli.anticommute(paulis[a], paulis[b])
    )


def get_ordering(strategy: str, ham: Hamiltonian,
                 rng: np.random.Generator) -> Ordering:
    """Dispatch a fixed-strategy name to its ordering."""
    if strategy == "random":
        return random_ordering(ham, rng)
    if strategy == "coefficient":
        return coefficient_ordering(ham)
    if strategy == "locality":
        return locality_ordering(ham)
    if strategy == "commutator":
        return commutator_ordering(ham)
    raise ValueError(f"unknown strategy {strategy!r}")


# ---------------------------------------------------------------------------
# Learned schedule router: graph features -> best fixed *schedule*
# ---------------------------------------------------------------------------
# Ten listing-order-invariant features of the commutator graph. The first eight
# are the original graph invariants; the last two summarise the commutator
# *error form* (how much of the leading error is sign-/ordering-reducible), which
# is what discriminates when a folded schedule pays off.
FEATURE_KEYS = (
    "n_terms", "anticommutation_density", "mean_degree", "max_degree",
    "n_color_groups", "largest_group_frac", "coeff_cv", "mean_weight",
    "colliding_frac", "irreducible_frac",
)


def feature_vector(ham: Hamiltonian) -> np.ndarray:
    """Concatenate the graph invariants with two commutator-error-form features.

    All ten are functions of the unordered weighted graph (listing-order
    invariant), so the router's decision does not depend on how terms are listed.
    """
    from . import error_form
    feats = cg.features(ham)
    cs = error_form.collision_stats(ham)
    n_edges = max(cs["n_anti_pairs"], 1.0)
    feats["colliding_frac"] = round(cs["colliding_pairs"] / n_edges, 6)
    feats["irreducible_frac"] = cs["irreducible_frac"]
    return np.array([feats[k] for k in FEATURE_KEYS], dtype=float)


class LearnedSchedulePolicy:
    """Predicts the gate-optimal *schedule* from commutator-graph features.

    Trained on (features, best-schedule) pairs where the label is whichever fixed
    schedule reached the target accuracy at the fewest rotations on a training
    Hamiltonian. At inference it returns the predicted schedule *name* -- a
    learned router over the interpretable schedules, not a black box. Falls back
    to ``symmetric`` when unfitted or single-class.
    """

    def __init__(self, fallback: str = "symmetric") -> None:
        self._clf = None
        self._fallback = fallback

    def fit(self, hams: Sequence[Hamiltonian], best_labels: Sequence[str]
            ) -> "LearnedSchedulePolicy":
        X = np.array([feature_vector(h) for h in hams], dtype=float)
        y = np.asarray(best_labels)
        if len(set(y.tolist())) < 2:
            self._fallback = y[0] if len(y) else self._fallback
            self._clf = None
            return self
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        self._clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        ).fit(X, y)
        return self

    def predict_schedule(self, ham: Hamiltonian) -> str:
        if self._clf is None:
            return self._fallback
        x = feature_vector(ham).reshape(1, -1)
        return str(self._clf.predict(x)[0])


# Backward-compatible alias: the router over fixed *orderings* (selects an
# ordering name and returns that ordering). Retained for the ordering-impotence
# comparison and the legacy ordering tests.
class LearnedOrderingPolicy(LearnedSchedulePolicy):
    def __init__(self) -> None:
        super().__init__(fallback="commutator")

    def predict_strategy(self, ham: Hamiltonian) -> str:
        return self.predict_schedule(ham)

    def ordering(self, ham: Hamiltonian, rng: np.random.Generator) -> Ordering:
        name = self.predict_schedule(ham)
        if name not in FIXED_STRATEGIES:
            name = "commutator"
        return get_ordering(name, ham, rng)


# All comparable orderings (references are handled separately in the runner).
ORDERINGS = ("random", "coefficient", "locality", "commutator", "learned")
