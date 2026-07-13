"""The commutator graph of a Hamiltonian's terms.

Nodes are Hamiltonian terms; an **edge connects two terms iff they
anticommute** (do not commute). This graph is the topological object the
ordering policies reason over:

* A first-order Trotter step :math:`\\prod_j e^{-i c_j P_j t/r}` is *exact* for
  any pair of terms that commute -- the Trotter error comes entirely from
  **anticommuting (adjacent-in-graph) pairs**, whose leading term is the nested
  commutator :math:`[P_a, P_b]` (Childs et al., 2021). Re-ordering the product
  to reduce the number of anticommuting *adjacencies in the schedule* shrinks
  that error at zero extra gate cost.
* A proper graph **colouring** partitions the terms into mutually-commuting
  groups; each colour class can be exponentiated exactly and simultaneously,
  the classic "commuting-group" Trotter optimisation.

The features exposed here (anticommutation density, degree statistics, greedy
colouring) are exactly the inputs the learned ordering policy regresses on.
"""
from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np

from . import pauli
from .hamiltonians import Hamiltonian


def build_graph(ham: Hamiltonian) -> nx.Graph:
    """Graph over term indices with an edge for every anticommuting pair."""
    g = nx.Graph()
    paulis = ham.paulis
    for i, p in enumerate(paulis):
        g.add_node(i, pauli=p, coeff=ham.coeffs[i],
                   weight=pauli.weight(p), support=pauli.support(p))
    for i in range(len(paulis)):
        for j in range(i + 1, len(paulis)):
            if pauli.anticommute(paulis[i], paulis[j]):
                g.add_edge(i, j)
    return g


def anticommutation_density(ham: Hamiltonian) -> float:
    """Fraction of term pairs that anticommute (graph edge density in [0,1])."""
    m = len(ham)
    if m < 2:
        return 0.0
    g = build_graph(ham)
    return 2.0 * g.number_of_edges() / (m * (m - 1))


def greedy_color_groups(ham: Hamiltonian) -> List[List[int]]:
    """Greedy colouring of the commutator graph -> mutually-commuting groups.

    Two terms with the same colour are non-adjacent, hence commute, hence can be
    exponentiated together exactly. Returns the colour classes as lists of term
    indices, ordered by colour id.
    """
    g = build_graph(ham)
    coloring = nx.algorithms.coloring.greedy_color(g, strategy="largest_first")
    groups: Dict[int, List[int]] = {}
    for node, col in coloring.items():
        groups.setdefault(col, []).append(node)
    # include any isolated nodes networkx may have skipped (none here, but safe)
    for node in g.nodes:
        if node not in coloring:
            groups.setdefault(-1, []).append(node)
    return [sorted(groups[c]) for c in sorted(groups)]


def features(ham: Hamiltonian) -> Dict[str, float]:
    """Scalar commutator-graph features used by the learned ordering policy.

    All are O(m^2) graph invariants of the Hamiltonian's term set -- independent
    of how the terms happen to be listed, so the policy's decision is
    listing-order invariant by construction.
    """
    g = build_graph(ham)
    m = len(ham)
    degs = np.array([d for _, d in g.degree()], dtype=float) if m else np.zeros(1)
    groups = greedy_color_groups(ham) if m else [[]]
    coeffs = np.abs(np.array(ham.coeffs)) if m else np.zeros(1)
    weights = np.array([pauli.weight(p) for p in ham.paulis]) if m else np.zeros(1)
    return {
        "n_terms": float(m),
        "anticommutation_density": round(anticommutation_density(ham), 6),
        "mean_degree": round(float(degs.mean()), 6),
        "max_degree": float(degs.max()),
        "n_color_groups": float(len(groups)),
        "largest_group_frac": round(float(max(len(c) for c in groups)) / max(1, m), 6),
        "coeff_cv": round(float(coeffs.std() / (coeffs.mean() + 1e-12)), 6),
        "mean_weight": round(float(weights.mean()), 6),
    }
