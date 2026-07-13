"""From-principles guide for commutator-graph Hamiltonian scheduling.

Hamiltonian simulation
----------------------
An n-qubit state is a vector in C**(2**n); exact dynamics are exp(-i H t). H is a
sum of Pauli strings with real coefficients, so dense matrices are avoided except
for small reference checks.

Pauli computation
-----------------
pauli.py implements three facts. Two Pauli strings commute iff they anticommute on
an even number of qubits (an O(n) parity test). Because P**2 = I, a single rotation
is exp(-i theta P) psi = cos(theta) psi - i sin(theta) P psi. And the product of two
Pauli strings is a phased Pauli, P Q = omega R -- the algebra behind the error form.

Commutator graph and the error form
------------------------------------
commutator_graph.py builds the graph (edge = anticommuting pair) and its commuting
cliques. error_form.py assembles the exact leading-order Trotter error operator
E_pi = sum_{a<b} [h_{pi(a)}, h_{pi(b)}] matrix-free as a signed sum of Pauli strings,
and validates it against the dense pair-commutator sum to 1e-9. Reordering only flips
the signs in E_pi, so a single ordering cannot reduce its norm below an instance floor
(ordering impotence).

Schedules and machine learning
------------------------------
schedules.py implements first-order, antithetic (a free second-order fold that
cancels E_pi at the identical L*r rotation count), and symmetric (Strang) folds over
coefficient or commuting-clique orderings. policies.py adds a small logistic-regression
router over ten commutator-graph features that selects the gate-optimal schedule per
instance.

Evaluation
----------
env.py scores each schedule against an exact reference (fidelity) and reports the
realised rotation count. runner.py aggregates over families, computes the rotations to
a target fidelity and the convergence rate, and writes results/summary.json, from which
the table and figure scripts derive every manuscript artifact.

Reproduction
------------
From submission/code:

    pip install .
    make test
    topoham-reproduce --config configs/full.yaml
"""


GUIDE = __doc__


def main() -> None:
    print(GUIDE)


if __name__ == "__main__":
    main()
