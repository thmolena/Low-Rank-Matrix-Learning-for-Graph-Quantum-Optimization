"""From-scratch foundations for commutator-graph Hamiltonian simulation."""

from __future__ import annotations

FOUNDATION_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "Hamiltonian simulation",
        "The exact evolution exp(-i H t) is approximated by a Trotter product "
        "over Pauli terms.  The code evaluates small systems exactly so it can "
        "measure fidelity and ordering effects without treating dense matrices "
        "as the scalable proposal.",
    ),
    (
        "Pauli algebra and the error form",
        "pauli.py implements Pauli strings, the O(n) commutation test, the exact "
        "single-Pauli rotation, and the phased product P Q = omega R.  error_form.py "
        "assembles the exact leading-order error operator E_pi as a signed sum of "
        "Pauli strings and validates it against the dense pair-commutator sum to 1e-9.",
    ),
    (
        "Ordering impotence",
        "Reordering only re-signs the commutators in E_pi, so its Hilbert-Schmidt "
        "norm is exactly invariant when the edge commutators are distinct.  No fixed "
        "ordering escapes the O(t^2/r) first-order rate -- the precise form of the "
        "'ordering is within noise' observation.",
    ),
    (
        "Schedules and baselines",
        "schedules.py implements first-order, antithetic (a free second-order fold "
        "that cancels E_pi at the identical L*r rotation count) and symmetric folds. "
        "policies.py adds a learned router over ten commutator-graph features that "
        "selects the gate-optimal schedule per instance.",
    ),
    (
        "Metrics",
        "metrics.py and runner.py compute fidelity, the rotations to reach a target "
        "fidelity (the matched-cost efficiency metric), the convergence rate, and "
        "95% confidence intervals across a seeded instance family.  Figures show the "
        "convergence and matched-cost frontiers and per-family breakdowns.",
    ),
    (
        "Audit discipline",
        "runner.py writes results/summary.json as the single source of truth.  "
        "make_tables.py and make_figures.py render only from that artifact, and "
        "audit_claims.py prevents unsupported superiority language.",
    ),
    (
        "Reproduction path",
        "Run scripts/reproduce_all.sh full from the code folder or use the "
        "topoham-reproduce entry point.  Dependencies are in pyproject.toml; "
        "no requirements text file is needed.",
    ),
)


def iter_foundations() -> tuple[tuple[str, str], ...]:
    return FOUNDATION_SECTIONS


def print_foundations() -> None:
    for index, (heading, body) in enumerate(FOUNDATION_SECTIONS, start=1):
        print(f"{index}. {heading}\n{body}\n")


if __name__ == "__main__":
    print_foundations()
