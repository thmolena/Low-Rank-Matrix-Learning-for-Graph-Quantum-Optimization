# SpecOps-CGS — Commutator-Graph Scheduling for Trotter Simulation

Reproducibility package for the manuscript

> **Zero-Overhead Cancellation of the Leading Trotter Error via Commutator-Graph Scheduling**
> Molena Huynh, North Carolina State University (2026).
> Part of the **spectral-truncation operators (SpecOps)** program.

pip distribution name: `specops-cgs`  ·  Python import name: `topoham` (historical)  ·  BibTeX key: `huynh2026topoham`

---

## Summary

For a Hamiltonian written in Pauli form `H = Σ_j c_j P_j`, one Lie–Trotter (first-order
product-formula) step costs exactly one rotation per term, so the total gate count `L·r`
is invariant under the term ordering `π`. This package builds the **commutator error form** —
the exact leading-order Trotter error operator `E_π = Σ_{a<b} [h_{π(a)}, h_{π(b)}]`, assembled
matrix-free as a signed sum of Pauli strings over the commutator graph and validated against
dense matrix exponentials to `1e-9` — and uses it to establish two facts and act on them:
reordering only *re-signs* `E_π` and cannot escape the `O(t²/r)` rate (ordering is impotent),
whereas a per-step **antithetic schedule** cancels `E_π` exactly and converges at `O(t³/r²)` at
the *identical* `L·r` rotation count. An interpretable learned router over ten commutator-graph
features selects the gate-optimal schedule per instance. One seeded command regenerates every
figure, table, and quoted number in the paper.

---

## Background and problem setting

Simulating quantum time evolution `e^{-iHt}` is a founding application of quantum computers and
the workhorse subroutine of digital quantum chemistry, condensed-matter, and materials
simulation. On near-term hardware the dominant primitive is the **product formula**: the exact
propagator `e^{-iHt}` is approximated by a product of easy single-term exponentials
`e^{-i c_j P_j t/r}`, repeated over `r` Trotter steps. The first-order (Lie–Trotter) approximant is

```
U_Trot(π) = ( Π_{j∈π} e^{-i c_j P_j t/r} )^r .
```

Two facts frame the problem an outsider needs. First, the *accuracy* of this approximation is
controlled by the **commutators** of the summands: the leading per-step error is
`(t²/2r) Σ_{a<b} [c_a P_a, c_b P_b]`, and it vanishes for any pair of terms that commute.
Second, the *cost* is one rotation per term per step — `L·r` rotations — **independently** of the
ordering `π`. The ordering is therefore a genuinely free knob. A large empirical literature has
asked how much accuracy that knob buys, and has repeatedly found the effect marginal, "within
noise." Why a free parameter should be so impotent had not been explained. The natural bookkeeping
object is the **commutator graph** `G(H)`: one node per Pauli term, one edge per anticommuting
pair. It is the same graph used to group commuting observables in measurement-reduction schemes
for variational algorithms. This package makes the graph the carrier of an exact error operator,
explains the impotence rigorously, and then removes it at no cost.

---

## Contributions

1. **The commutator error form (matrix-free, certified).** The exact leading-order Trotter error
   operator `E_π = Σ_{a<b} [h_{π(a)}, h_{π(b)}]` (with `h_j = c_j P_j`) is assembled in `O(L²n)`
   time as a signed sum of Pauli strings indexed by the edges of the commutator graph, without ever
   forming a `2ⁿ×2ⁿ` matrix, and validated against the dense pair-commutator sum to `1e-9`.
2. **An ordering-impotence theorem.** Reordering acts on `E_π` only by flipping edge signs; the
   Hilbert–Schmidt norm of `E_π` is *exactly* ordering-invariant on collision-free families, and in
   general only a small "colliding" fraction is ordering-reducible. No fixed ordering escapes the
   `O(t²/r)` rate — the precise content of the "ordering is within noise" folklore.
3. **A free second-order schedule.** The step-alternating **antithetic** schedule (apply an order
   `π` on even steps and its reverse on odd steps) makes each step-pair a symmetric product formula,
   cancels `E_π`, and converges at `O(t³/r²)` at the identical `L·r` rotation budget
   (Proposition: rotation-count invariance; Theorem: antithetic cancellation).
4. **Commuting-clique compression.** A proper coloring of `G(H)` partitions terms into commuting
   cliques whose rotations factor exactly, so schedule error depends only on inter-clique
   commutators; low-chromatic Hamiltonians are near-exact already at first order.
5. **A quantitative theory of the residual.** A self-contained Baker–Campbell–Hausdorff / Magnus
   remainder bounds the post-cancellation error in operator norm; leading-cancellation is proved
   strictly more accurate than every fixed ordering at fine step size; residual minimization is
   placed in an NP-hard combinatorial landscape with a spectral relaxation; and an unbiased
   shot-based estimator with a finite-sample confidence interval makes the accuracy claim
   empirically *certifiable* rather than merely asymptotic.
6. **An interpretable learned scheduler.** A standardized logistic-regression router over ten
   listing-order-invariant commutator-graph features selects the gate-optimal schedule per instance,
   trained leave-one-instance-out; its prediction is provably invariant to how the terms are listed.

---

## Method

Each Hamiltonian instance is decomposed into Pauli terms; the commutator graph places one node per
term and one edge per anticommuting pair (an `O(n)` parity test per pair), and a greedy coloring
partitions terms into commuting cliques. A **schedule** fixes, for every Trotter step, a term order
and a fold — first-order, or the step-alternating antithetic fold — while holding the realized
rotation count fixed across schedules. The product formula is executed by an **exact, matrix-free
statevector simulator** built on the identity `e^{-iθP} = cosθ·I − i sinθ·P`, and scored both by
fidelity against `e^{-iHt}` and by the rotations needed to reach a target fidelity. No number is
reported until three integrity gates pass to `1e-9`: (i) the fast Pauli-rotation kernel versus dense
`scipy.linalg.expm`; (ii) the matrix-free error form versus the dense pair-commutator sum; and
(iii) a matrix-free Krylov reference versus dense `expm`, which extends the exact benchmark beyond
the dense-exponential ceiling to larger qubit counts.

---

## Main results

All numbers are generated from a single seeded run and transcribed from `results/summary.json`
(they are not entered by hand). The full configuration sweeps **144 structured instances** across
three families (transverse-field Ising, Heisenberg, random molecular-like) and six sizes
`n ∈ {4, 6, 8, 10, 12, 14}` qubits, eight seeded draws each, at evolution time `t = 1.0`.

- **Convergence-order doubling.** The fitted infidelity-vs-`r` slope rises from **1.96** for every
  first-order ordering to **4.07** for the antithetic schedule (**4.03** for the symmetric Strang
  reference) — a clean doubling at no change in gate count.
- **Matched-cost snapshot** at `r = 6` steps (**118** mean rotations): the antithetic schedule
  reaches mean fidelity **0.9589** against **0.9403** for the best first-order ordering, reducing
  mean infidelity from **0.0597** to **0.0411** (a factor of **1.5**), for free.
- **Rotations to a target fidelity.** To reach `F ≥ 0.99` the best first-order ordering needs
  **315** rotations, the antithetic schedule **236** and the symmetric schedule **225** — a
  **1.40×** reduction. To reach `F ≥ 0.999` no first-order schedule succeeds within the swept
  budget, while the antithetic and symmetric schedules reach it at **315** and **300** rotations.
- **Ordering impotence, measured.** A median ordering sits within a factor **1.097** of the exact
  leading-error floor and the coefficient ordering within **1.074**; the matrix-free Hilbert–Schmidt
  surrogate moves within **1.004**. The ordering-invariant fraction is **1.00** on the
  transverse-field Ising and Heisenberg families and **0.98** on the dense molecular-like family.
- **Size independence via Krylov.** A matrix-free Krylov reference extends the exact benchmark to
  `n = 14` qubits; the order doubling (first-order rate ≈ 2, antithetic rate ≈ 4) is size-independent
  and the matched-cost speedup reaches **2.06×** at `n = 14`.
- **Learned scheduler.** The interpretable router attains **93%** leave-one-instance-out accuracy
  against the per-instance oracle and recovers near-oracle efficiency: on the **92** instances where
  every fixed schedule reaches the target it needs **114.0** rotations on average against the
  oracle's **112.4**.

---

## Significance

For first-order product formulas the schedule is a genuinely free lever: it changes no gate, qubit,
or depth count. Converting an `O(t²/r)` method into an `O(t³/r²)` method at fixed cost is the
difference between reaching and not reaching a target accuracy, as the unreachable `0.999` column
makes concrete. The commutator graph supplies both halves of the recommendation at no additional
cost: it certifies when ordering cannot help, and its clique coloring tells the learned router which
instances to fold and which to leave alone. The palindromic symmetrization mechanism is classical
(Suzuki–Hatano); the contribution is the framing — that the free object is the *schedule*, that a
fixed ordering provably cannot realize it — together with the matrix-free certificate, the
matched-cost quantification, and the interpretable selector.

**Boundaries.** The second-order advantage is a small-step-size effect with an honest crossover;
systems are bounded (`n ≤ 14`) because an exact reference must be formed, though the convergence-rate
argument is size-independent; families are synthetic/structured proxies rather than production-mapped
molecular Hamiltonians; cost is the realized rotation count, ignoring rotation synthesis and
connectivity; and all results are classical CPU statevector simulations with no hardware validation.

---

## Installation and reproduction

From this directory (`submission/code`):

```bash
pip install .
# or, for development:
pip install -e .
# with the optional learned-GPU / test extras:
pip install ".[gpu,dev]"
```

Requires Python ≥ 3.9. Core dependencies: numpy, scipy, networkx, scikit-learn, matplotlib, pyyaml.
On macOS with mixed conda + pip OpenMP runtimes, export `KMP_DUPLICATE_LIB_OK=TRUE` (the reproduce
entrypoint sets this for its subprocesses automatically).

The console entrypoint runs the full pipeline (experiment → tables → figures) and syncs artifacts to
the manuscript:

```bash
cgs-reproduce                               # full config (configs/full.yaml)
cgs-reproduce --quick                       # fast demo, a few seconds (= --config configs/smoke.yaml)
cgs-reproduce --skip-run                    # rebuild tables/figures from an existing run
```

**Expected runtime.** The full configuration completes in **2–4 minutes** wall-clock on a
commodity laptop CPU (the archived run: 209 s, ~500 MB peak memory; a re-verification run on the
same class of machine: 119 s), single-threaded. `--quick` completes in a few seconds; its numbers
are demo-scale and are *not* the ones quoted in the paper.

This writes:

* `results/summary.json` — every number, with provenance (seed, platform, runtime, peak memory);
* `results/macros.tex` — a `\newcommand` for each number the manuscript prose quotes;
* `results/tab_*.tex` — the LaTeX table fragments the manuscript `\input`s;
* `figures/fig_*.pdf` — the manuscript figures;

and copies the figures into `submission/figures/` and the macros/tables into
`submission/tables/`, which is exactly where the manuscript reads them
(`\graphicspath{{figures/}}`, `\input{tables/...}`), so **no number is entered by hand**.

Equivalent Make targets: `make full-run && make tables && make figures` (and `make demo`, `make
test`, `make audit`).

### Figure and table provenance map

Every figure and table in the manuscript is generated by a named function reading
`results/summary.json` only:

| Manuscript artifact | File | Generating function |
|---|---|---|
| Fig. 1 (pipeline schematic) | `figures/fig_schematic.pdf` | `topoham.plotting.fig_schematic` |
| Fig. 2 (ordering impotence bars) | `figures/fig_impotence.pdf` | `topoham.plotting.fig_impotence` |
| Fig. 3 (infidelity vs. steps, log–log) | `figures/fig_convergence.pdf` | `topoham.plotting.fig_convergence` |
| Fig. 4 (rotations-to-target frontier) | `figures/fig_frontier.pdf` | `topoham.plotting.fig_frontier` |
| Fig. 5 (size-scaling panels, Krylov) | `figures/fig_scaling.pdf` | `topoham.plotting.fig_scaling` |
| Fig. 6 (per-family gate budgets) | `figures/fig_family.pdf` | `topoham.plotting.fig_family_gates` |
| Table: impotence / collisions | `tables/tab_impotence.tex` | `scripts/make_tables.py: tab_impotence` |
| Table: matched-cost snapshot | `tables/tab_matched.tex` | `scripts/make_tables.py: tab_matched` |
| Table: gates to target | `tables/tab_gates.tex` | `scripts/make_tables.py: tab_gates` |
| Table: size scaling | `tables/tab_scaling.tex` | `scripts/make_tables.py: tab_scaling` |
| Table: per-family results | `tables/tab_family.tex` | `scripts/make_tables.py: tab_family` |
| All prose numbers | `tables/macros.tex` | `scripts/make_tables.py: macros` |

---

## Extend / tweak

All experiment parameters live in a YAML config (`configs/full.yaml`, `configs/smoke.yaml`) loaded
into `topoham.config.Config`. Copy a config, edit fields, and run
`cgs-reproduce --config configs/my.yaml`. Every field:

| field               | type        | meaning |
|---------------------|-------------|---------|
| `name`              | str         | run label, stored in the summary provenance |
| `seed`              | int         | master RNG seed (fully determines instances) |
| `families`          | list[str]   | Hamiltonian families to sweep: `tfim`, `heisenberg`, `molecular_like` |
| `sizes`             | list[int]   | qubit counts `n` to sweep |
| `n_per_family`      | int         | number of seeded instances per (family, size) |
| `time`              | float       | evolution time `t` |
| `steps`             | int         | reference (matched-budget) Trotter step count `r` |
| `steps_grid`        | list[int]   | step counts for the gate-budget / convergence frontier |
| `schedules`         | list[str]   | schedules compared: `random`, `coefficient`, `commutator`, `antithetic`, `symmetric`, `learned` |
| `targets`           | list[float] | target fidelities for the gates-to-target tables |
| `target_ref`        | float       | target fidelity the learned router optimizes for |
| `impotence_samples` | int         | random orderings per instance for the impotence sweep (Theorem 1) |
| `impotence_max_n`   | int         | cap `n` for the exact dense spectral-norm impotence sweep; larger `n` use the matrix-free HS surrogate |
| `reference_backend` | str         | exact reference: `expm` (dense) or `krylov` (matrix-free, scales to larger `n`) |

Any field omitted from the YAML falls back to the dataclass default in `src/topoham/config.py`.

### Add a new Hamiltonian family

Add a builder in `src/topoham/hamiltonians.py` returning the Pauli-term list
`[(coeff, pauli_string), ...]`, register its name in the family dispatch there, then list the name
under `families:` in your config.

### Add a new schedule

Implement it in `src/topoham/schedules.py` (map an ordering + step index to the per-step term order /
sign pattern), add its name to `schedules:` in the config, and add a display label in
`scripts/make_tables.py` (the `DISP` dict) so it appears in the tables.

### Add a new config parameter

Add a field to the `Config` dataclass in `src/topoham/config.py` (with a default), read it wherever
it is used in `src/topoham/runner.py`, and set it in your YAML.

### Use the library in your own project

```python
from topoham.config import Config
from topoham.runner import run
from pathlib import Path

cfg = Config(families=["tfim"], sizes=[6], n_per_family=4, time=1.0, steps=8)
summary = run(cfg, Path("my_results"))
print(summary["headline"]["gate_speedup_vs_first_order"])
```

Lower-level building blocks are importable directly: `topoham.error_form` (the matrix-free
commutator error operator), `topoham.commutator_graph`, `topoham.hamiltonians`,
`topoham.schedules`, `topoham.policies` (learned router), and `topoham.metrics`.

---

## Cite this work

If you use this software or its results, please cite the paper:

```bibtex
@article{huynh2026topoham,
  title   = {Zero-Overhead Cancellation of the Leading Trotter Error via Commutator-Graph Scheduling},
  author  = {Huynh, Molena},
  year    = {2026},
  note    = {Part of the spectral-truncation operators (SpecOps) program},
}
```

See `CITATION.cff` for machine-readable citation metadata.

---

## Reproducibility

All numbers, tables, and figures are script-generated from a single seeded run;
`results/summary.json` records full provenance (seed, command, platform, package versions, runtime,
peak memory). All experiments run on commodity CPU hardware.
