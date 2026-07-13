# topoham — code artifact

CPU reference implementation for **Commutator-Graph Term Ordering for First-Order
Trotterization**. First-order Trotter evolution is computed with an exact numpy statevector
simulator built from single-Pauli rotations (`cos θ · ψ − i sin θ · Pψ`) and validated
against dense `scipy.linalg.expm` and Krylov `scipy.sparse.linalg.expm_multiply`
references. The ordering policies use networkx (commutator graph) and scikit-learn (the
learned router); an optional PyTorch backend is declared under the `gpu` extra. The
implementation relies on the standard scientific-Python stack alone.

## Install

The package installs from a checkout of this directory; the `dev` extra adds pytest and the
build backend.

```bash
pip install -e ".[dev]"
```

A conda environment that pins the same CPU stack is provided as an alternative:

```bash
conda env create -f environment.yml && conda activate topoham
```

## Reproduce

The installed console entry point regenerates every artifact from one seeded run. Invoke it
from this `code/` directory:

```bash
topoham-reproduce --config configs/full.yaml   # reported scale (120 instances, ~minutes)
topoham-reproduce --config configs/smoke.yaml   # smoke scale (18 instances, seconds)
```

The command runs the experiment (`scripts/run.py` → `results/summary.json`), renders the
tables (`scripts/make_tables.py`), draws the figures (`scripts/make_figures.py`), and
applies the readiness-gate audit (`scripts/audit_claims.py`). The `--skip-run` flag rebuilds
tables and figures from an existing `results/summary.json`; `--skip-audit` omits the final
gate. The equivalent Makefile targets:

```bash
make test        # Pauli-rotation ↔ expm cross-check, Trotter convergence, core claim
make demo        # smoke config (seconds) -> results/summary.json
make tables      # results/main_results.tex
make figures     # figures/fig_frontier.pdf, figures/fig_family.pdf
make audit       # readiness gate
make full-run    # reported-scale config (minutes)
# or, one command:
bash scripts/reproduce_all.sh         # smoke
bash scripts/reproduce_all.sh full    # reported scale
```

> macOS: the Makefile and scripts export `KMP_DUPLICATE_LIB_OK=TRUE` because conda and
> pip-PyTorch can each bundle an OpenMP runtime; the setting averts the duplicate-libomp abort
> and leaves results unchanged.

## Figures and tables regenerated

| Artifact | Producer | Content |
|---|---|---|
| `results/summary.json` | `scripts/run.py` | Authoritative artifact: per-ordering fidelity (mean, std, 95% CI), infidelity, Z₀ observable error, anticommuting-adjacency count, the fidelity-vs-gate-budget frontier, the per-family breakdown, and the `pauli_algebra_verified` integrity flag. |
| `results/main_results.tex` | `scripts/make_tables.py` | LaTeX table of the main per-ordering metrics at the matched gate proxy. |
| `figures/fig_frontier.pdf` | `scripts/make_figures.py` | Fidelity versus gate budget across `r ∈ {1,2,3,4,6,8,12}` for each ordering. |
| `figures/fig_family.pdf` | `scripts/make_figures.py` | Per-family fidelity bars (TFIM, Heisenberg, molecular-like) with 95% confidence intervals. |

Each table, figure, and macro is derived from `results/summary.json`, so every reported
number traces to one run.

## Determinism

The pipeline is seeded from the configuration (`seed: 0` in both `configs/smoke.yaml` and
`configs/full.yaml`) through `topoham.seed`, which fixes the numpy and Python RNG state
before instance generation. Numeric results are therefore reproduced exactly across runs on
a fixed dependency set; only the provenance fields (timestamp, runtime, peak memory) vary.
Thread counts are pinned (`OMP_NUM_THREADS=1`) so the dense `expm` reference stays
bitwise-stable. The smoke configuration yields commutator fidelity `0.7201` against random
`0.4435`; the reported configuration yields commutator fidelity `0.5481` against random
`0.2054`.

## Pinned dependencies

Runtime dependencies and their floors are declared in `pyproject.toml`:

| Package | Constraint |
|---|---|
| numpy | `>=1.24` |
| scipy | `>=1.10` |
| networkx | `>=3.0` |
| scikit-learn | `>=1.2` |
| matplotlib | `>=3.6` |
| pyyaml | `>=6.0` |

The `dev` extra adds `pytest>=7.0` and `build>=1.0`; the `gpu` extra adds `torch>=2.0`. The
provenance block in `results/summary.json` records the exact versions used for the committed
run (Python 3.13.12, numpy 2.4.3, scipy 1.17.1, networkx 3.6.1, scikit-learn 1.8.0,
matplotlib 3.10.8). `environment.yml` mirrors the CPU stack for conda.

## Layout

```
src/topoham/
  pauli.py             Pauli strings · commute()/anticommute() · fast e^{-iθP} rotation
  hamiltonians.py      TFIM · Heisenberg · molecular_like (random local 2-body) families
  commutator_graph.py  anticommutation graph · greedy commuting-group colouring · features
  env.py               first-order Trotter statevector + exact expm / Krylov references
  policies.py          random / coefficient / locality / commutator / learned orderings
  metrics.py           fidelity · observable error · gate proxy · summaries
  runner.py            end-to-end protocol -> results/summary.json
  reproduce.py         topoham-reproduce console entry point
  audit.py             forbidden-claims + traceable-number checks
  plotting.py          fidelity-vs-gate-budget frontier + per-family figures
  config.py · seed.py  YAML configuration loading · deterministic seeding
scripts/   run.py · make_tables.py · make_figures.py · audit_claims.py · reproduce_all.sh
configs/   smoke.yaml (demo) · full.yaml (reported)
tests/     pauli↔expm · Trotter convergence · commutator-beats-random · metrics bounds
```

## What is computed

`results/summary.json` holds, per term-ordering policy at a **matched gate budget**
(`num_terms × Trotter steps`): the mean Trotter fidelity with 95% confidence intervals, mean
infidelity, mean Z₀ observable error, mean number of anticommuting adjacencies, the
fidelity-vs-gate-budget frontier curve, and the per-family breakdown. The integrity flag
`pauli_algebra_verified` records that the fast Pauli-rotation kernel matched dense `expm`
before any fidelity number was produced. It is the authoritative artifact for every table,
figure, and macro.

All experiments are reproducible on commodity hardware; runtime and memory are reported for each benchmark.
