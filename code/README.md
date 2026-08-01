# qaoa-anchor-learning reproduction package

CPU-only implementation for **A Source-Disjoint Audit of QR-Anchor-Gated
Residual Transfer for Depth-Two QAOA**.

## Commands

```bash
python -m pip install .
qaoa-anchor-reproduce --cache-dir /path/to/gset-cache --verify
python -m pytest -q tests
qaoa-anchor-benchmark --cache-dir /path/to/gset-cache
qaoa-anchor-figures
python scripts/validate_release.py
```

`--verify` recomputes all deterministic landscapes and comparisons, checks
them numerically against `results/locked_results.json`, and does not overwrite
the locked record. The benchmark is separate because wall time is
host-specific.

## Scientific design

- `src/hqml_trotter_scheduling/experiment.py`: authenticated Gset parser,
  matrix-free state-vector kernel, established exact depth-one formula,
  QR-anchor-gated transfer, linear and zero-query baselines, split metrics, and
  locked replay.
- `src/hqml_trotter_scheduling/benchmark.py`: seven-repeat single-host timing
  on one authenticated G67-derived confirmation task.
- `src/hqml_trotter_scheduling/figures.py`: exactly four quantitative figures.
- `tests/test_reproduction.py`: formula, interpolation, regret, provenance,
  split, and falsifying-baseline tests.
- `results/`: locked JSON, CSV evidence, runtime record, and hash manifest.

The central boundary is deliberate: anchor gating lowers confirmation surface
RMSE to `0.006095`, but the zero-query nearest-depth-one residual baseline
matches its 16/16 confirmation decisions. The package does not claim a new
DEIM, CUR, pivoted-QR, nearest-neighbor, or landscape-completion method.
