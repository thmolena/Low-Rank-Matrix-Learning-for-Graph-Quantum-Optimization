# Source-Disjoint Audit of Four-Anchor Residual Interpolation

This repository contains a reproducible conditional/negative study of a
four-query surrogate for finite-grid, depth-two MaxCut QAOA.

The candidate algorithm subtracts the established exact depth-one expectation,
selects four residual columns by pivoted QR, and transfers the training
residual nearest at those anchors. The experiment uses 16 checksum-pinned
Stanford Gset files: six sources for training, eight for development, and two for final
confirmation, with eight derived ten-vertex tasks per source.

On the 16 G67/G77 confirmation tasks, anchor-gated transfer has lower surface
RMSE (`0.006095`) than the zero-depth-two-query nearest-depth-one rule
(`0.011280`) and a four-query linear rule (`0.011811`). However, all three make
all 16 grid decisions correctly. The paper therefore does not claim that four
queries caused confirmation decision success or that the QR/nearest-transfer
ingredients are novel. A development preference for salience temperature 2
was only a `6e-18` floating-point tie artifact; the frozen method uses uniform
QR (`tau=0`).

## Reproduce

```bash
python -m pip install ./code
qaoa-anchor-reproduce --cache-dir /path/to/gset-cache --verify
python -m pytest -q code/tests
python code/scripts/validate_release.py
```

Regenerate the host-specific timing record explicitly, then regenerate the
four paper figures:

```bash
qaoa-anchor-benchmark --cache-dir /path/to/gset-cache
qaoa-anchor-figures
```

## Evidence boundary

- Real external inputs: 16 authenticated Stanford Gset source files.
- Computed responses: noiseless state-vector expectations, not hardware data.
- Scope: 128 derived ten-vertex tasks, depth two, fixed 256-point grid.
- Confirmation: only two source files (16 dependent derived tasks).
- No claims of quantum advantage, global MaxCut solution quality, continuous
  parameter optimality, general low-rank novelty, or hardware speedup.

The manuscript is [main.pdf](main.pdf). Code, locked evidence, source hashes,
and generated figures are under `code/`.
