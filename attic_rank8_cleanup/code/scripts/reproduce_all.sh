#!/usr/bin/env bash
# One-command reproduction. Pass "full" for the reported-scale configuration.
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE   # pip-torch + conda can both ship libomp on macOS

CFG="configs/smoke.yaml"
if [[ "${1:-}" == "full" ]]; then CFG="configs/full.yaml"; fi

echo "==> config: $CFG"
python scripts/run.py --config "$CFG" --out results
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/audit_claims.py
echo "==> done. See results/summary.json, results/*.tex, figures/*.pdf"
