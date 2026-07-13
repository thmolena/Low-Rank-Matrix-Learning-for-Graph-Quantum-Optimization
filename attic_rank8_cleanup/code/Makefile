# topoham -- reproducible build. macOS: export KMP_DUPLICATE_LIB_OK to avoid
# the duplicate-libomp abort when conda + pip both ship an OpenMP runtime.
export KMP_DUPLICATE_LIB_OK = TRUE
PY ?= python

.PHONY: install test demo full-run tables figures audit clean

install:
	$(PY) -m pip install .

test:
	$(PY) -m pytest -q

demo:
	$(PY) scripts/run.py --config configs/smoke.yaml --out results

full-run:
	$(PY) scripts/run.py --config configs/full.yaml --out results

tables:
	$(PY) scripts/make_tables.py

figures:
	$(PY) scripts/make_figures.py

audit:
	$(PY) scripts/audit_claims.py

clean:
	rm -rf results/summary.json results/*.tex results/*.md figures/*.pdf
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
