"""Installed reproduction command for the topology-guided Hamiltonian artifact.

Exposes ``topoham-reproduce`` (see ``pyproject.toml`` console scripts): one command
that runs the full pipeline behind every reported number -- the experiment runner
(``scripts/run.py`` -> ``results/summary.json``), table and figure generation
(``make_tables.py``, ``make_figures.py``), and the readiness-gate audit
(``audit_claims.py``). It then syncs the regenerated table and figure artifacts into
the ``submission/`` folder used to build the manuscript, so the paper's numbers,
tables, and figures all trace to one run (Methods, "Reproducibility, software, and
provenance").
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _code_dir() -> Path:
    """The submission/code directory holding scripts/ and configs/.

    Prefer the current working directory (so ``cd submission/code &&
    topoham-reproduce`` works for a non-editable ``pip install .``), falling back
    to the packaged dev-tree location for an editable install.
    """
    cwd = Path.cwd()
    if (cwd / "scripts" / "run.py").exists():
        return cwd
    cand = Path(__file__).resolve().parents[2]
    if (cand / "scripts" / "run.py").exists():
        return cand
    raise SystemExit(
        "topoham-reproduce must be run from the submission/code directory "
        "(it needs scripts/ and configs/).")


def _run(*args: str) -> None:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run([sys.executable, *args], cwd=_code_dir(), env=env, check=True)


def _sync_submission() -> None:
    code = _code_dir()
    submission = code.parent
    (submission / "figures").mkdir(exist_ok=True)
    (submission / "tables").mkdir(exist_ok=True)
    for path in (code / "figures").glob("*"):
        if path.suffix.lower() in {".pdf", ".png"}:
            shutil.copy2(path, submission / "figures" / path.name)
    for name in ("macros.tex", "tab_matched.tex", "tab_gates.tex",
                 "tab_family.tex", "tab_impotence.tex"):
        src = code / "results" / name
        if src.exists():
            shutil.copy2(src, submission / "tables" / name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.skip_run:
        _run("scripts/run.py", "--config", args.config, "--out", "results")
    _run("scripts/make_tables.py")
    _run("scripts/make_figures.py")
    _sync_submission()


if __name__ == "__main__":
    main()
