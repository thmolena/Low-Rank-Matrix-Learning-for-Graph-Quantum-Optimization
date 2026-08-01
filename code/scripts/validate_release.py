#!/usr/bin/env python3
"""Self-contained structural and evidence validator for the public release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
CODE = REPOSITORY / "code"
REQUIRED_ROOT = {
    ".gitignore", "LICENSE", "README.md", "index.html", "main.tex", "main.pdf", "code"
}
REQUIRED_CODE = {
    ".gitignore", "CITATION.cff", "LICENSE", "README.md", "configs", "data",
    "manuscript_assets", "pyproject.toml", "requirements.txt", "results", "scripts",
    "src", "tests",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    root_names = {path.name for path in REPOSITORY.iterdir() if path.name != ".git"}
    if root_names != REQUIRED_ROOT:
        fail(f"root contract mismatch: {sorted(root_names ^ REQUIRED_ROOT)}")
    code_names = {path.name for path in CODE.iterdir()}
    if code_names != REQUIRED_CODE:
        fail(f"code contract mismatch: {sorted(code_names ^ REQUIRED_CODE)}")

    forbidden_artifacts = [
        path for path in CODE.rglob("*")
        if path.name in {"__pycache__", ".pytest_cache"}
        or path.name.endswith(".egg-info")
        or path.suffix in {".pyc", ".pyo"}
    ]
    if forbidden_artifacts:
        fail(f"cache/build artifacts present: {[str(p) for p in forbidden_artifacts]}")

    manuscript = (REPOSITORY / "main.tex").read_text(encoding="utf-8")
    if len(re.findall(r"\\begin\{table\*?\}", manuscript)) != 3:
        fail("manuscript must contain exactly three substantive tables")
    if len(re.findall(r"\\begin\{figure\*?\}", manuscript)) != 4:
        fail("manuscript must contain exactly four quantitative figures")
    theorem_count = len(re.findall(r"\\begin\{(?:theorem|proposition)\}", manuscript))
    if theorem_count != 5:
        fail(f"expected five proved theorem/proposition results, found {theorem_count}")
    if manuscript.count(r"\noindent\textit{Proof.}") != theorem_count:
        fail("every theorem/proposition must have an explicit proof")
    normalized_manuscript = " ".join(manuscript.split())
    required_text = [
        "zero target depth-two evaluations",
        "does not show that four queries are necessary",
        "not a new QR/nearest-neighbor method",
        "two untouched confirmation sources",
        "not an equal-accuracy solver speedup",
        "8.88\\times10^{-16}",
        "0.011280",
        "0.011811",
        "0.006095",
    ]
    missing = [token for token in required_text if token not in normalized_manuscript]
    if missing:
        fail(f"required claim-boundary text missing: {missing}")
    forbidden_claims = [
        "quantum advantage", "first algorithm", "first use", "provably faster",
        "global optimum", "hardware speedup",
    ]
    # Scope sentences may negate these phrases; only catch positive headline forms.
    positive_patterns = [
        r"we (?:prove|demonstrate|establish) quantum advantage",
        r"our (?:novel|new) (?:DEIM|CUR|pivoted-QR)",
        r"provably faster than",
    ]
    if any(re.search(pattern, manuscript, re.I) for pattern in positive_patterns):
        fail("unsupported positive novelty or speedup claim")

    figure_stems = {
        "temperature_selection", "rank_regret", "confirmation_rmse", "runtime_scaling"
    }
    for stem in figure_stems:
        for suffix in ("pdf", "png"):
            path = CODE / "manuscript_assets" / "figures" / f"{stem}.{suffix}"
            if not path.is_file() or path.stat().st_size < 1000:
                fail(f"missing or empty figure: {path}")
    figure_names = {
        path.name for path in (CODE / "manuscript_assets" / "figures").iterdir()
        if path.is_file()
    }
    expected_figure_names = {
        f"{stem}.{suffix}" for stem in figure_stems for suffix in ("pdf", "png")
    }
    if figure_names != expected_figure_names:
        fail(f"unexpected figure artifacts: {sorted(figure_names ^ expected_figure_names)}")

    locked = json.loads((CODE / "results" / "locked_results.json").read_text())
    if locked.get("schema_version") != 3:
        fail("locked result schema is not 3")
    if not all(locked["hypotheses"].values()):
        fail("one or more locked hypotheses is false")
    train = set(locked["training_sources"])
    development = set(locked["development_sources"])
    confirmation = set(locked["confirmation_sources"])
    if not (
        train.isdisjoint(development)
        and train.isdisjoint(confirmation)
        and development.isdisjoint(confirmation)
        and len(train | development | confirmation) == 16
    ):
        fail("source split is not a disjoint sixteen-source partition")
    if locked.get("salience_temperature") != 0.0:
        fail("frozen numerical-tie rule did not select uniform weighting")
    rows = {
        row["method"]: row
        for row in locked["split_comparison"]
        if row["split"] == "confirmation"
    }
    if rows["nearest_p1_residual"]["queries"] != 0:
        fail("falsifying nearest-p1 comparator is not zero-query")
    if rows["nearest_p1_residual"]["mean_grid_regret"] != 0.0:
        fail("locked nearest-p1 confirmation regret changed")
    if not (
        rows["nearest_p1_residual"]["surface_rmse"]
        < rows["salience_residual"]["surface_rmse"]
    ):
        fail("falsifying nearest-p1 RMSE comparison changed")
    if not (
        rows["anchor_gated_residual"]["surface_rmse"]
        < rows["nearest_p1_residual"]["surface_rmse"]
        and rows["anchor_gated_residual"]["exact_decision_fraction"] == 1.0
        and rows["nearest_p1_residual"]["exact_decision_fraction"] == 1.0
    ):
        fail("gated reconstruction gain or zero-query decision tie changed")

    runtime = json.loads((CODE / "results" / "runtime.json").read_text())
    if runtime.get("source") != "G67":
        fail("runtime audit is not the registered G67 confirmation task")
    if {row["repeats"] for row in runtime["rows"]} != {7}:
        fail("runtime audit does not use seven measured repetitions")

    manifest_path = CODE / "results" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        path = REPOSITORY / relative
        if not path.is_file():
            fail(f"manifest path missing: {relative}")
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            fail(f"manifest mismatch: {relative}")

    pdf = REPOSITORY / "main.pdf"
    if pdf.stat().st_size < 50_000 or not pdf.read_bytes().startswith(b"%PDF"):
        fail("main.pdf is missing, stale, or implausibly small")
    if "qaoa-anchor-reproduce" not in (CODE / "pyproject.toml").read_text():
        fail("reproduction console entry point is missing")
    if "qaoa-anchor-benchmark" not in (CODE / "pyproject.toml").read_text():
        fail("benchmark console entry point is missing")

    print(
        json.dumps(
            {
                "release_contract": "PASS",
                "tables": 3,
                "figures": 4,
                "proved_results": theorem_count,
                "locked_hypotheses": len(locked["hypotheses"]),
                "manifest_files": len(manifest["files"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
