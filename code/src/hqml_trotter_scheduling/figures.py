"""Line and bar figures for anchor-gated residual matrix transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "p1_baseline": "depth-one baseline",
    "mean_residual": "mean residual (0 query)",
    "nearest_p1_residual": "nearest p=1 residual (0 query)",
    "salience_residual": "weighted anchor-linear",
    "uniform_residual": "uniform anchor-linear",
    "random_residual": "fixed random anchors",
    "salience_raw": "uniform raw-linear",
    "anchor_gated_residual": "anchor-gated transfer",
    "exhaustive_grid": "exhaustive grid",
}


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def _temperature_line(result: dict, output: Path) -> None:
    rows = result["temperature_sweep"]
    x = [float(row["temperature"]) for row in rows]
    regret = [float(row["mean_grid_regret"]) for row in rows]
    rmse = [float(row["surface_rmse"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    axes[0].plot(x, regret, "o-", color="tab:blue")
    axes[0].set_xlabel(r"salience temperature $\tau$")
    axes[0].set_ylabel("development mean grid regret")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(x, rmse, "s-", color="tab:orange")
    axes[1].set_xlabel(r"salience temperature $\tau$")
    axes[1].set_ylabel("development surface RMSE")
    axes[1].grid(True, alpha=0.25)
    _save(figure, output / "temperature_selection")


def _rank_line(result: dict, output: Path) -> None:
    rows = [row for row in result["rank_sweep"] if row["split"] == "development"]
    figure, axis = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    styles = {
        "anchor_gated_residual": "o-",
        "uniform_residual": "s--",
        "random_residual": "^:",
        "salience_raw": "d-.",
    }
    for method, style in styles.items():
        selected = [row for row in rows if row["method"] == method]
        axis.plot(
            [int(row["rank"]) for row in selected],
            [float(row["mean_grid_regret"]) for row in selected],
            style,
            label=LABELS[method],
        )
    axis.set_yscale("symlog", linthresh=1.0e-5)
    axis.set_xlabel("target QAOA queries / anchor count")
    axis.set_ylabel("development mean grid regret")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=7)
    _save(figure, output / "rank_regret")


def _confirmation_rmse_bar(result: dict, output: Path) -> None:
    rows = [
        row for row in result["primary_comparison"]
        if row["method"]
        in (
            "p1_baseline",
            "mean_residual",
            "nearest_p1_residual",
            "uniform_residual",
            "anchor_gated_residual",
            "salience_raw",
        )
    ]
    labels = [LABELS[row["method"]] for row in rows]
    values = [float(row["surface_rmse"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    positions = np.arange(len(rows))
    axis.bar(
        positions,
        values,
        color=("0.55", "0.72", "tab:cyan", "tab:blue", "tab:green", "tab:red", "tab:purple"),
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=22, ha="right")
    axis.set_ylabel("confirmation surface RMSE")
    axis.grid(True, axis="y", alpha=0.25)
    _save(figure, output / "confirmation_rmse")


def _runtime_line(runtime_path: Path, output: Path) -> None:
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    figure, axis = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    for method, style, color in (
        ("exhaustive_grid", "o-", "0.25"),
        ("four_anchor_transfer", "s-", "tab:blue"),
    ):
        rows = [row for row in runtime["rows"] if row["method"] == method]
        axis.plot(
            [int(row["candidate_count"]) for row in rows],
            [1.0e3 * float(row["median_seconds"]) for row in rows],
            style,
            color=color,
            label=method.replace("_", " "),
        )
    axis.set_xlabel("candidate count")
    axis.set_ylabel("median wall time (ms)")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    _save(figure, output / "runtime_scaling")


def make_all(result_path: Path, output: Path) -> list[Path]:
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    _temperature_line(result, output)
    _rank_line(result, output)
    _confirmation_rmse_bar(result, output)
    _runtime_line(Path(result_path).with_name("runtime.json"), output)
    return [
        output / f"{stem}.{suffix}"
        for stem in ("temperature_selection", "rank_regret", "confirmation_rmse", "runtime_scaling")
        for suffix in ("pdf", "png")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("code/results/locked_results.json"))
    parser.add_argument("--output", type=Path, default=Path("code/manuscript_assets/figures"))
    args = parser.parse_args()
    for path in make_all(args.results, args.output):
        print(path)


if __name__ == "__main__":
    main()
