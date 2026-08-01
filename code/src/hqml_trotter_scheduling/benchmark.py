"""Single-host runtime audit for exhaustive and anchor-gated target evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from .experiment import (
    CANDIDATES,
    PRIMARY_RANK,
    RANDOM_SEED,
    SOURCE_HASHES,
    TASK_VERTICES,
    cost_diagonal,
    default_cache_dir,
    download_source,
    edge_statistics,
    induced_bfs_edges,
    p1_formula,
    parse_source,
    qaoa_energy,
)


def _median_seconds(function, repeats: int) -> tuple[float, list[float]]:
    function()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) * 1.0e-9)
    return float(np.median(samples)), samples


def run_benchmark(
    result_path: Path,
    cache_dir: Path | None = None,
    *,
    repeats: int = 7,
) -> dict[str, object]:
    """Benchmark one authenticated confirmation task without changing science data."""

    if repeats < 3:
        raise ValueError("at least three timing repeats are required")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    model = result["primary_model"]
    anchors = np.asarray(model["anchors"], dtype=np.int64)
    training_residual = np.asarray(model["training_residual"], dtype=np.float64)
    if anchors.size != PRIMARY_RANK or training_residual.shape[1] != CANDIDATES.shape[0]:
        raise RuntimeError("locked primary model has an unexpected shape")

    source_name = "G67"
    source_path = download_source(source_name, cache_dir or default_cache_dir())
    graph = parse_source(source_name, source_path)
    source_index = 14  # G67's registered position in the sixteen-source protocol.
    start_vertex = (97 * source_index) % graph.vertices
    edges = induced_bfs_edges(
        graph,
        start_vertex,
        seed=RANDOM_SEED + 10007 * source_index,
    )
    diagonal = cost_diagonal(edges, TASK_VERTICES)
    statistics = edge_statistics(edges, TASK_VERTICES)
    normalizer = float(len(edges))

    def analytic_baseline(count: int) -> np.ndarray:
        lookup: dict[tuple[float, float], float] = {}
        values = np.empty(count, dtype=np.float64)
        for index, candidate in enumerate(CANDIDATES[:count]):
            pair = (float(candidate[0]), float(candidate[1]))
            if pair not in lookup:
                lookup[pair] = p1_formula(statistics, *pair) / normalizer
            values[index] = lookup[pair]
        return values

    anchor_truth = np.asarray(
        [qaoa_energy(diagonal, CANDIDATES[index]) / normalizer for index in anchors]
    )
    anchor_baseline = np.asarray(
        [
            p1_formula(statistics, *CANDIDATES[index, :2]) / normalizer
            for index in anchors
        ]
    )
    anchor_residual = anchor_truth - anchor_baseline

    def transferred_residual(measured: np.ndarray) -> np.ndarray:
        distances = np.mean(
            (training_residual[:, anchors] - measured[None, :]) ** 2,
            axis=1,
        )
        return training_residual[int(np.argmin(distances))]

    rows: list[dict[str, object]] = []
    for count in (16, 32, 64, 128, 256):
        def exhaustive() -> np.ndarray:
            return np.asarray(
                [
                    qaoa_energy(diagonal, candidate) / normalizer
                    for candidate in CANDIDATES[:count]
                ]
            )

        def anchored() -> np.ndarray:
            # The four target evaluations are included in every timed call.
            measured = np.asarray(
                [
                    qaoa_energy(diagonal, CANDIDATES[index]) / normalizer
                    for index in anchors
                ]
            )
            residual = measured - anchor_baseline
            return analytic_baseline(count) + transferred_residual(residual)[:count]

        exhaustive_median, exhaustive_samples = _median_seconds(exhaustive, repeats)
        anchored_median, anchored_samples = _median_seconds(anchored, repeats)
        reference = exhaustive()
        reconstruction = analytic_baseline(count) + transferred_residual(anchor_residual)[:count]
        rows.extend(
            [
                {
                    "candidate_count": count,
                    "method": "exhaustive_grid",
                    "depth_two_queries": count,
                    "median_seconds": exhaustive_median,
                    "minimum_seconds": min(exhaustive_samples),
                    "maximum_seconds": max(exhaustive_samples),
                    "repeats": repeats,
                    "checksum": float(np.sum(reference)),
                    "surface_rmse": 0.0,
                    "maximum_absolute_error": 0.0,
                },
                {
                    "candidate_count": count,
                    "method": "four_anchor_transfer",
                    "depth_two_queries": PRIMARY_RANK,
                    "median_seconds": anchored_median,
                    "minimum_seconds": min(anchored_samples),
                    "maximum_seconds": max(anchored_samples),
                    "repeats": repeats,
                    "checksum": float(np.sum(reconstruction)),
                    "surface_rmse": float(
                        np.sqrt(np.mean((reference - reconstruction) ** 2))
                    ),
                    "maximum_absolute_error": float(
                        np.max(np.abs(reference - reconstruction))
                    ),
                },
            ]
        )

    return {
        "schema_version": 1,
        "scope": "single-host implementation timing; not a hardware-independent speedup claim",
        "source": source_name,
        "source_sha256": SOURCE_HASHES[source_name],
        "task": 0,
        "start_vertex": start_vertex,
        "task_vertices": TASK_VERTICES,
        "task_edges": len(edges),
        "anchors": anchors.tolist(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rows": rows,
    }


def write_benchmark(result: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    rows = list(result["rows"])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("code/results/locked_results.json")
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("code/results/runtime.json"))
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    result = run_benchmark(args.results, args.cache_dir, repeats=args.repeats)
    write_benchmark(result, args.output)
    print(json.dumps({"scope": result["scope"], "rows": result["rows"]}, indent=2))


if __name__ == "__main__":
    main()
