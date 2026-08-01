"""Authenticated depth-two QAOA study for anchor-gated residual transfer.

The released algorithm learns a small set of QAOA angle anchors from source
graphs.  On a new graph it evaluates only those anchors, subtracts an exact
depth-one analytic baseline, and transfers the training residual whose anchor
values are nearest.  Source files are downloaded from the
Stanford Gset collection and checked against registered SHA-256 digests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.linalg import qr

BASE_URL = "https://web.stanford.edu/~yyye/yyye/Gset"
SOURCE_HASHES = {
    "G1": "73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665",
    "G6": "e3dd8b29205280f1ad5806849321cdff6f11324c75cb95647d5783f3e47a1103",
    "G11": "c2a760d2926db4fefd23b25c098dcd6311f711b355dbd1cc689fa25660c73174",
    "G14": "dc769b978a40d458f693d5bd2cf8b8cceabd430b8e976204746696179c3d5945",
    "G18": "6185af31fae0f7d59e232dad91178753574b7caa111dd6e6bf52f11c80e928d8",
    "G22": "9baeee06eb147b1c9ca42b43be86592d4e6fc60784a85af9be5b63d1362ef28e",
    "G32": "9760fce6b601832db1b1173c1a09c8556d64bfb27ba603ef52ca24ee6c8527cc",
    "G43": "9af5445b4b066cbf1eabe218d4e0d907cb6f211651cae557c761ec344dc37be8",
    "G55": "7537bbb613a6e6d0fe784937be68099eeb36227d85bb273299f22cb7d3ff2227",
    "G70": "0d965a2ff1444cc71d7740824b5b3a5e09fc1843cbf6d90691b52d5221987061",
    "G60": "b6480c1716ecca3bb9dac8e444746d6bf0914e9faed2d4e63cececa2af68a5c4",
    "G72": "d0495e1325cf400bfd6488d7b843670a32ffae7dcdeda0c2a9d9c2452223ef18",
    "G65": "8f110238fb6b9b7f19d6ba981f69768a8a55dda8005dae2ff9a463e8d4e12a04",
    "G66": "0af24b7e39277c707faab38f3a8baf773c2b4fa283454915f0c0371233a9a163",
    "G67": "2a8bd22b13b13e43ccc2c1fd936397e5c1680b60bf6086b794dd9bf487fc82d5",
    "G77": "b90932aa0d671319aed088853a6054739e5879986f314b4a98a1233b71b96fb1",
}
TRAIN_SOURCES = ("G1", "G6", "G11", "G14", "G18", "G22")
DEVELOPMENT_SOURCES = ("G32", "G43", "G55", "G70", "G60", "G72", "G65", "G66")
CONFIRMATION_SOURCES = ("G67", "G77")
SOURCES = TRAIN_SOURCES + DEVELOPMENT_SOURCES + CONFIRMATION_SOURCES

TASKS_PER_SOURCE = 8
TASK_VERTICES = 10
GAMMAS = np.linspace(0.18, 1.82, 4)
BETAS = np.linspace(0.08, 0.72, 4)
CANDIDATES = np.asarray(
    [
        (gamma_1, beta_1, gamma_2, beta_2)
        for gamma_1 in GAMMAS
        for beta_1 in BETAS
        for gamma_2 in GAMMAS
        for beta_2 in BETAS
    ],
    dtype=np.float64,
)
PRIMARY_RANK = 4
SALIENCE_TEMPERATURE = 0.0
RANKS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
TEMPERATURES = (0.0, 2.0, 4.0, 8.0, 12.0, 20.0, 40.0, 80.0)
RANDOM_SEED = 1808


@dataclass(frozen=True)
class SourceGraph:
    """One authenticated source graph."""

    name: str
    vertices: int
    declared_edges: int
    edges: tuple[tuple[int, int], ...]
    adjacency: tuple[tuple[int, ...], ...]
    sha256: str


@dataclass(frozen=True)
class TaskRecord:
    """A normalized QAOA response surface on an induced source subgraph."""

    source: str
    task: int
    start_vertex: int
    edges: int
    surface: NDArray[np.float64]
    baseline: NDArray[np.float64]
    formula_error: float


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_cache_dir() -> Path:
    """Return the external cache used for source files."""

    override = os.environ.get("QAOA_ANCHOR_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "qaoa-anchor-learning" / "gset"


def download_source(name: str, cache_dir: Path | None = None) -> Path:
    """Download one registered Gset file and verify its byte stream."""

    if name not in SOURCE_HASHES:
        raise KeyError(f"unregistered Gset source: {name}")
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / name
    expected = SOURCE_HASHES[name]
    if destination.is_file() and sha256(destination) == expected:
        return destination
    temporary = cache / f".{name}.download"
    urllib.request.urlretrieve(f"{BASE_URL}/{name}", temporary)  # noqa: S310
    observed = sha256(temporary)
    if observed != expected:
        raise RuntimeError(
            f"checksum mismatch for {name}: expected {expected}, observed {observed}"
        )
    temporary.replace(destination)
    return destination


def parse_source(name: str, path: Path) -> SourceGraph:
    """Parse one one-indexed Gset edge list as an unweighted simple graph."""

    observed = sha256(path)
    expected = SOURCE_HASHES[name]
    if observed != expected:
        raise RuntimeError(
            f"checksum mismatch for {name}: expected {expected}, observed {observed}"
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    vertices, declared_edges = map(int, lines[0].split())
    edge_set: set[tuple[int, int]] = set()
    adjacency: list[set[int]] = [set() for _ in range(vertices)]
    for line in lines[1:]:
        if not line.strip():
            continue
        left, right, _weight = line.split()
        u, v = int(left) - 1, int(right) - 1
        if u == v:
            continue
        edge = (min(u, v), max(u, v))
        edge_set.add(edge)
        adjacency[u].add(v)
        adjacency[v].add(u)
    if len(edge_set) != declared_edges:
        raise RuntimeError(
            f"{name} declares {declared_edges} edges but contains {len(edge_set)}"
        )
    return SourceGraph(
        name=name,
        vertices=vertices,
        declared_edges=declared_edges,
        edges=tuple(sorted(edge_set)),
        adjacency=tuple(tuple(sorted(neighbors)) for neighbors in adjacency),
        sha256=observed,
    )


def induced_bfs_edges(
    graph: SourceGraph,
    start: int,
    size: int = TASK_VERTICES,
    *,
    seed: int = 0,
) -> tuple[tuple[int, int], ...]:
    """Return an induced graph on a reproducibly permuted BFS prefix."""

    queue = [int(start)]
    seen = {int(start)}
    generator = np.random.default_rng(seed)
    for vertex in queue:
        neighbors = np.asarray(graph.adjacency[vertex], dtype=np.int64)
        for neighbor in generator.permutation(neighbors):
            neighbor = int(neighbor)
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
                if len(queue) == size:
                    break
        if len(queue) == size:
            break
    if len(queue) != size:
        raise RuntimeError(f"BFS from {start} in {graph.name} found only {len(queue)} vertices")
    relabel = {vertex: index for index, vertex in enumerate(queue)}
    return tuple(
        (relabel[u], relabel[v])
        for u, v in graph.edges
        if u in relabel and v in relabel
    )


def cost_diagonal(
    edges: Iterable[tuple[int, int]], vertices: int = TASK_VERTICES
) -> NDArray[np.float64]:
    """Return the unweighted MaxCut diagonal in the computational basis."""

    states = np.arange(1 << vertices, dtype=np.uint32)
    diagonal = np.zeros(states.size, dtype=np.float64)
    for u, v in edges:
        diagonal += ((states >> u) ^ (states >> v)) & 1
    return diagonal


def apply_mixer(
    state: NDArray[np.complex128], beta: float, vertices: int = TASK_VERTICES
) -> NDArray[np.complex128]:
    """Apply the product mixer without forming its matrix."""

    result = np.asarray(state, dtype=np.complex128).copy()
    cosine = math.cos(float(beta))
    sine = -1j * math.sin(float(beta))
    for qubit in range(vertices):
        block = 1 << qubit
        view = result.reshape(-1, 2 * block)
        left = view[:, :block].copy()
        right = view[:, block:].copy()
        view[:, :block] = cosine * left + sine * right
        view[:, block:] = sine * left + cosine * right
    return result


def qaoa_energy(
    diagonal: NDArray[np.float64],
    angles: Iterable[float],
    *,
    depth: int = 2,
) -> float:
    """Evaluate a depth-one or depth-two QAOA expectation by state vectors."""

    gamma_1, beta_1, gamma_2, beta_2 = tuple(float(value) for value in angles)
    vertices = int(round(math.log2(diagonal.size)))
    if 1 << vertices != diagonal.size:
        raise ValueError("diagonal length must be a power of two")
    state = np.ones(diagonal.size, dtype=np.complex128) / math.sqrt(diagonal.size)
    state *= np.exp(-1j * gamma_1 * diagonal)
    state = apply_mixer(state, beta_1, vertices)
    if depth == 2:
        state *= np.exp(-1j * gamma_2 * diagonal)
        state = apply_mixer(state, beta_2, vertices)
    elif depth != 1:
        raise ValueError("depth must be one or two")
    return float(np.dot(np.abs(state) ** 2, diagonal))


def edge_statistics(
    edges: Iterable[tuple[int, int]], vertices: int = TASK_VERTICES
) -> tuple[tuple[int, int, int], ...]:
    """Return exclusive degrees and triangle counts for every edge."""

    edge_list = tuple(edges)
    neighbors = [set() for _ in range(vertices)]
    for u, v in edge_list:
        neighbors[u].add(v)
        neighbors[v].add(u)
    return tuple(
        (
            len(neighbors[u]) - 1,
            len(neighbors[v]) - 1,
            len(neighbors[u] & neighbors[v]),
        )
        for u, v in edge_list
    )


def p1_formula(
    statistics: Iterable[tuple[int, int, int]], gamma: float, beta: float
) -> float:
    """Evaluate the established exact depth-one unweighted-MaxCut formula."""

    cosine = math.cos(float(gamma))
    common_cosine = math.cos(2.0 * float(gamma))
    linear = 0.25 * math.sin(4.0 * float(beta)) * math.sin(float(gamma))
    triangle = 0.25 * math.sin(2.0 * float(beta)) ** 2
    expectation = 0.0
    for exclusive_u, exclusive_v, common in statistics:
        expectation += 0.5
        expectation += linear * (
            cosine**exclusive_u + cosine**exclusive_v
        )
        expectation -= (
            triangle
            * cosine ** (exclusive_u + exclusive_v - 2 * common)
            * (1.0 - common_cosine**common)
        )
    return expectation


def build_task_records(
    graphs: dict[str, SourceGraph],
) -> list[TaskRecord]:
    """Generate all deterministic source-graph tasks and response surfaces."""

    records: list[TaskRecord] = []
    unique_first_layer = tuple(
        (float(gamma), float(beta)) for gamma in GAMMAS for beta in BETAS
    )
    for source_index, name in enumerate(SOURCES):
        graph = graphs[name]
        for task in range(TASKS_PER_SOURCE):
            start = (37 * task + 97 * source_index) % graph.vertices
            edges = induced_bfs_edges(
                graph,
                start,
                seed=RANDOM_SEED + 10007 * source_index + task,
            )
            if not edges:
                raise RuntimeError(f"empty induced task {name}:{task}")
            diagonal = cost_diagonal(edges)
            normalizer = float(len(edges))
            surface = np.asarray(
                [qaoa_energy(diagonal, candidate) / normalizer for candidate in CANDIDATES],
                dtype=np.float64,
            )
            statistics = edge_statistics(edges)
            baseline_lookup = {
                pair: p1_formula(statistics, *pair) / normalizer
                for pair in unique_first_layer
            }
            baseline = np.asarray(
                [baseline_lookup[(float(row[0]), float(row[1]))] for row in CANDIDATES],
                dtype=np.float64,
            )
            formula_error = max(
                abs(
                    baseline_lookup[pair]
                    - qaoa_energy(diagonal, (*pair, 0.0, 0.0), depth=1)
                    / normalizer
                )
                for pair in unique_first_layer
            )
            records.append(
                TaskRecord(
                    source=name,
                    task=task,
                    start_vertex=start,
                    edges=len(edges),
                    surface=surface,
                    baseline=baseline,
                    formula_error=float(formula_error),
                )
            )
    return records


def salience_weights(
    training_surfaces: NDArray[np.float64], temperature: float
) -> NDArray[np.float64]:
    """Return mean softmax mass over candidate columns, normalized to mean one."""

    if temperature < 0.0:
        raise ValueError("temperature must be nonnegative")
    if temperature == 0.0:
        return np.ones(training_surfaces.shape[1], dtype=np.float64)
    shifted = training_surfaces - np.max(training_surfaces, axis=1, keepdims=True)
    weights = np.exp(float(temperature) * shifted)
    weights /= np.sum(weights, axis=1, keepdims=True)
    salience = np.mean(weights, axis=0)
    return salience / np.mean(salience)


def pivot_columns(
    matrix: NDArray[np.float64],
    rank: int,
    salience: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Select a nested column set by pivoted QR of the weighted matrix."""

    if rank < 1 or rank > min(matrix.shape):
        raise ValueError("rank is outside the admissible range")
    if salience.shape != (matrix.shape[1],) or np.any(salience <= 0.0):
        raise ValueError("salience must be a positive candidate vector")
    weighted = matrix * np.sqrt(salience)[None, :]
    _q, _r, pivots = qr(weighted, mode="economic", pivoting=True)
    return np.asarray(pivots[:rank], dtype=np.int64)


def fit_interpolation(
    training_matrix: NDArray[np.float64], anchors: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return the minimum-norm anchor-to-surface matrix."""

    sampled = training_matrix[:, anchors]
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        sampled, training_matrix, rcond=1.0e-12
    )
    return np.asarray(coefficients, dtype=np.float64)


def metric_record(
    prediction: NDArray[np.float64], truth: NDArray[np.float64]
) -> dict[str, float]:
    """Return surface and finite-grid decision metrics."""

    error = prediction - truth
    predicted = np.argmax(prediction, axis=1)
    optimal = np.argmax(truth, axis=1)
    regret = np.max(truth, axis=1) - truth[np.arange(truth.shape[0]), predicted]
    infinity_error = np.max(np.abs(error), axis=1)
    bound_slack = 2.0 * infinity_error - regret
    return {
        "surface_rmse": float(np.sqrt(np.mean(error**2))),
        "mean_infinity_error": float(np.mean(infinity_error)),
        "mean_grid_regret": float(np.mean(regret)),
        "maximum_grid_regret": float(np.max(regret)),
        "exact_decision_fraction": float(np.mean(predicted == optimal)),
        "minimum_regret_bound_slack": float(np.min(bound_slack)),
        "regret_bound_covered": float(np.all(bound_slack >= -1.0e-14)),
    }


def method_prediction(
    method: str,
    rank: int,
    train_y: NDArray[np.float64],
    train_b: NDArray[np.float64],
    target_y: NDArray[np.float64],
    target_b: NDArray[np.float64],
    salience: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Fit one declared comparator and reconstruct the target surfaces."""

    residual_train = train_y - train_b
    residual_target = target_y - target_b
    if method == "salience_residual":
        matrix, target, offset = residual_train, residual_target, target_b
        anchors = pivot_columns(matrix, rank, salience)
    elif method == "uniform_residual":
        matrix, target, offset = residual_train, residual_target, target_b
        anchors = pivot_columns(matrix, rank, np.ones_like(salience))
    elif method == "random_residual":
        matrix, target, offset = residual_train, residual_target, target_b
        permutation = np.random.default_rng(RANDOM_SEED).permutation(matrix.shape[1])
        anchors = np.sort(permutation[:rank]).astype(np.int64)
    elif method == "salience_raw":
        matrix, target, offset = train_y, target_y, 0.0
        anchors = pivot_columns(matrix, rank, salience)
    else:
        raise ValueError(f"unknown method: {method}")
    interpolation = fit_interpolation(matrix, anchors)
    prediction = offset + target[:, anchors] @ interpolation
    return np.asarray(prediction), anchors, interpolation


def zero_query_prediction(
    method: str,
    train_y: NDArray[np.float64],
    train_b: NDArray[np.float64],
    target_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict without depth-two target evaluations.

    Both comparators use only training responses and the target's analytic
    depth-one surface.  ``mean_residual`` adds the mean training residual.
    ``nearest_p1_residual`` transfers the residual of the training task whose
    depth-one surface is closest in root-mean-square distance.
    """

    residual = train_y - train_b
    if method == "mean_residual":
        return target_b + np.mean(residual, axis=0, keepdims=True)
    if method == "nearest_p1_residual":
        squared_distance = np.mean(
            (target_b[:, None, :] - train_b[None, :, :]) ** 2,
            axis=2,
        )
        nearest = np.argmin(squared_distance, axis=1)
        return target_b + residual[nearest]
    raise ValueError(f"unknown zero-query method: {method}")


def nearest_anchor_prediction(
    train_y: NDArray[np.float64],
    train_b: NDArray[np.float64],
    target_y: NDArray[np.float64],
    target_b: NDArray[np.float64],
    anchors: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Transfer the nearest training residual using the measured anchors."""

    train_residual = train_y - train_b
    target_anchor_residual = target_y[:, anchors] - target_b[:, anchors]
    squared_distance = np.mean(
        (
            target_anchor_residual[:, None, :]
            - train_residual[None, :, anchors]
        ) ** 2,
        axis=2,
    )
    nearest = np.argmin(squared_distance, axis=1)
    return target_b + train_residual[nearest]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def run_study(cache_dir: Path | None = None) -> dict[str, object]:
    """Execute the fixed training, development, and confirmation protocol."""

    paths = {name: download_source(name, cache_dir) for name in SOURCES}
    graphs = {name: parse_source(name, paths[name]) for name in SOURCES}
    records = build_task_records(graphs)
    sources = np.asarray([record.source for record in records])
    y = np.stack([record.surface for record in records])
    baseline = np.stack([record.baseline for record in records])
    train_mask = np.isin(sources, TRAIN_SOURCES)
    development_mask = np.isin(sources, DEVELOPMENT_SOURCES)
    confirmation_mask = np.isin(sources, CONFIRMATION_SOURCES)
    train_y, train_b = y[train_mask], baseline[train_mask]
    salience = salience_weights(train_y, SALIENCE_TEMPERATURE)

    source_rows: list[dict[str, object]] = []
    for name in SOURCES:
        graph = graphs[name]
        split = (
            "training" if name in TRAIN_SOURCES else
            "development" if name in DEVELOPMENT_SOURCES else
            "confirmation"
        )
        task_rows = [record for record in records if record.source == name]
        source_rows.append(
            {
                "source": name,
                "split": split,
                "vertices": graph.vertices,
                "declared_edges": graph.declared_edges,
                "derived_tasks": len(task_rows),
                "minimum_task_edges": min(record.edges for record in task_rows),
                "maximum_task_edges": max(record.edges for record in task_rows),
                "sha256": graph.sha256,
                "url": f"{BASE_URL}/{name}",
            }
        )

    tau_rows: list[dict[str, object]] = []
    for temperature in TEMPERATURES:
        temperature_salience = salience_weights(train_y, temperature)
        anchors = pivot_columns(
            train_y - train_b,
            PRIMARY_RANK,
            temperature_salience,
        )
        prediction = nearest_anchor_prediction(
            train_y,
            train_b,
            y[development_mask],
            baseline[development_mask],
            anchors,
        )
        tau_rows.append(
            {
                "temperature": temperature,
                "rank": PRIMARY_RANK,
                "anchors": " ".join(map(str, anchors.tolist())),
                **metric_record(prediction, y[development_mask]),
            }
        )

    methods = (
        "salience_residual",
        "uniform_residual",
        "random_residual",
        "salience_raw",
    )
    rank_rows: list[dict[str, object]] = []
    for split, mask in (
        ("development", development_mask),
        ("confirmation", confirmation_mask),
    ):
        for rank in RANKS:
            gated_anchors = pivot_columns(
                train_y - train_b,
                rank,
                salience,
            )
            gated_prediction = nearest_anchor_prediction(
                train_y,
                train_b,
                y[mask],
                baseline[mask],
                gated_anchors,
            )
            rank_rows.append(
                {
                    "split": split,
                    "method": "anchor_gated_residual",
                    "rank": rank,
                    "queries": rank,
                    "query_fraction": rank / CANDIDATES.shape[0],
                    "anchors": " ".join(map(str, gated_anchors.tolist())),
                    "interpolation_one_norm": 0.0,
                    **metric_record(gated_prediction, y[mask]),
                }
            )
            for method in methods:
                prediction, anchors, interpolation = method_prediction(
                    method,
                    rank,
                    train_y,
                    train_b,
                    y[mask],
                    baseline[mask],
                    salience,
                )
                rank_rows.append(
                    {
                        "split": split,
                        "method": method,
                        "rank": rank,
                        "queries": rank,
                        "query_fraction": rank / CANDIDATES.shape[0],
                        "anchors": " ".join(map(str, anchors.tolist())),
                        "interpolation_one_norm": float(
                            np.linalg.norm(interpolation, ord=1)
                        ),
                        **metric_record(prediction, y[mask]),
                    }
                )

    split_rows: list[dict[str, object]] = []
    split_predictions: dict[str, dict[str, NDArray[np.float64]]] = {}
    primary_model: dict[str, object] | None = None
    for split, mask in (
        ("development", development_mask),
        ("confirmation", confirmation_mask),
    ):
        target_y, target_b = y[mask], baseline[mask]
        predictions: dict[str, NDArray[np.float64]] = {"p1_baseline": target_b}
        split_rows.append(
            {
                "split": split,
                "method": "p1_baseline",
                "queries": 0,
                "query_fraction": 0.0,
                "anchors": "",
                "interpolation_one_norm": 0.0,
                **metric_record(target_b, target_y),
            }
        )
        for method in ("mean_residual", "nearest_p1_residual"):
            prediction = zero_query_prediction(method, train_y, train_b, target_b)
            predictions[method] = prediction
            split_rows.append(
                {
                    "split": split,
                    "method": method,
                    "queries": 0,
                    "query_fraction": 0.0,
                    "anchors": "",
                    "interpolation_one_norm": 0.0,
                    **metric_record(prediction, target_y),
                }
            )
        salience_anchors: NDArray[np.int64] | None = None
        for method in methods:
            prediction, anchors, interpolation = method_prediction(
                method,
                PRIMARY_RANK,
                train_y,
                train_b,
                target_y,
                target_b,
                salience,
            )
            predictions[method] = prediction
            split_rows.append(
                {
                    "split": split,
                    "method": method,
                    "queries": PRIMARY_RANK,
                    "query_fraction": PRIMARY_RANK / CANDIDATES.shape[0],
                    "anchors": " ".join(map(str, anchors.tolist())),
                    "interpolation_one_norm": float(
                        np.linalg.norm(interpolation, ord=1)
                    ),
                    **metric_record(prediction, target_y),
                }
            )
            if method == "salience_residual":
                salience_anchors = anchors
                if split == "confirmation":
                    primary_model = {
                        "anchors": anchors,
                        "interpolation": interpolation,
                    }
        assert salience_anchors is not None
        prediction = nearest_anchor_prediction(
            train_y, train_b, target_y, target_b, salience_anchors
        )
        predictions["anchor_gated_residual"] = prediction
        split_rows.append(
            {
                "split": split,
                "method": "anchor_gated_residual",
                "queries": PRIMARY_RANK,
                "query_fraction": PRIMARY_RANK / CANDIDATES.shape[0],
                "anchors": " ".join(map(str, salience_anchors.tolist())),
                "interpolation_one_norm": 0.0,
                **metric_record(prediction, target_y),
            }
        )
        if split == "confirmation":
            primary_model = {
                "anchors": salience_anchors,
                "training_residual": train_y - train_b,
            }
        predictions["exhaustive_grid"] = target_y
        split_rows.append(
            {
                "split": split,
                "method": "exhaustive_grid",
                "queries": CANDIDATES.shape[0],
                "query_fraction": 1.0,
                "anchors": "all",
                "interpolation_one_norm": 1.0,
                **metric_record(target_y, target_y),
            }
        )
        split_predictions[split] = predictions

    primary_rows = [row for row in split_rows if row["split"] == "confirmation"]
    predictions = split_predictions["confirmation"]

    per_source_rows: list[dict[str, object]] = []
    for name in CONFIRMATION_SOURCES:
        local = sources[confirmation_mask] == name
        for method, prediction in predictions.items():
            per_source_rows.append(
                {
                    "source": name,
                    "method": method,
                    **metric_record(prediction[local], y[confirmation_mask][local]),
                }
            )

    residual = train_y - train_b
    raw_singular = np.linalg.svd(train_y, compute_uv=False)
    residual_singular = np.linalg.svd(residual, compute_uv=False)
    weighted_singular = np.linalg.svd(
        residual * np.sqrt(salience)[None, :], compute_uv=False
    )
    singular_rows = [
        {
            "index": index + 1,
            "raw": float(raw_singular[index]),
            "residual": float(residual_singular[index]),
            "salience_weighted_residual": float(weighted_singular[index]),
        }
        for index in range(min(len(raw_singular), len(residual_singular)))
    ]

    primary = {str(row["method"]): row for row in primary_rows}
    development = {
        str(row["method"]): row
        for row in split_rows
        if row["split"] == "development"
    }
    best_dev_tau = min(
        tau_rows,
        key=lambda row: (
            round(float(row["mean_grid_regret"]), 12),
            round(float(row["surface_rmse"]), 12),
            float(row["temperature"]),
        ),
    )
    hypotheses = {
        "source_splits_are_pairwise_disjoint": (
            set(TRAIN_SOURCES).isdisjoint(DEVELOPMENT_SOURCES)
            and set(TRAIN_SOURCES).isdisjoint(CONFIRMATION_SOURCES)
            and set(DEVELOPMENT_SOURCES).isdisjoint(CONFIRMATION_SOURCES)
        ),
        "all_source_hashes_match": all(
            graphs[name].sha256 == SOURCE_HASHES[name] for name in SOURCES
        ),
        "depth_one_formula_max_error_below_1e_12": max(
            record.formula_error for record in records
        ) < 1.0e-12,
        "development_selects_registered_temperature": float(
            best_dev_tau["temperature"]
        ) == SALIENCE_TEMPERATURE,
        "development_anchor_gating_improves_zero_query_nearest_rmse": float(
            development["anchor_gated_residual"]["surface_rmse"]
        ) < float(development["nearest_p1_residual"]["surface_rmse"]),
        "confirmation_anchor_gating_improves_zero_query_nearest_rmse": float(
            primary["anchor_gated_residual"]["surface_rmse"]
        ) < float(primary["nearest_p1_residual"]["surface_rmse"]),
        "confirmation_anchor_gating_has_smallest_nonexhaustive_rmse": float(
            primary["anchor_gated_residual"]["surface_rmse"]
        ) == min(
            float(row["surface_rmse"])
            for row in primary_rows
            if row["method"] != "exhaustive_grid"
        ),
        "confirmation_anchor_gating_preserves_every_grid_decision": float(
            primary["anchor_gated_residual"]["exact_decision_fraction"]
        ) == 1.0,
        "anchor_method_uses_four_of_256_target_queries": int(
            primary["anchor_gated_residual"]["queries"]
        ) == 4 and CANDIDATES.shape[0] == 256,
        "regret_bound_covers_every_reported_prediction": all(
            float(row["regret_bound_covered"]) == 1.0
            for row in rank_rows + split_rows
        ),
    }
    result: dict[str, object] = {
        "schema_version": 3,
        "title": "Source-Disjoint Audit of QR-Anchor-Gated Residual Transfer for Graph Quantum Optimization",
        "algorithm": "QR-anchor-gated residual transfer",
        "data_source": "Stanford Gset",
        "training_sources": list(TRAIN_SOURCES),
        "development_sources": list(DEVELOPMENT_SOURCES),
        "confirmation_sources": list(CONFIRMATION_SOURCES),
        "tasks_per_source": TASKS_PER_SOURCE,
        "task_vertices": TASK_VERTICES,
        "candidate_count": int(CANDIDATES.shape[0]),
        "primary_rank": PRIMARY_RANK,
        "salience_temperature": SALIENCE_TEMPERATURE,
        "candidates": CANDIDATES,
        "source_manifest": source_rows,
        "temperature_sweep": tau_rows,
        "rank_sweep": rank_rows,
        "split_comparison": split_rows,
        "primary_comparison": primary_rows,
        "confirmation_by_source": per_source_rows,
        "singular_values": singular_rows,
        "primary_model": primary_model,
        "maximum_formula_error": max(record.formula_error for record in records),
        "hypotheses": hypotheses,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }
    return _jsonable(result)


def write_outputs(result: dict[str, object], output_dir: Path) -> None:
    """Write locked JSON and tabular evidence."""

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "source_manifest.csv": result["source_manifest"],
        "temperature_sweep.csv": result["temperature_sweep"],
        "rank_sweep.csv": result["rank_sweep"],
        "split_comparison.csv": result["split_comparison"],
        "primary_comparison.csv": result["primary_comparison"],
        "confirmation_by_source.csv": result["confirmation_by_source"],
        "singular_values.csv": result["singular_values"],
    }
    for name, rows in mapping.items():
        _write_csv(output_dir / name, list(rows))
    (output_dir / "locked_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("code/results"))
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("code/manuscript_assets/figures"),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="recompute deterministic evidence and compare it with the locked JSON",
    )
    return parser


def _compare_locked(observed, expected, path: str = "root") -> None:
    """Recursively compare a replay with the locked scientific record."""

    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise RuntimeError(f"locked replay key mismatch at {path}")
        for key in expected:
            _compare_locked(observed[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise RuntimeError(f"locked replay length mismatch at {path}")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _compare_locked(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if not math.isclose(float(observed), expected, rel_tol=2.0e-12, abs_tol=2.0e-13):
            raise RuntimeError(
                f"locked replay numeric mismatch at {path}: {observed} != {expected}"
            )
        return
    if observed != expected:
        raise RuntimeError(f"locked replay mismatch at {path}: {observed!r} != {expected!r}")


def main() -> None:
    args = build_parser().parse_args()
    result = run_study(args.cache_dir)
    if args.verify:
        locked_path = args.output_dir / "locked_results.json"
        locked = json.loads(locked_path.read_text(encoding="utf-8"))
        # Environment strings are provenance rather than scientific results.
        observed_environment = result.pop("environment")
        locked_environment = locked.pop("environment")
        _compare_locked(result, locked)
        print(
            json.dumps(
                {
                    "locked_replay": "PASS",
                    "current_environment": observed_environment,
                    "locked_environment": locked_environment,
                    "hypotheses": result["hypotheses"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    write_outputs(result, args.output_dir)
    from .figures import make_all

    figure_paths = make_all(args.output_dir / "locked_results.json", args.figure_dir)
    tracked = [
        args.output_dir / name
        for name in (
            "source_manifest.csv",
            "temperature_sweep.csv",
            "rank_sweep.csv",
            "split_comparison.csv",
            "primary_comparison.csv",
            "confirmation_by_source.csv",
            "singular_values.csv",
            "locked_results.json",
            "runtime.json",
            "runtime.csv",
        )
    ] + figure_paths
    repository = Path.cwd().resolve()
    manifest = {
        "schema_version": 3,
        "files": {
            str(path.resolve().relative_to(repository))
            if path.resolve().is_relative_to(repository)
            else path.name: {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in tracked
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"hypotheses": result["hypotheses"]}, indent=2, sort_keys=True))
    if not all(result["hypotheses"].values()):
        raise SystemExit("one or more release hypotheses failed")


if __name__ == "__main__":
    main()
