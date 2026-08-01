"""Network-free tests for low-rank residual interpolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hqml_trotter_scheduling.experiment import (
    CONFIRMATION_SOURCES,
    DEVELOPMENT_SOURCES,
    SOURCE_HASHES,
    TRAIN_SOURCES,
    cost_diagonal,
    edge_statistics,
    fit_interpolation,
    metric_record,
    p1_formula,
    pivot_columns,
    qaoa_energy,
    salience_weights,
    sha256,
    nearest_anchor_prediction,
    zero_query_prediction,
)


def test_depth_one_formula_matches_state_vector_on_triangle():
    edges = ((0, 1), (1, 2), (0, 2))
    diagonal = cost_diagonal(edges, vertices=3)
    statistics = edge_statistics(edges, vertices=3)
    for gamma in (0.19, 0.73, 1.41):
        for beta in (0.08, 0.31, 0.69):
            exact = qaoa_energy(
                diagonal, (gamma, beta, 0.0, 0.0), depth=1
            )
            np.testing.assert_allclose(
                p1_formula(statistics, gamma, beta), exact, atol=2.0e-14
            )


def test_interpolation_recovers_rank_three_matrix_from_three_anchors():
    rng = np.random.default_rng(7)
    left = rng.standard_normal((18, 3))
    right = rng.standard_normal((3, 25))
    matrix = left @ right
    anchors = pivot_columns(matrix, 3, np.ones(matrix.shape[1]))
    interpolation = fit_interpolation(matrix, anchors)
    reconstructed = matrix[:, anchors] @ interpolation
    np.testing.assert_allclose(reconstructed, matrix, atol=2.0e-12)


def test_salience_concentrates_on_consistently_large_columns():
    surfaces = np.asarray(
        [[0.0, 1.0, 0.2], [0.1, 0.9, 0.0], [0.0, 1.1, 0.3]]
    )
    weights = salience_weights(surfaces, temperature=20.0)
    assert weights.shape == (3,)
    assert weights[1] > weights[2] > weights[0]
    np.testing.assert_allclose(np.mean(weights), 1.0)


def test_regret_bound_is_covered_for_arbitrary_predictions():
    truth = np.asarray([[0.1, 0.8, 0.6], [0.9, 0.2, 0.3]])
    prediction = np.asarray([[0.4, 0.3, 0.6], [0.2, 0.7, 0.4]])
    metrics = metric_record(prediction, truth)
    assert metrics["regret_bound_covered"] == 1.0
    assert metrics["minimum_regret_bound_slack"] >= -1.0e-14


def test_registered_sources_have_complete_digests():
    assert len(SOURCE_HASHES) == 16
    assert all(len(value) == 64 for value in SOURCE_HASHES.values())


def test_sha256_detects_modified_source(tmp_path: Path):
    source = tmp_path / "source"
    source.write_text("3 2\n1 2 1\n2 3 1\n", encoding="utf-8")
    first = sha256(source)
    source.write_text("3 1\n1 3 1\n", encoding="utf-8")
    assert sha256(source) != first


def test_pivot_rejects_nonpositive_salience():
    with pytest.raises(ValueError, match="positive"):
        pivot_columns(np.eye(3), 2, np.asarray([1.0, 0.0, 1.0]))


def test_source_splits_are_pairwise_disjoint():
    assert set(TRAIN_SOURCES).isdisjoint(DEVELOPMENT_SOURCES)
    assert set(TRAIN_SOURCES).isdisjoint(CONFIRMATION_SOURCES)
    assert set(DEVELOPMENT_SOURCES).isdisjoint(CONFIRMATION_SOURCES)
    assert set(TRAIN_SOURCES + DEVELOPMENT_SOURCES + CONFIRMATION_SOURCES) == set(
        SOURCE_HASHES
    )


def test_zero_query_nearest_transfers_only_training_residual():
    train_b = np.asarray([[0.0, 0.2, 0.4], [1.0, 1.2, 1.4]])
    train_y = train_b + np.asarray([[0.3, 0.2, 0.1], [-0.2, 0.0, 0.2]])
    target_b = np.asarray([[0.01, 0.19, 0.39]])
    prediction = zero_query_prediction(
        "nearest_p1_residual", train_y, train_b, target_b
    )
    np.testing.assert_allclose(prediction, target_b + (train_y - train_b)[[0]])


def test_nearest_anchor_uses_only_selected_target_entries():
    train_b = np.zeros((2, 4))
    train_y = np.asarray([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])
    target_b = np.zeros((1, 4))
    target_y = np.asarray([[100.0, 1.1, -100.0, 2.9]])
    prediction = nearest_anchor_prediction(
        train_y, train_b, target_y, target_b, np.asarray([1, 3])
    )
    np.testing.assert_allclose(prediction, train_y[[0]])


def test_interpolation_off_subspace_infinity_bound():
    rng = np.random.default_rng(91)
    matrix = rng.standard_normal((12, 4)) @ rng.standard_normal((4, 19))
    anchors = pivot_columns(matrix, 4, np.ones(19))
    interpolation = fit_interpolation(matrix, anchors)
    fixed = rng.standard_normal(12) @ matrix
    target = fixed + 1.0e-3 * rng.standard_normal(19)
    reconstruction = target[anchors] @ interpolation
    left = np.linalg.norm(target - reconstruction, ord=np.inf)
    right = (1.0 + np.linalg.norm(interpolation, ord=1)) * np.linalg.norm(
        target - fixed, ord=np.inf
    )
    assert left <= right + 1.0e-13


def test_locked_record_retains_strong_zero_query_and_linear_comparators():
    path = Path(__file__).resolve().parents[1] / "results" / "locked_results.json"
    import json

    result = json.loads(path.read_text(encoding="utf-8"))
    confirmation = {
        row["method"]: row
        for row in result["split_comparison"]
        if row["split"] == "confirmation"
    }
    assert confirmation["nearest_p1_residual"]["queries"] == 0
    assert confirmation["nearest_p1_residual"]["mean_grid_regret"] == 0.0
    assert confirmation["anchor_gated_residual"]["queries"] == 4
    assert confirmation["anchor_gated_residual"]["mean_grid_regret"] == 0.0
    assert confirmation["anchor_gated_residual"]["surface_rmse"] < min(
        confirmation[name]["surface_rmse"]
        for name in (
            "nearest_p1_residual",
            "salience_residual",
            "uniform_residual",
            "salience_raw",
        )
    )
