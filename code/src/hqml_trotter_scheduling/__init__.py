"""Low-rank residual matrix learning for graph quantum optimization."""

from .experiment import (
    CANDIDATES,
    PRIMARY_RANK,
    SALIENCE_TEMPERATURE,
    fit_interpolation,
    metric_record,
    nearest_anchor_prediction,
    p1_formula,
    pivot_columns,
    run_study,
    salience_weights,
    zero_query_prediction,
)

__all__ = [
    "CANDIDATES",
    "PRIMARY_RANK",
    "SALIENCE_TEMPERATURE",
    "fit_interpolation",
    "metric_record",
    "nearest_anchor_prediction",
    "p1_formula",
    "pivot_columns",
    "run_study",
    "salience_weights",
    "zero_query_prediction",
]

__version__ = "4.0.0"
