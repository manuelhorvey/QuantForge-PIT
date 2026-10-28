"""Experiment specification, fold planning, and out-of-sample evaluation."""

from .errors import (
    ExperimentError,
    ExperimentIntegrityError,
    PredictionCoverageError,
)
from .evaluation import (
    ExperimentResult,
    FoldResult,
    evaluate_experiment,
    evaluate_fold,
)
from .folds import FoldPlan, PurgedFold, WalkForwardSpec, plan_folds
from .spec import ExperimentPlan, ExperimentSpec, plan_experiment

__all__ = [
    "ExperimentError",
    "ExperimentIntegrityError",
    "ExperimentPlan",
    "ExperimentResult",
    "ExperimentSpec",
    "FoldPlan",
    "FoldResult",
    "PredictionCoverageError",
    "PurgedFold",
    "WalkForwardSpec",
    "evaluate_experiment",
    "evaluate_fold",
    "plan_experiment",
    "plan_folds",
]
