"""Errors raised by experiment specification and planning."""

from __future__ import annotations


class ExperimentError(ValueError):
    """Raised when an experiment specification or fold plan is invalid."""


class ExperimentIntegrityError(ExperimentError):
    """Raised when a dataset, specification, or plan binding is inconsistent."""


class PredictionCoverageError(ExperimentError):
    """Raised when predictions do not exactly cover a fold's test keys."""
