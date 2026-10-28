"""Declarative, point-in-time feature specifications."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Callable, Mapping

import pandas as pd

from .factors import momentum, realized_volatility, rolling_zscore
from .fractional import fracdiff_ffd

FeatureFunction = Callable[[pd.DataFrame], Any]

_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RESERVED_SUFFIXES = (
    "__available_at",
    "__source_max_available_at",
    "__source",
    "__revision",
    "__status",
    "__reason",
)
GLOBAL_METADATA_COLUMNS = (
    "feature_available_at",
    "feature_source_max_available_at",
    "feature_source",
    "feature_revision",
    "feature_status",
    "feature_reason",
)


@dataclass(frozen=True)
class FeatureSpec:
    """A feature definition evaluated only against PIT-eligible history."""

    name: str
    inputs: tuple[str, ...]
    compute: FeatureFunction
    window: int | None = None
    min_periods: int | None = None
    lag: int = 0
    version: str = "1"
    params: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not _NAME_PATTERN.match(self.name):
            raise ValueError("feature name must be a simple identifier")
        if any(self.name.endswith(suffix) for suffix in RESERVED_SUFFIXES):
            raise ValueError("feature name uses a reserved metadata suffix")
        if self.name in GLOBAL_METADATA_COLUMNS:
            raise ValueError("feature name collides with aggregate metadata")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if not self.inputs:
            raise ValueError("feature inputs cannot be empty")
        if any(not isinstance(name, str) or not name for name in self.inputs):
            raise ValueError("feature inputs must be non-empty strings")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("feature inputs must be unique")
        if not callable(self.compute):
            raise ValueError("compute must be callable")
        if self.window is not None and (
            not isinstance(self.window, Integral)
            or isinstance(self.window, bool)
            or self.window < 1
        ):
            raise ValueError("window must be a positive integer")
        if self.min_periods is not None and (
            not isinstance(self.min_periods, Integral)
            or isinstance(self.min_periods, bool)
            or self.min_periods < 0
        ):
            raise ValueError("min_periods must be a non-negative integer")
        if (
            self.window is not None
            and self.min_periods is not None
            and self.min_periods > self.window
        ):
            raise ValueError("min_periods cannot exceed window")
        if (
            not isinstance(self.lag, Integral)
            or isinstance(self.lag, bool)
            or self.lag < 0
        ):
            raise ValueError("lag must be a non-negative integer")
        if not self.version:
            raise ValueError("version is required")
        object.__setattr__(self, "params", dict(self.params))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "window": self.window,
            "min_periods": self.min_periods,
            "lag": self.lag,
            "version": self.version,
            "params": dict(self.params),
            "description": self.description,
        }


def momentum_spec(lookback: int = 20, *, name: str | None = None) -> FeatureSpec:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    feature_name = name or f"momentum_{lookback}"

    def compute(frame: pd.DataFrame) -> float:
        return float(momentum(frame["close"], lookback).iloc[-1])

    return FeatureSpec(
        name=feature_name,
        inputs=("close",),
        compute=compute,
        window=lookback + 1,
        min_periods=lookback + 1,
        params={"lookback": lookback},
        description="Trailing simple return over the supplied close series.",
    )


def realized_volatility_spec(
    window: int = 20,
    *,
    annualization: float = 1.0,
    name: str | None = None,
) -> FeatureSpec:
    if window < 2 or annualization <= 0:
        raise ValueError("invalid window/annualization")
    feature_name = name or f"realized_volatility_{window}"

    def compute(frame: pd.DataFrame) -> float:
        return float(
            realized_volatility(
                frame["close"], window, annualization=annualization
            ).iloc[-1]
        )

    return FeatureSpec(
        name=feature_name,
        inputs=("close",),
        compute=compute,
        window=window + 1,
        min_periods=window + 1,
        params={"window": window, "annualization": annualization},
        description="Trailing close-to-close volatility.",
    )


def rolling_zscore_spec(window: int = 50, *, name: str | None = None) -> FeatureSpec:
    if window < 2:
        raise ValueError("window must be >= 2")
    feature_name = name or f"rolling_zscore_{window}"

    def compute(frame: pd.DataFrame) -> float:
        return float(rolling_zscore(frame["close"], window).iloc[-1])

    return FeatureSpec(
        name=feature_name,
        inputs=("close",),
        compute=compute,
        window=window,
        min_periods=window,
        params={"window": window},
        description="Trailing z-score of close.",
    )


def _identifier_token(value: float) -> str:
    """Return a deterministic, identifier-safe token preserving the float value."""

    token = repr(float(value))
    return token.replace("-", "m").replace("+", "p").replace(".", "_")


def fracdiff_spec(
    d: float = 0.5,
    *,
    threshold: float = 1e-5,
    name: str | None = None,
) -> FeatureSpec:
    if not isinstance(d, Real) or isinstance(d, bool):
        raise ValueError("d must be a real number in [0, 1]")
    if not isinstance(threshold, Real) or isinstance(threshold, bool):
        raise ValueError("threshold must be a real number")
    d = float(d)
    threshold = float(threshold)
    if not 0 <= d <= 1:
        raise ValueError("d must be in [0, 1]")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a positive finite number")
    feature_name = name or (
        f"fracdiff_{_identifier_token(d)}_threshold_{_identifier_token(threshold)}"
    )

    def compute(frame: pd.DataFrame) -> float:
        return float(fracdiff_ffd(frame["close"], d, threshold=threshold).iloc[-1])

    return FeatureSpec(
        name=feature_name,
        inputs=("close",),
        compute=compute,
        min_periods=1,
        params={"d": d, "threshold": threshold},
        description="Causal fixed-width fractional differentiation.",
    )
