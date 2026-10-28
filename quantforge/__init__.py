"""QuantForge point-in-time trading research foundation."""

from .dataset import (
    ResearchDataset,
    build_dataset,
    canonical_json,
    hash_frame,
    hash_payload,
    validate_dataset,
)
from .features import (
    FeatureResult,
    FeatureSpec,
    compute_features,
    fracdiff_spec,
    momentum_spec,
    realized_volatility_spec,
    rolling_zscore_spec,
)
from .labels import LabelResult, LabelSpec, compute_labels
from .models import TimeSplitResult, purged_split, purged_time_split

__version__ = "0.1.0"

__all__ = [
    "FeatureResult",
    "FeatureSpec",
    "LabelResult",
    "LabelSpec",
    "ResearchDataset",
    "TimeSplitResult",
    "build_dataset",
    "canonical_json",
    "compute_features",
    "compute_labels",
    "fracdiff_spec",
    "hash_frame",
    "hash_payload",
    "momentum_spec",
    "purged_split",
    "purged_time_split",
    "realized_volatility_spec",
    "rolling_zscore_spec",
    "validate_dataset",
]
