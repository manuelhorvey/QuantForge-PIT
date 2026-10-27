from __future__ import annotations
import pandas as pd


def signal_to_target_weights(signals: pd.Series, *, gross_limit: float = 1.0,
                             per_asset_limit: float = 0.2) -> pd.Series:
    """Normalize signed signals into target weights; not a full covariance optimizer."""
    if gross_limit <= 0 or per_asset_limit <= 0: raise ValueError("limits must be positive")
    x = signals.astype(float).fillna(0).clip(-1, 1)
    x = x.clip(-per_asset_limit, per_asset_limit)
    gross = x.abs().sum()
    if gross > gross_limit: x = x * (gross_limit / gross)
    return x.rename("target_weight")
