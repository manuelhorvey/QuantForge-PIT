from __future__ import annotations
import pandas as pd


def momentum(close: pd.Series, lookback: int = 20) -> pd.Series:
    if lookback < 1: raise ValueError("lookback must be >= 1")
    return close.pct_change(lookback, fill_method=None).rename(f"mom_{lookback}")


def realized_volatility(close: pd.Series, window: int = 20, annualization: float = 1.0) -> pd.Series:
    if window < 2 or annualization <= 0: raise ValueError("invalid window/annualization")
    returns = close.pct_change(fill_method=None)
    return (returns.rolling(window, min_periods=window).std(ddof=1) * annualization**0.5).rename(f"rv_{window}")


def rolling_zscore(series: pd.Series, window: int = 50) -> pd.Series:
    if window < 2: raise ValueError("window must be >= 2")
    roll = series.rolling(window, min_periods=window)
    std = roll.std(ddof=1).replace(0, float("nan"))
    return ((series - roll.mean()) / std).rename(f"z_{window}")
