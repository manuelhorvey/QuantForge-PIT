from __future__ import annotations
import numpy as np
import pandas as pd


def fracdiff_ffd(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """Fixed-width fractional differentiation; output at t uses only x[t-k..t].

    Weight truncation threshold is a modelling choice and must be fixed before OOS testing.
    NaNs are preserved at the start until a complete finite window is available.
    """
    if not 0 <= d <= 1:
        raise ValueError("d must be in [0, 1]")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    weights = [1.0]
    for k in range(1, len(series)):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    w = np.asarray(weights[::-1], dtype=float)
    arr = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    width = len(w)
    for i in range(width - 1, len(arr)):
        window = arr[i - width + 1:i + 1]
        if np.isfinite(window).all():
            out[i] = float(np.dot(w, window))
    return pd.Series(out, index=series.index, name=f"{series.name or 'x'}_fd{d:g}")
