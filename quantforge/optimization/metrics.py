from __future__ import annotations
import numpy as np
from scipy.stats import norm, skew, kurtosis


def sharpe_ratio(returns, periods_per_year: float = 252.0) -> float:
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.std(x, ddof=1) == 0: return float("nan")
    return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(periods_per_year))


def deflated_sharpe_probability(returns, *, trials: int, periods_per_year: float = 252.0) -> float:
    """Approximate DSR probability that observed Sharpe exceeds a multiple-testing benchmark.

    Uses the expected maximum of N approximately Gaussian trial Sharpes. This is a diagnostic,
    not a universal correction: dependence among trials and nonstationarity require care.
    Returns probability in [0,1]; does not convert it into a claim of profitability.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if trials < 1: raise ValueError("trials must be >= 1")
    if n < 3: return float("nan")
    sr = np.mean(x) / np.std(x, ddof=1) * np.sqrt(periods_per_year)
    # Expected maximum of N standard normal variates, finite-sample approximation.
    euler_gamma = 0.5772156649
    if trials == 1: benchmark_z = 0.0
    else:
        a = norm.ppf(1 - 1 / trials)
        b = norm.ppf(1 - 1 / (trials * np.e))
        benchmark_z = (1 - euler_gamma) * a + euler_gamma * b
    # Sharpe standard error approximation including skewness and excess kurtosis.
    sr_unannual = np.mean(x) / np.std(x, ddof=1)
    s = float(skew(x, bias=False))
    k = float(kurtosis(x, fisher=True, bias=False))
    variance = max(1e-12, (1 - s * sr_unannual + ((k + 2) / 4) * sr_unannual**2) / (n - 1))
    # Trials benchmark is expressed in standard-normal units; translate by Sharpe SE.
    benchmark_sr = benchmark_z * np.sqrt(variance) * np.sqrt(periods_per_year)
    return float(norm.cdf((sr - benchmark_sr) / (np.sqrt(variance) * np.sqrt(periods_per_year))))
