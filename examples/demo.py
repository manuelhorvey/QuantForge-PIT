import numpy as np
import pandas as pd
from quantforge.features import momentum
from quantforge.backtest import run_backtest, BacktestConfig, CostModel

rng = np.random.default_rng(7)
idx = pd.date_range("2020-01-01", periods=1000, freq="B", tz="UTC")
close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx))))
bars = pd.DataFrame({"close": close}, index=idx)
sig = np.sign(momentum(bars.close, 20).fillna(0))
result = run_backtest(bars, sig, config=BacktestConfig(position_fraction=0.25),
                      costs=CostModel(commission_bps=0.5, half_spread_bps=1.0,
                                      impact_coefficient=0.1, volatility=0.01, adv_notional=1_000_000))
print(result.tail())
print("Demo only; synthetic data and uncalibrated costs are not evidence of a trading edge.")
