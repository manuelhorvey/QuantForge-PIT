from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .costs import CostModel

@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    max_gross_leverage: float = 1.0
    trade_on_next_bar: bool = True
    position_fraction: float = 0.1

    def __post_init__(self):
        if self.initial_cash <= 0 or self.max_gross_leverage <= 0: raise ValueError("capital/leverage must be positive")
        if not 0 <= self.position_fraction <= 1: raise ValueError("position_fraction must be in [0,1]")


def run_backtest(bars: pd.DataFrame, signal: pd.Series, *, config: BacktestConfig = BacktestConfig(),
                 costs: CostModel = CostModel()) -> pd.DataFrame:
    """Single-asset close-to-close reference simulator.

    `signal[t]` is assumed known after bar t is complete; by default position is applied to
    the following bar's return. Signal must be indexed exactly like bars. No leverage funding,
    borrow, market closures, partial fills, stop execution, or margin-liquidation logic is implied.
    """
    if len(bars) != len(signal) or not bars.index.equals(signal.index):
        raise ValueError("bars and signal must have identical indices and lengths")
    if "close" not in bars or (bars.close <= 0).any(): raise ValueError("bars require positive close")
    sig = signal.astype(float).clip(-1, 1)
    if sig.isna().any(): raise ValueError("signals must be explicit; fill/abstention policy belongs upstream")
    position = sig.shift(1, fill_value=0.0) if config.trade_on_next_bar else sig
    position *= config.position_fraction
    gross = bars.close.pct_change(fill_method=None).fillna(0.0) * position
    turnover = position.diff().abs().fillna(position.abs())
    # Use prior close as a rough notional base. Costs are applied to turnover, not returns.
    notional = (turnover * config.initial_cash).clip(lower=0)
    cost = notional.map(costs.cost_fraction)
    net = gross - cost
    equity = config.initial_cash * (1 + net).cumprod()
    return pd.DataFrame({"signal": sig, "position": position, "gross_return": gross,
                         "turnover": turnover, "cost_return": cost, "net_return": net,
                         "equity": equity}, index=bars.index)
