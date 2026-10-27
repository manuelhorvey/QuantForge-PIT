from dataclasses import dataclass

class RiskRejected(RuntimeError): pass

@dataclass(frozen=True)
class RiskLimits:
    max_gross_leverage: float = 1.0
    max_order_notional: float = 10_000.0
    max_daily_loss_fraction: float = 0.02
    max_drawdown_fraction: float = 0.10

    def __post_init__(self):
        if min(self.max_gross_leverage, self.max_order_notional, self.max_daily_loss_fraction, self.max_drawdown_fraction) <= 0:
            raise ValueError("risk limits must be positive")
        if self.max_daily_loss_fraction >= 1 or self.max_drawdown_fraction >= 1:
            raise ValueError("loss fractions must be below 1")


def pretrade_check(*, order_notional: float, projected_gross_leverage: float,
                   daily_pnl_fraction: float, drawdown_fraction: float,
                   limits: RiskLimits) -> None:
    """Fail-closed pretrade check. Fractions are losses as positive magnitudes."""
    values = (order_notional, projected_gross_leverage, daily_pnl_fraction, drawdown_fraction)
    if not all(map(lambda x: isinstance(x, (int, float)), values)): raise RiskRejected("non-numeric risk input")
    if order_notional < 0 or projected_gross_leverage < 0: raise RiskRejected("negative exposure input")
    if order_notional > limits.max_order_notional: raise RiskRejected("order notional limit")
    if projected_gross_leverage > limits.max_gross_leverage: raise RiskRejected("gross leverage limit")
    if daily_pnl_fraction <= -limits.max_daily_loss_fraction: raise RiskRejected("daily loss limit")
    if drawdown_fraction >= limits.max_drawdown_fraction: raise RiskRejected("drawdown limit")
