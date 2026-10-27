from dataclasses import dataclass
import math

@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.5
    half_spread_bps: float = 1.0
    impact_coefficient: float = 0.1
    volatility: float = 0.01
    adv_notional: float = 1_000_000.0

    def __post_init__(self):
        if min(self.commission_bps, self.half_spread_bps, self.impact_coefficient, self.volatility) < 0:
            raise ValueError("cost parameters cannot be negative")
        if self.adv_notional <= 0: raise ValueError("adv_notional must be positive")

    def cost_fraction(self, notional: float) -> float:
        """Approximate one-way fractional cost; impact follows a square-root law.

        Calibrate to instrument/venue/order-size data. This is not a substitute for L2 replay.
        """
        if notional < 0: raise ValueError("notional cannot be negative")
        participation = min(notional / self.adv_notional, 1.0)
        impact_bps = self.impact_coefficient * self.volatility * 10_000 * math.sqrt(participation)
        return (self.commission_bps + self.half_spread_bps + impact_bps) / 10_000
