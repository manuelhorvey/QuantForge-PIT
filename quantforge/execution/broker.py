from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Literal

@dataclass(frozen=True)
class Order:
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: float
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: float | None = None
    client_order_id: str = ""

    def __post_init__(self):
        if not self.symbol.strip() or self.quantity <= 0: raise ValueError("symbol and positive quantity required")
        if self.order_type == "LIMIT" and (self.limit_price is None or self.limit_price <= 0):
            raise ValueError("limit order requires positive limit_price")

class Broker(Protocol):
    def submit_order(self, order: Order) -> str: ...
    def cancel_order(self, broker_order_id: str) -> bool: ...
    def positions(self) -> dict[str, float]: ...
    def account_state(self) -> dict[str, float]: ...

class DryRunBroker:
    """Safe adapter for integration tests; never sends external orders."""
    def __init__(self): self.orders: list[Order] = []
    def submit_order(self, order: Order) -> str:
        self.orders.append(order)
        return f"DRY-{len(self.orders):08d}"
    def cancel_order(self, broker_order_id: str) -> bool: return broker_order_id.startswith("DRY-")
    def positions(self) -> dict[str, float]: return {}
    def account_state(self) -> dict[str, float]: return {"equity": 0.0, "available_margin": 0.0}
