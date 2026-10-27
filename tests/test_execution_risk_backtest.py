import pandas as pd
import pytest
from quantforge.backtest import run_backtest, BacktestConfig, CostModel
from quantforge.execution import Order, DryRunBroker
from quantforge.governance.risk import RiskLimits, pretrade_check, RiskRejected


def test_backtest_applies_signal_next_bar():
    idx = pd.date_range("2024-01-01", periods=3, tz="UTC")
    bars = pd.DataFrame({"close": [100., 110., 121.]}, index=idx)
    sig = pd.Series([1., 1., 1.], index=idx)
    out = run_backtest(bars, sig, config=BacktestConfig(position_fraction=1), costs=CostModel())
    assert out.position.iloc[0] == 0
    assert out.position.iloc[1] == 1


def test_order_contract_and_dry_run():
    broker = DryRunBroker()
    oid = broker.submit_order(Order("ABC", "BUY", 2))
    assert oid.startswith("DRY-") and len(broker.orders) == 1
    with pytest.raises(ValueError): Order("ABC", "BUY", 0)


def test_risk_gate_fails_closed():
    with pytest.raises(RiskRejected):
        pretrade_check(order_notional=20_000, projected_gross_leverage=.5, daily_pnl_fraction=0,
                       drawdown_fraction=0, limits=RiskLimits(max_order_notional=10_000))
