import pandas as pd
import pytest
from quantforge.data.contracts import validate_bars, assert_point_in_time, asof_join, DataContractError


def sample():
    return pd.DataFrame({"timestamp": ["2024-01-01T00:01:00Z"], "symbol": ["ABC"],
        "open": [10.], "high": [11.], "low": [9.], "close": [10.5], "volume": [100.],
        "available_at": ["2024-01-01T00:01:01Z"]})


def test_validate_bars_normalizes_time():
    out = validate_bars(sample())
    assert str(out.timestamp.dtype) == "datetime64[ns, UTC]"


def test_reject_duplicate_bars():
    df = pd.concat([sample(), sample()], ignore_index=True)
    with pytest.raises(DataContractError): validate_bars(df)


def test_reject_future_availability_in_decision():
    df = sample()
    with pytest.raises(DataContractError):
        assert_point_in_time(df, pd.Timestamp("2024-01-01T00:01:00Z"))


def test_asof_join_never_uses_future_record():
    decisions = pd.DataFrame({"symbol": ["ABC"], "decision_timestamp": ["2024-01-01T00:00:30Z"]})
    records = pd.DataFrame({"symbol": ["ABC"], "available_at": ["2024-01-01T00:01:00Z"], "value": [7]})
    out = asof_join(decisions, records, by="symbol")
    assert pd.isna(out.loc[0, "value"])
