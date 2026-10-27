from __future__ import annotations
import pandas as pd

REQUIRED_BAR_COLUMNS = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "available_at"}

class DataContractError(ValueError):
    """Raised when market data violates the canonical contract."""


def validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate canonical bars without silently sorting, deduplicating, or imputing them."""
    missing = REQUIRED_BAR_COLUMNS - set(frame.columns)
    if missing:
        raise DataContractError(f"Missing required columns: {sorted(missing)}")
    out = frame.copy()
    for col in ("timestamp", "available_at"):
        out[col] = pd.to_datetime(out[col], utc=True, errors="raise")
    if out[["timestamp", "symbol"]].duplicated().any():
        raise DataContractError("Duplicate (timestamp, symbol) bars")
    if (out["available_at"] < out["timestamp"]).any():
        raise DataContractError("Bar cannot be available before its event/end timestamp")
    numeric = ["open", "high", "low", "close", "volume"]
    if out[numeric].isna().any().any():
        raise DataContractError("Null OHLCV values; repair upstream with an auditable policy")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataContractError("OHLC prices must be positive for this spot-style contract")
    if (out["volume"] < 0).any():
        raise DataContractError("Volume cannot be negative")
    bad = (out["high"] < out[["open", "close", "low"]].max(axis=1)) | (out["low"] > out[["open", "close", "high"]].min(axis=1))
    if bad.any():
        raise DataContractError("Invalid OHLC envelope")
    return out.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def assert_point_in_time(records: pd.DataFrame, decision_time: pd.Series | pd.Timestamp,
                         available_col: str = "available_at") -> None:
    """Fail if any selected record was not available at the associated decision time."""
    available = pd.to_datetime(records[available_col], utc=True)
    decision = pd.to_datetime(decision_time, utc=True)
    if (available > decision).any():
        raise DataContractError("Point-in-time violation: available_at > decision_timestamp")


def asof_join(decisions: pd.DataFrame, records: pd.DataFrame, *, by: str,
              decision_col: str = "decision_timestamp", available_col: str = "available_at") -> pd.DataFrame:
    """Attach the latest known record per key. Uses availability time, not observation date."""
    left, right = decisions.copy(), records.copy()
    left[decision_col] = pd.to_datetime(left[decision_col], utc=True)
    right[available_col] = pd.to_datetime(right[available_col], utc=True)
    if by not in left or by not in right:
        raise DataContractError(f"Join key {by!r} missing")
    chunks = []
    # Per-key as-of prevents cross-asset matching. Keep stable ordering for reproducibility.
    for key, lgrp in left.groupby(by, sort=False, dropna=False):
        rgrp = right[right[by].eq(key)].sort_values(available_col)
        lgrp = lgrp.sort_values(decision_col)
        if rgrp.empty:
            lgrp = lgrp.copy()
            for c in right.columns:
                if c not in lgrp.columns and c != by:
                    lgrp[c] = pd.NA
            chunks.append(lgrp)
            continue
        merged = pd.merge_asof(lgrp, rgrp.drop(columns=[by]), left_on=decision_col,
                               right_on=available_col, direction="backward", allow_exact_matches=True)
        chunks.append(merged)
    result = pd.concat(chunks, ignore_index=True).sort_values(decision_col, kind="stable")
    if available_col in result:
        known = result[available_col].notna()
        if (result.loc[known, available_col] > result.loc[known, decision_col]).any():
            raise DataContractError("As-of join created future data")
    return result.reset_index(drop=True)
