"""Validation for user-supplied new demand rows (the "add new data" path)."""
from __future__ import annotations

import pandas as pd

from src.simulate import DATA_COLUMNS

REQUIRED_UPLOAD = ["date", "demand", "promo", "discount", "holiday", "temp_fcst"]


def validate_new_rows(new: pd.DataFrame, last_date: pd.Timestamp, max_rows: int) -> pd.DataFrame:
    """Return the rows in DATA_COLUMNS order, or raise ValueError with a readable reason."""
    missing = [c for c in REQUIRED_UPLOAD if c not in new.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if new.empty:
        raise ValueError("the file has no rows")
    new = new.copy()
    try:
        new["date"] = pd.to_datetime(new["date"])
        for c in REQUIRED_UPLOAD[1:] + (["temp"] if "temp" in new else []):
            new[c] = pd.to_numeric(new[c])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"could not read a value ({exc})") from None
    new = new.sort_values("date").reset_index(drop=True)
    expected = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(new), freq="D")
    if not (new["date"].to_numpy() == expected.to_numpy()).all():
        raise ValueError(f"dates must be consecutive days starting {expected[0].date()}")
    if new[REQUIRED_UPLOAD[1:]].isna().any().any():
        raise ValueError("every row needs demand, promo, discount, holiday and temp_fcst")
    if (new["demand"] < 0).any():
        raise ValueError("demand cannot be negative")
    if not new["promo"].isin([0, 1]).all() or not new["holiday"].isin([0, 1]).all():
        raise ValueError("promo and holiday must be 0 or 1")
    if len(new) > max_rows:
        raise ValueError(f"at most {max_rows} more days fit in the simulated planning horizon")
    if "temp" not in new:
        new["temp"] = new["temp_fcst"]
    return new[DATA_COLUMNS]
