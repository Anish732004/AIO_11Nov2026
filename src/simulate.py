"""Synthetic daily demand for one SKU.

Demand is built multiplicatively from interpretable drivers:

    expected = base x trend x annual x weekly x temperature x promo x post_promo x holiday
    demand   = expected + noise   (noise sd proportional to expected -> heteroscedastic)

We generate a "world" longer than the history we show. The extra days are the future:
the dashboard reveals them a few at a time to mimic new data arriving.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Fixed-date holidays (month-day). Planned/known in advance, so usable as forecast inputs.
HOLIDAYS = {
    "01-01": "New Year",
    "01-26": "Republic Day",
    "08-15": "Independence Day",
    "10-02": "Gandhi Jayanti",
    "11-01": "Diwali (approx.)",
    "12-25": "Christmas",
}

# Columns a user-supplied CSV (or the simulator) provides. `temp` is the actual temperature,
# only known after the fact; `temp_fcst` is the weather forecast that was available beforehand.
DATA_COLUMNS = ["date", "demand", "promo", "discount", "holiday", "temp", "temp_fcst"]


@dataclass(frozen=True)
class SimConfig:
    base: float = 100.0                 # units/day at the start, before multipliers
    trend_per_year: float = 0.15        # +15 % per year, linear
    annual_amp: float = 0.20            # +/-20 % annual swing ...
    annual_peak_doy: int = 300          # ... peaking in the festive season (late Oct)
    weekly: tuple = (0.90, 0.90, 0.95, 1.00, 1.10, 1.25, 1.15)  # Mon..Sun
    temp_mean: float = 27.0             # deg C
    temp_amp: float = 6.0
    temp_peak_doy: int = 135            # mid-May: summer peak, deliberately not the demand peak
    temp_ar_phi: float = 0.8            # persistence of hot/cold spells
    temp_ar_sd: float = 1.5
    temp_sensitivity: float = 0.01      # +1 % demand per deg C above mean (small and honest)
    temp_forecast_sd: float = 2.0       # weather-forecast error, deg C
    promo_gap: tuple = (15, 40)         # days between campaigns
    promo_len: tuple = (3, 7)           # campaign length, days
    promo_discount: tuple = (0.10, 0.30)
    promo_lift_per_discount: float = 2.0  # 20 % off -> +40 % demand
    post_promo_days: int = 3            # pull-forward dip after a campaign ...
    post_promo_dip: float = 0.12        # ... of -12 %
    holiday_lift: float = 0.35
    noise_cv: float = 0.10              # noise sd = 10 % of expected demand
    seed: int = 7


def post_promo_flag(promo: np.ndarray, days: int) -> np.ndarray:
    """1 on the `days` days right after a campaign ends (and no campaign running)."""
    promo = np.asarray(promo, dtype=float)
    flag = np.zeros(len(promo))
    for k in range(1, len(promo)):
        if promo[k] == 0 and promo[max(0, k - days):k].sum() > 0:
            flag[k] = 1
    return flag


def _promo_calendar(n: int, cfg: SimConfig, rng: np.random.Generator):
    promo, discount = np.zeros(n), np.zeros(n)
    i = 0
    while True:
        i += int(rng.integers(*cfg.promo_gap))
        length = int(rng.integers(*cfg.promo_len))
        if i + length >= n:
            break
        promo[i:i + length] = 1
        discount[i:i + length] = round(float(rng.uniform(*cfg.promo_discount)), 2)
        i += length
    return promo, discount


def generate_world(cfg: SimConfig = SimConfig(), history_days: int = 3 * 365,
                   future_days: int = 365, start: str = "2023-01-01") -> pd.DataFrame:
    """Return history + future days. Columns: DATA_COLUMNS + `expected` + driver multipliers (f_*)."""
    n = history_days + future_days
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(start, periods=n, freq="D")
    t = np.arange(n)
    doy = dates.day_of_year.to_numpy()

    trend = 1 + cfg.trend_per_year * t / 365.25
    annual = 1 + cfg.annual_amp * np.cos(2 * np.pi * (doy - cfg.annual_peak_doy) / 365.25)
    weekly = np.asarray(cfg.weekly)[dates.day_of_week.to_numpy()]

    ar = np.zeros(n)
    shocks = rng.normal(0, cfg.temp_ar_sd, n)
    for k in range(1, n):
        ar[k] = cfg.temp_ar_phi * ar[k - 1] + shocks[k]
    temp = cfg.temp_mean + cfg.temp_amp * np.cos(2 * np.pi * (doy - cfg.temp_peak_doy) / 365.25) + ar
    temp_fcst = temp + rng.normal(0, cfg.temp_forecast_sd, n)
    temp_eff = 1 + cfg.temp_sensitivity * (temp - cfg.temp_mean)

    promo, discount = _promo_calendar(n, cfg, rng)
    promo_eff = 1 + cfg.promo_lift_per_discount * discount
    post_eff = 1 - cfg.post_promo_dip * post_promo_flag(promo, cfg.post_promo_days)

    md = dates.strftime("%m-%d")
    holiday = np.isin(md, list(HOLIDAYS)).astype(float)
    hol_eff = 1 + cfg.holiday_lift * holiday

    expected = cfg.base * trend * annual * weekly * temp_eff * promo_eff * post_eff * hol_eff
    demand = np.maximum(0, rng.normal(expected, cfg.noise_cv * expected)).round()

    return pd.DataFrame({
        "date": dates, "demand": demand, "promo": promo, "discount": discount,
        "holiday": holiday, "temp": temp.round(1), "temp_fcst": temp_fcst.round(1),
        "expected": expected,
        "f_trend": trend, "f_annual": annual, "f_weekly": weekly, "f_temp": temp_eff,
        "f_promo": promo_eff, "f_post_promo": post_eff, "f_holiday": hol_eff,
    })
