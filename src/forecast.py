"""Lead-time demand forecasting: predict total demand over the next L days.

Forecast origin = end of day i (demand known up to and including day i).
Target          = demand[i+1] + ... + demand[i+L]

Feature groups (used for the ablation study):
  lags      - recent demand, scaled by the 28-day mean so the model is level-free
  calendar  - weekday the window starts on, time of year, holidays inside the window
  promo     - planned promotions inside the window, recent promotions (post-promo dip)
  external  - *forecast* temperature over the window (actual temperature is unknown in advance)

Trend handling: tree models cannot extrapolate beyond the target range they were trained on.
We therefore predict the ratio  target / (L x 28-day mean)  and multiply back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

FEATURE_GROUPS = {
    "lags": ["lag0_rel", "lag6_rel", "rm7_rel", "lastL_rel", "momentum"],
    "calendar": ["dow_next", "doy_sin", "doy_cos", "holidays_in_window"],
    "promo": ["promo_days_in_window", "discount_sum_in_window", "promo_days_prev7"],
    "external": ["temp_fcst_in_window"],
}
ALL_GROUPS = list(FEATURE_GROUPS)
MODELS = {"Random Forest": "rf", "Gradient Boosting": "gb"}


def _fwd_sum(x, L: int) -> np.ndarray:
    """out[i] = x[i+1] + ... + x[i+L]; NaN where the window runs past the data or hits a NaN."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n <= L:
        return out
    cs = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(n - L)
    out[i] = cs[i + L + 1] - cs[i + 1]
    return out


def build_features(df: pd.DataFrame, L: int) -> pd.DataFrame:
    """df: observed history (+ optional future plan rows with demand = NaN), sorted by date."""
    df = df.sort_values("date").reset_index(drop=True)
    d = df["demand"].astype(float)
    f = pd.DataFrame({"date": df["date"]})

    rm28 = d.rolling(28).mean()
    f["level"] = rm28
    f["lag0_rel"] = d / rm28
    f["lag6_rel"] = d.shift(6) / rm28              # same weekday as the first day of the window
    f["rm7_rel"] = d.rolling(7).mean() / rm28
    f["lastL_rel"] = d.rolling(L).sum() / (L * rm28)
    f["momentum"] = rm28 / rm28.shift(28)

    nxt = df["date"] + pd.Timedelta(days=1)
    mid = df["date"] + pd.Timedelta(days=(L + 1) / 2)
    f["dow_next"] = nxt.dt.day_of_week
    f["doy_sin"] = np.sin(2 * np.pi * mid.dt.day_of_year / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * mid.dt.day_of_year / 365.25)
    f["holidays_in_window"] = _fwd_sum(df["holiday"], L)

    f["promo_days_in_window"] = _fwd_sum(df["promo"], L)
    f["discount_sum_in_window"] = _fwd_sum(df["discount"], L)
    f["promo_days_prev7"] = df["promo"].rolling(7).sum()

    f["temp_fcst_in_window"] = _fwd_sum(df["temp_fcst"], L) / L

    f["target"] = _fwd_sum(d, L)
    f["observed"] = d.notna().to_numpy()
    return f


def feature_columns(groups) -> list[str]:
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def make_model(kind: str, seed: int = 0):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.6,
                                     n_jobs=-1, random_state=seed)
    if kind == "gb":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                         subsample=0.8, random_state=seed)
    raise ValueError(kind)


def _fit(feat: pd.DataFrame, rows, cols, kind: str, ratio: bool, seed: int):
    X = feat.loc[rows, cols]
    y = feat.loc[rows, "target"]
    if ratio:
        y = y / (feat.loc[rows, "level"] * _L(feat))
    return make_model(kind, seed).fit(X, y)


def _predict(model, feat: pd.DataFrame, rows, cols, ratio: bool) -> np.ndarray:
    p = model.predict(feat.loc[rows, cols])
    return p * feat.loc[rows, "level"].to_numpy() * _L(feat) if ratio else p


def _L(feat: pd.DataFrame) -> int:
    return int(feat.attrs["L"])


def walk_forward(df: pd.DataFrame, L: int, kind: str = "gb", groups=ALL_GROUPS,
                 test_days: int = 182, refit_every: int = 28, ratio: bool = True,
                 seed: int = 0) -> pd.DataFrame:
    """Out-of-sample backtest that mimics production.

    Every `refit_every` days the model is retrained on everything known at that moment
    (rows whose L-day target window had already finished), then used for the next block.
    """
    feat = build_features(df, L)
    feat.attrs["L"] = L
    cols = feature_columns(groups)
    valid = feat.dropna(subset=cols + ["target", "level"]).index
    origins = valid[-test_days:]
    out = []
    for b in range(0, len(origins), refit_every):
        block = origins[b:b + refit_every]
        train = valid[valid <= block[0] - L]   # target window must be complete at the origin
        model = _fit(feat, train, cols, kind, ratio, seed)
        out.append(pd.DataFrame({
            "origin": feat.loc[block, "date"].to_numpy(),
            "actual": feat.loc[block, "target"].to_numpy(),
            "forecast": _predict(model, feat, block, cols, ratio),
            "naive": (feat.loc[block, "lastL_rel"] * feat.loc[block, "level"] * L).to_numpy(),
            "moving_avg": (feat.loc[block, "level"] * L).to_numpy(),
            "refit": [True] + [False] * (len(block) - 1),
        }))
    return pd.concat(out, ignore_index=True)


def metrics(actual, pred) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    e = p - a
    return {
        "MAPE %": float(np.mean(np.abs(e) / a) * 100),
        "WAPE %": float(np.abs(e).sum() / a.sum() * 100),
        "Bias %": float(e.sum() / a.sum() * 100),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
    }


def forecast_next(df: pd.DataFrame, L: int, kind: str, backtest: pd.DataFrame,
                  groups=ALL_GROUPS, seed: int = 0, levels=(0.1, 0.9)) -> dict:
    """Fit on all available history; forecast the next L days from the last observed day.

    `df` must include L future plan rows (promo, discount, holiday, temp_fcst; demand NaN).
    The interval comes from the backtest's relative errors (actual / forecast), which
    scale with the level - matching the heteroscedastic noise.
    """
    feat = build_features(df, L)
    feat.attrs["L"] = L
    cols = feature_columns(groups)
    train = feat.dropna(subset=cols + ["target", "level"]).index
    model = _fit(feat, train, cols, kind, True, seed)
    origin = feat.index[feat["observed"]][-1]
    point = float(_predict(model, feat, [origin], cols, True)[0])
    rel = (backtest["actual"] / backtest["forecast"]).to_numpy()
    lo, hi = np.quantile(rel, levels)
    importances = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    return {"origin": feat.loc[origin, "date"], "point": point, "low": point * lo,
            "high": point * hi, "importances": importances}
