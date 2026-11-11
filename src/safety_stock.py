"""Safety stock and reorder point: analytic formula vs Monte Carlo, classic vs AI inputs.

Analytic (assumes normal demand and normal lead time):
    SS  = Z * sqrt(sigma_D^2 * L_mean + D_bar^2 * sigma_L^2)
    ROP = D_bar * L_mean + SS

Two ways to feed the formula:
  classic - D_bar = 28-day average demand, sigma_D = raw spread of daily demand (90 days).
            Raw spread includes *predictable* swings (weekdays, promotions), so it over-states risk,
            and the average ignores promotions that are already planned.
  AI      - D_bar = model forecast for the next L days / L, sigma_D from the model's own errors.
            Only the *unpredictable* part of demand needs a buffer.

Monte Carlo drops the normal assumptions: lead time is drawn from its real (skewed) distribution,
and forecast errors are bootstrapped from the back-test, so their real shape is kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm

CLASSIC_MEAN_DAYS, CLASSIC_SD_DAYS = 28, 90


# ---------------------------------------------------------------- lead time
def lead_time_pmf(mean: float, sd: float, shape: str = "skewed"):
    """Whole-day lead-time distribution. Returns (days, probabilities).

    skewed: lognormal with the given mean and sd (delays more likely than early arrivals).
    normal: normal with the given mean and sd. Both are rounded to whole days, minimum 1.
    """
    if sd <= 0:
        return np.array([max(1, round(mean))]), np.array([1.0])
    days = np.arange(1, int(np.ceil(mean + 8 * sd)) + 2)
    if shape == "skewed":
        s = np.sqrt(np.log(1 + (sd / mean) ** 2))
        cdf = lognorm(s, scale=mean * np.exp(-s ** 2 / 2)).cdf
    else:
        cdf = norm(mean, sd).cdf
    p = cdf(days + 0.5) - cdf(days - 0.5)
    p[0] += cdf(0.5)                     # anything under 1 day counts as 1 day
    keep = p > 1e-7
    return days[keep], p[keep] / p[keep].sum()


def pmf_moments(days, p):
    m = float((days * p).sum())
    return m, float(np.sqrt(((days - m) ** 2 * p).sum()))


# ---------------------------------------------------------------- formula and simulation
def analytic(z: float, d_bar: float, sigma_d: float, L_mean: float, sigma_L: float) -> dict:
    ss = z * np.sqrt(sigma_d ** 2 * L_mean + d_bar ** 2 * sigma_L ** 2)
    return {"d_bar": d_bar, "sigma_d": sigma_d, "ss": float(ss), "rop": float(d_bar * L_mean + ss)}


def ai_inputs(forecast: float, L_fc: int, rel_errors) -> tuple[float, float]:
    """Daily rate and daily sigma implied by an L-day forecast and its relative errors.

    If the L-day total has relative error sd s, its sd is s*forecast; spread over L independent
    days that is a daily sd of s*forecast/sqrt(L) (the sigma_D in the formula).
    """
    d_bar = forecast / L_fc
    sigma_d = float(np.std(rel_errors, ddof=1)) * forecast / np.sqrt(L_fc)
    return d_bar, sigma_d


def simulate_ltd(forecast: float, L_fc: int, rel_errors, days, p, n: int = 20_000,
                 rng: np.random.Generator | None = None):
    """Monte Carlo lead-time demand.

    Draw a lead time l and a past forecast error e, then
        LTD = d_hat * (l + e * sqrt(l * L_fc))
    At l = L_fc this is exactly forecast x (1 + e); for other lead times the error scales
    with sqrt(l), matching the formula's sigma_D^2 * L term. Returns (ltd, lead_times).
    """
    rng = rng or np.random.default_rng(0)
    d_hat = forecast / L_fc
    ell = rng.choice(days, n, p=p)
    e = rng.choice(np.asarray(rel_errors, float), n)
    return d_hat * (ell + e * np.sqrt(ell * L_fc)), ell


def monte_carlo(ltd: np.ndarray, csl: float, d_bar: float, L_mean: float) -> dict:
    rop = float(np.quantile(ltd, csl))
    return {"rop": rop, "ss": rop - d_bar * L_mean}


def classic_inputs(history: pd.Series) -> tuple[float, float]:
    return (float(history.iloc[-CLASSIC_MEAN_DAYS:].mean()),
            float(history.iloc[-CLASSIC_SD_DAYS:].std(ddof=1)))


# ---------------------------------------------------------------- honest check over time
def dynamic_backtest(demand: pd.Series, bt: pd.DataFrame, L_fc: int, days, p, csl: float,
                     eval_days: int = 182, n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Recompute safety stock every day and check it against what really happened.

    demand : actual daily demand for the whole world (history and beyond), indexed by date
    bt     : walk-forward forecasts (origin, actual, forecast). Rows before the evaluation period
             only serve as error history.
    For each origin t, only forecast errors whose window had finished by t are used (no look-ahead).
    Achieved service = probability (over the lead-time distribution) that demand during the lead
    time stayed at or below the reorder point, using the *actual* demand that followed.
    """
    rng = np.random.default_rng(seed)
    z = norm.ppf(csl)
    L_mean, sigma_L = pmf_moments(days, p)
    bt = bt.reset_index(drop=True)
    rel = (bt["actual"] / bt["forecast"] - 1).to_numpy()
    origins = bt["origin"].to_numpy()
    d = demand.to_numpy(dtype=float)
    pos = pd.Index(demand.index)
    kmax = int(days.max())
    rows = []
    for i in range(max(0, len(bt) - eval_days), len(bt)):
        t = bt.loc[i, "origin"]
        known = rel[origins <= t - pd.Timedelta(days=L_fc)]
        if len(known) < 20:
            continue
        j = pos.get_loc(t)
        if j + kmax >= len(d):
            break
        cum = np.cumsum(d[j + 1:j + 1 + kmax])          # actual demand over the next 1..kmax days
        fc = bt.loc[i, "forecast"]

        def achieved(rop):
            return float((p * (cum[days - 1] <= rop)).sum())

        d_hat, s_ai = ai_inputs(fc, L_fc, known)
        ai_f = analytic(z, d_hat, s_ai, L_mean, sigma_L)
        ltd, _ = simulate_ltd(fc, L_fc, known, days, p, n, rng)
        ai_mc = monte_carlo(ltd, csl, d_hat, L_mean)
        d_cl, s_cl = classic_inputs(demand.iloc[:j + 1])
        cl = analytic(z, d_cl, s_cl, L_mean, sigma_L)
        rows.append({
            "origin": t,
            "ss_classic": cl["ss"], "rop_classic": cl["rop"], "svc_classic": achieved(cl["rop"]),
            "ss_ai_formula": ai_f["ss"], "rop_ai_formula": ai_f["rop"], "svc_ai_formula": achieved(ai_f["rop"]),
            "ss_ai_mc": ai_mc["ss"], "rop_ai_mc": ai_mc["rop"], "svc_ai_mc": achieved(ai_mc["rop"]),
        })
    return pd.DataFrame(rows)


METHODS = {"classic": "Classic formula", "ai_formula": "AI + formula", "ai_mc": "AI + Monte Carlo"}


def summarize(dyn: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        label: {"Avg safety stock": dyn[f"ss_{k}"].mean(),
                "Avg reorder point": dyn[f"rop_{k}"].mean(),
                "Achieved service %": dyn[f"svc_{k}"].mean() * 100}
        for k, label in METHODS.items()}).T
