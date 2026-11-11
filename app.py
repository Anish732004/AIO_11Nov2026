"""Reorder Point Lab - AI-driven demand forecasting, safety stock and reorder-point optimization."""
from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

# ---------------------------------------------------------------- keep code and cache in step
# A redeploy (e.g. Streamlit Cloud pulling a new commit) can leave the running server with
# (a) cached results computed by the previous src/ code - st.cache_data only tracks changes to the
#     cached function's own body - and (b) the previous src/ modules still imported.
# Either mismatch can crash the app. Fingerprint src/; when it changes, drop the old modules so the
# imports below load the new code, and clear every cached result. Runs once per fingerprint.
SRC_VERSION = hashlib.sha256(b"".join(
    p.read_bytes() for p in sorted((Path(__file__).parent / "src").glob("*.py")))).hexdigest()[:12]


@st.cache_resource(show_spinner=False)
def _sync_with_code(version: str) -> str:
    for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[name]
    st.cache_data.clear()
    return version


_sync_with_code(SRC_VERSION)

from src.data_io import REQUIRED_UPLOAD, validate_new_rows  # noqa: E402  (must follow the sync above)
from src.forecast import ALL_GROUPS, FEATURE_GROUPS, MODELS, forecast_next, metrics, walk_forward
from src.optimize import Costs, Problem, bayes_opt, best_feasible, best_so_far, grid_search, lead_scenarios
from src.safety_stock import (METHODS, ai_inputs, analytic, classic_inputs, dynamic_backtest,
                              lead_time_pmf, monte_carlo, pmf_moments, simulate_ltd, summarize)
from src.simulate import DATA_COLUMNS, HOLIDAYS, SimConfig, generate_world

st.set_page_config(page_title="Reorder Point Lab", page_icon="📦", layout="wide")

HISTORY_DAYS, FUTURE_DAYS = 3 * 365, 365
MAX_REVEAL = FUTURE_DAYS - 30
# Categorical slots in fixed order (validated palette); gray = baseline / de-emphasised.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"

FEATURE_LABELS = {
    "lag0_rel": "Yesterday's demand", "lag6_rel": "Demand a week ago", "rm7_rel": "Last 7-day average",
    "lastL_rel": "Last lead-time total", "momentum": "Recent trend", "dow_next": "Weekday the window starts",
    "doy_sin": "Time of year", "doy_cos": "Time of year", "holidays_in_window": "Holidays ahead",
    "promo_days_in_window": "Promotion days ahead", "discount_sum_in_window": "Planned discounts ahead",
    "promo_days_prev7": "Recent promotions", "temp_fcst_in_window": "Weather forecast",
}
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
GROUP_LABELS = {"lags": "recent demand", "calendar": "the calendar", "promo": "planned promotions",
                "external": "the weather forecast"}


# ---------------------------------------------------------------- data & models (cached)
@st.cache_data(show_spinner=False)
def world(seed: int) -> pd.DataFrame:
    return generate_world(replace(SimConfig(), seed=seed), HISTORY_DAYS, FUTURE_DAYS)


@st.cache_data(show_spinner="Back-testing the model (walk-forward)...")
def backtest(obs: pd.DataFrame, L: int, kind: str, groups: tuple = tuple(ALL_GROUPS),
             test_days: int = 182) -> pd.DataFrame:
    return walk_forward(obs, L, kind, list(groups), test_days=test_days)


@st.cache_data(show_spinner="Re-running 6 months of daily safety-stock decisions...")
def ss_backtest(demand: pd.Series, bt: pd.DataFrame, L: int, sigma_L: float, shape: str,
                csl_grid: tuple) -> pd.DataFrame:
    """Back-tested average safety stock and achieved service per method, for each nominal CSL."""
    days, p = lead_time_pmf(L, sigma_L, shape)
    out = []
    for c in csl_grid:
        s = summarize(dynamic_backtest(demand, bt, L, days, p, c))
        s["csl"] = c
        out.append(s.rename_axis("method").reset_index())
    return pd.concat(out, ignore_index=True)


@st.cache_data(show_spinner="Simulating inventory and optimizing (Bayesian vs grid)...")
def run_optimization(policy: str, service: str, target: float, costs: tuple, budget: int,
                     demand: np.ndarray, mu: np.ndarray, sigma: np.ndarray, L: int, sigma_L: float,
                     shape: str, seed: int, with_grids: bool = True) -> dict:
    days, p = lead_time_pmf(L, sigma_L, shape)
    lead = lead_scenarios(days, p, 20, len(demand), seed)
    pr = Problem(policy, demand, mu, sigma, lead, Costs(*costs), service, target)
    out = {"names": pr.names, "bounds": pr.bounds, "bo": bayes_opt(pr, budget, seed)}
    if with_grids:
        out |= {"grid": grid_search(pr, int(round(np.sqrt(budget)))), "full": grid_search(pr, 40)}
    return out


@st.cache_data(show_spinner=False)
def ss_timeline(demand: pd.Series, bt: pd.DataFrame, L: int, sigma_L: float, shape: str, csl: float):
    days, p = lead_time_pmf(L, sigma_L, shape)
    return dynamic_backtest(demand, bt, L, days, p, csl)


@st.cache_data(show_spinner="Refitting on all data and forecasting...")
def next_forecast(df_plan: pd.DataFrame, L: int, kind: str, bt: pd.DataFrame) -> dict:
    return forecast_next(df_plan, L, kind, bt)


def observed(w: pd.DataFrame, n: int, overrides: pd.DataFrame) -> pd.DataFrame:
    """World history up to the released day, with uploaded rows replacing the simulated ones by date."""
    obs = w.iloc[:HISTORY_DAYS + n][DATA_COLUMNS].copy()
    if len(overrides):
        ov = overrides[DATA_COLUMNS].astype(obs.dtypes.to_dict())   # uploaded rows start as generic objects
        obs = pd.concat([obs[~obs["date"].isin(ov["date"])], ov]).sort_values("date").reset_index(drop=True)
    return obs


def with_plan(w: pd.DataFrame, obs: pd.DataFrame, L: int) -> pd.DataFrame:
    """Append the next L days of *known-in-advance* inputs (promo plan, holidays, weather forecast)."""
    plan = w.iloc[len(obs):len(obs) + L][DATA_COLUMNS].copy()
    plan[["demand", "temp"]] = np.nan
    return pd.concat([obs, plan], ignore_index=True)


# ---------------------------------------------------------------- UI helpers
def style(fig: go.Figure, ytitle: str, height: int = 340) -> go.Figure:
    """Chart chrome. Titles live outside the chart (chart_title) so they can carry a '?' help icon."""
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=28, b=8), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title=ytitle, zeroline=False)
    return fig


def chart_title(title: str, read: str, notice: str):
    st.markdown(f"**{title}**", help=f"**How to read it:** {read}\n\n**What to notice:** {notice}")


def line(x, y, name, color, width=2.0, **kw):
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=width), **kw)


def tab_settings_box():
    box = st.container(border=True)
    box.markdown("⚙️ **Settings for this tab only**")
    return box


def using_sidebar(*items: str):
    st.caption("Using sidebar settings: " + " · ".join(items) + ". Change them in the sidebar.")


# ---------------------------------------------------------------- session state
ss = st.session_state
ss.setdefault("n", 0)
ss.setdefault("overrides", pd.DataFrame(columns=DATA_COLUMNS))
ss.setdefault("log", {})
ss.setdefault("upload_seen", None)


def reset():
    # upload_seen is kept, so a file still sitting in the uploader is not re-applied after a reset
    ss.n, ss.overrides, ss.log = 0, pd.DataFrame(columns=DATA_COLUMNS), {}


def reveal(k: int):
    ss.n = min(ss.n + k, MAX_REVEAL)


# ---------------------------------------------------------------- sidebar (global settings)
with st.sidebar:
    st.markdown("### Global settings")
    st.caption("Everything here applies to **all three tabs**. Each tab also has its own boxed settings.")

    st.markdown("**🌍 Simulated world**")
    seed = st.number_input("World number", 1, 9999, 7, on_change=reset,
                           help="Each number creates a different 3-year demand history with the same rules. "
                                "Use it to check that the conclusions hold in other worlds. Affects all tabs.")

    st.markdown("**🔮 Forecast model**")
    model_name = st.radio("Model", list(MODELS), index=1, horizontal=True, label_visibility="collapsed",
                          help="Both are tree-based machine-learning models. The forecast feeds the safety "
                               "stock (tab 2) and the optimization (tab 3). Affects all tabs.")
    kind = MODELS[model_name]

    st.markdown("**🚚 Supplier lead time**")
    L = st.slider("Average lead time (days)", 3, 14, 5,
                  help="How long a delivery takes on average. The model forecasts demand over this many days, "
                       "because that is what an order must cover. Affects all tabs.")
    sigma_L_in = st.slider("Lead-time variability (± days)", 0.0, 4.0, 1.5, 0.5,
                           help="Standard deviation of delivery time. 0 = the supplier is always on time. "
                                "Affects tabs 2 and 3.")
    shape_label = st.radio("Delays pattern", ["Right-skewed (delays)", "Normal"], horizontal=True,
                           help="Real suppliers are late more often than early (right-skewed). The textbook "
                                "formula assumes a symmetric bell curve (normal). Affects tabs 2 and 3.")
    shape = "skewed" if shape_label.startswith("Right") else "normal"

    st.divider()
    st.markdown("**📥 New data**")
    st.caption("Release the next days of the simulated world. The model retrains on everything it has seen, "
               "and every tab updates.")
    c1, c2, c3 = st.columns(3)
    c1.button("+1 d", on_click=reveal, args=(1,), width="stretch", help="Release 1 more day")
    c2.button("+7 d", on_click=reveal, args=(7,), width="stretch", help="Release 7 more days")
    c3.button("+30 d", on_click=reveal, args=(30,), width="stretch", help="Release 30 more days")
    st.button("Reset to original history", on_click=reset, width="stretch")

w = world(seed)
obs = observed(w, ss.n, ss.overrides)

with st.sidebar:
    with st.expander("Upload your own new rows (CSV)"):
        st.caption(f"Columns: {', '.join(REQUIRED_UPLOAD)} (+ optional temp). "
                   f"Consecutive daily rows starting {(obs['date'].iloc[-1] + pd.Timedelta(days=1)).date()}.")
        template = w.iloc[len(obs):len(obs) + 14][DATA_COLUMNS]
        st.download_button("Download a 14-day template", template.to_csv(index=False),
                           file_name="new_demand_rows.csv", mime="text/csv", width="stretch")
        up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
        if up is not None and up.file_id != ss.upload_seen:
            ss.upload_seen = up.file_id
            try:
                new = validate_new_rows(pd.read_csv(up), obs["date"].iloc[-1], MAX_REVEAL - ss.n)
                ss.overrides = pd.concat([ss.overrides, new], ignore_index=True)
                ss.n += len(new)
                st.rerun()
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(f"Could not use this file: {exc}")

obs = observed(w, ss.n, ss.overrides)
last_day = obs["date"].iloc[-1]
lead_desc = f"lead time {L} ± {sigma_L_in:g} days ({'skewed' if shape == 'skewed' else 'normal'})"

# ---------------------------------------------------------------- header + guide
st.title("Reorder Point Lab")
st.caption("When should a store reorder, and how much? Three steps, one per tab. Work through them left to right.")
g1, g2, g3 = st.columns(3)
g1.container(border=True).markdown("**① Forecast demand**  \nPredict how much will sell while a delivery is "
                                   "on its way, including promotions and seasons.")
g2.container(border=True).markdown("**② Size the safety stock**  \nTurn the forecast's uncertainty into a "
                                   "buffer that really hits a service target.")
g3.container(border=True).markdown("**③ Optimize the policy**  \nLet Bayesian optimization pick when and how "
                                   "much to order at the lowest cost.")
st.caption("ⓘ Sidebar settings change every tab. Settings in a ⚙️ box inside a tab change only that tab. "
           "Hover the ? icons for how to read each chart.")

tab1, tab2, tab3 = st.tabs(["① Forecast demand", "② Safety stock", "③ Optimize reorder policy"])

# ================================================================= TAB 1
with tab1:
    # One year of walk-forward forecasts: the first half only builds up error history for the
    # safety-stock tab, the second half (last 182 days) is what tab 1 scores.
    bt_full = backtest(obs, L, kind, test_days=364)
    bt = bt_full.tail(182).reset_index(drop=True)
    bt_other = backtest(obs, L, "rf" if kind == "gb" else "gb", test_days=364).tail(182)
    fc = next_forecast(with_plan(w, obs, L), L, kind, bt_full)

    log = ss.log.setdefault(f"{seed}-{L}-{kind}", {})
    log.setdefault(fc["origin"], (fc["point"], fc["low"], fc["high"]))

    m_model = metrics(bt["actual"], bt["forecast"])
    m_ma = metrics(bt["actual"], bt["moving_avg"])
    win = f"{(last_day + pd.Timedelta(days=1)):%d %b} – {(last_day + pd.Timedelta(days=L)):%d %b %Y}"
    imp = fc["importances"].groupby(lambda f: FEATURE_LABELS[f]).sum().sort_values(ascending=False)
    top_group = GROUP_LABELS[GROUP_OF[fc["importances"].index[0]]]

    st.markdown(f"#### How much will sell in the next {L} days?")
    using_sidebar(f"world {seed}", model_name, f"{L}-day lead time", f"data up to {last_day:%d %b %Y}")
    st.info(f"**Key takeaway:** the {model_name} forecast is off by **{m_model['WAPE %']:.1f} %** on average, "
            f"against **{m_ma['WAPE %']:.1f} %** for a simple 28-day average: "
            f"{m_ma['WAPE %'] / m_model['WAPE %']:.1f}× more accurate. It leans most on **{top_group}**, "
            "which a moving average cannot see coming.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Forecast, next {L} days", f"{fc['point']:,.0f} units",
              help=f"Total demand expected from {win}. 80 % of the time the actual should fall within "
                   f"{fc['low']:,.0f}–{fc['high']:,.0f} units (range taken from the model's past errors).")
    k1.caption(f"{win} · likely {fc['low']:,.0f}–{fc['high']:,.0f}")
    k2.metric(f"{model_name} error", f"{m_model['WAPE %']:.1f} %",
              delta=f"{m_model['WAPE %'] - m_ma['WAPE %']:+.1f} pts vs simple average", delta_color="inverse",
              help="WAPE: total forecast error ÷ total demand, over the last 6 months of forecasts the model "
                   "had never seen. Lower is better.")
    k3.metric("Simple 28-day average error", f"{m_ma['WAPE %']:.1f} %",
              help="The baseline: assume the next days look like the average of the last 28. "
                   "A model is only worth using if it clearly beats this.")
    k4.metric("Days of history", f"{len(obs):,}", delta=f"+{ss.n} new" if ss.n else None,
              help="Days the model learns from. Grows when you release new data in the sidebar.")

    # ---- data
    chart_title("Daily demand",
                "the blue line is units sold per day. Orange dots are promotion days, green diamonds are "
                "holidays. Drag the slider under the chart to see all three years.",
                "spikes line up with promotions and holidays, demand climbs into the festive season "
                "(Oct–Nov), and the whole series grows year on year. These are the patterns the model learns.")
    view = obs.copy()
    fig = go.Figure()
    fig.add_trace(line(view["date"], view["demand"], "Daily demand", BLUE, 1.4))
    p = view[view["promo"] == 1]
    fig.add_trace(go.Scatter(x=p["date"], y=p["demand"], name="Promotion day", mode="markers",
                             marker=dict(color=ORANGE, size=7, line=dict(color="white", width=1)),
                             customdata=p["discount"] * 100, hovertemplate="%{y:.0f} units · %{customdata:.0f}% off"))
    h = view[view["holiday"] == 1]
    fig.add_trace(go.Scatter(x=h["date"], y=h["demand"], name="Holiday", mode="markers",
                             marker=dict(color=AQUA, size=9, symbol="diamond", line=dict(color="white", width=1)),
                             text=h["date"].dt.strftime("%m-%d").map(HOLIDAYS), hovertemplate="%{y:.0f} units · %{text}"))
    if ss.n:
        fig.add_vrect(x0=w["date"].iloc[HISTORY_DAYS], x1=last_day, fillcolor=GRAY, opacity=0.12, line_width=0,
                      annotation_text="new data", annotation_position="top left")
    fig.update_xaxes(range=[last_day - pd.Timedelta(days=365), last_day], rangeslider=dict(visible=True, thickness=0.06))
    st.plotly_chart(style(fig, "Units / day", 360), width="stretch")

    with st.expander("Behind the data: how this synthetic demand is generated"):
        c_left, c_right = st.columns([1, 1])
        with c_left:
            hist = w.iloc[:len(obs)]
            drivers = pd.DataFrame([
                ("Trend", "+15 %/year growth", hist["f_trend"]),
                ("Annual season", "±20 %, peak late Oct (festive)", hist["f_annual"]),
                ("Day of week", "Mon 0.90 … Sat 1.25", hist["f_weekly"]),
                ("Promotion", "1 + 2 × discount", hist["f_promo"]),
                ("Post-promo dip", "−12 % for 3 days after", hist["f_post_promo"]),
                ("Holiday", "+35 % on the day", hist["f_holiday"]),
                ("Temperature", "+1 % per °C above 27 °C", hist["f_temp"]),
            ], columns=["Driver", "Rule", "f"])
            drivers["Range"] = drivers["f"].map(lambda s: f"×{s.min():.2f} – {s.max():.2f}")
            st.dataframe(drivers[["Driver", "Rule", "Range"]], hide_index=True, width="stretch")
            st.caption("Demand = 100 × all multipliers, plus random noise of about 10 %. "
                       "Range = smallest and largest multiplier each driver produced.")
        with c_right:
            chart_title("External factor: temperature",
                        "daily temperature in °C. It peaks in summer (May), not in the festive season.",
                        "its effect on demand is deliberately small (about +1 % per °C), so it adds little "
                        "accuracy. That is an honest result: not every external factor matters.")
            tfig = go.Figure(line(view["date"], view["temp"], "Actual temperature", BLUE, 1.2))
            tfig.update_xaxes(range=[last_day - pd.Timedelta(days=365), last_day])
            st.plotly_chart(style(tfig, "°C", 240), width="stretch")

    # ---- back-test
    st.markdown("#### Would the forecast have worked? A 6-month test")
    chart_title(f"Actual vs forecast {L}-day demand",
                f"each date on the x-axis is a day a forecast was made, using only data up to that day. "
                f"Blue is the demand that really followed over the next {L} days, orange is the forecast, gray "
                "is the simple 28-day average. The model was retrained every 28 days, as it would be in real use.",
                "orange rises with the promotion peaks before they happen; gray stays flat and misses them.")
    bfig = go.Figure([
        line(bt["origin"], bt["moving_avg"], "Simple 28-day average", GRAY, 1.5),
        line(bt["origin"], bt["actual"], f"Actual {L}-day demand", BLUE),
        line(bt["origin"], bt["forecast"], f"{model_name} forecast", ORANGE),
    ])
    st.plotly_chart(style(bfig, "Units", 340), width="stretch")

    other_name = next(n for n, k in MODELS.items() if k != kind)
    table = pd.DataFrame({
        model_name: m_model,
        other_name: metrics(bt_other["actual"], bt_other["forecast"]),
        "Simple 28-day average": m_ma,
        f"Naïve (repeat last {L} days)": metrics(bt["actual"], bt["naive"]),
    }).T
    c_left, c_right = st.columns([3, 2])
    with c_left:
        chart_title("Accuracy of each method (lower is better)",
                    "WAPE is the headline: total error ÷ total demand. MAPE is the average % error per forecast. "
                    "Bias above 0 means over-forecasting (extra stock); below 0, under-forecasting (stockouts). "
                    "RMSE is in units and punishes big misses most.",
                    "both machine-learning models roughly halve the error of the simple methods, and their bias "
                    "stays close to zero.")
        st.dataframe(table.style.format("{:.1f}"), width="stretch")
    with c_right:
        chart_title("What the model relies on",
                    "how much each input helped the model reduce its error during training. Longer bar = "
                    "relied on more.",
                    "planned discounts dominate. Treat this as a hint, not proof: related inputs can share credit. "
                    "The ablation below is the stronger test.")
        top = imp.head(7)[::-1]
        ifig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=BLUE,
                                hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        ifig.update_layout(hovermode="closest")
        st.plotly_chart(style(ifig, "", 280), width="stretch")

    with st.expander("Which drivers really matter? Remove one at a time (about 10 s)"):
        st.caption("The model is re-tested with one group of inputs taken away. The bigger the error jump, "
                   "the more that group is worth. Recent-demand inputs are always kept.")
        if st.toggle("Run the test on the current data"):
            sets = {"All inputs": tuple(ALL_GROUPS)}
            sets |= {f"Without {GROUP_LABELS[g]}": tuple(x for x in ALL_GROUPS if x != g) for g in ALL_GROUPS[1:]}
            res = {k: metrics(*backtest(obs, L, kind, v)[["actual", "forecast"]].T.values)["WAPE %"]
                   for k, v in sets.items()}
            r = pd.Series(res)
            afig = go.Figure(go.Bar(x=r.index, y=r.values, text=[f"{v:.1f} %" for v in r.values],
                                    textposition="outside",
                                    marker_color=[BLUE] + [GRAY] * (len(r) - 1),
                                    hovertemplate="%{x}: %{y:.2f} % error<extra></extra>"))
            afig.update_layout(hovermode="closest")
            afig.update_yaxes(range=[0, r.max() * 1.2])
            st.plotly_chart(style(afig, "Error (WAPE %)", 300), width="stretch")

    # ---- forecast log
    st.markdown("#### Keeping the forecast up to date")
    chart_title("Forecast log",
                "each row is a forecast made on that date. Once its window has passed, the actual demand is "
                "filled in and checked against the forecast's likely range.",
                "press +7 d in the sidebar a few times: the model retrains each time, and about 8 in 10 actuals "
                "should land inside the range. Far fewer would mean the model is over-confident.")
    d = obs.set_index("date")["demand"]
    rows = []
    for origin, (pt, lo, hi) in sorted(log.items(), reverse=True):
        end = origin + pd.Timedelta(days=L)
        known = end <= last_day
        actual = d.loc[origin + pd.Timedelta(days=1):end].sum() if known else None
        rows.append({"Forecast made": origin.date(), "Window": f"{origin + pd.Timedelta(days=1):%d %b} – {end:%d %b}",
                     "Forecast": f"{pt:,.0f}", "Likely range": f"{lo:,.0f} – {hi:,.0f}",
                     "Actual": f"{actual:,.0f}" if known else "—",
                     "Error %": f"{(pt - actual) / actual * 100:+.1f}" if known else "—",
                     "Inside range": ("✅ yes" if lo <= actual <= hi else "❌ no") if known else "pending"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("**Next → tab ②:** the forecast is never perfect. How much extra stock covers its errors?")

# ================================================================= TAB 2
with tab2:
    days, p = lead_time_pmf(L, sigma_L_in, shape)
    L_mean, sigma_L = pmf_moments(days, p)          # moments of the whole-day distribution actually used

    st.markdown("#### How much safety stock do we need, and does it really work?")
    using_sidebar(f"world {seed}", model_name, lead_desc)
    takeaway2 = st.container()

    with tab_settings_box():
        target = st.slider("Target service level", 80.0, 99.5, 95.0, 0.5, format="%.1f %%",
                           help="Chance of not running out between placing an order and receiving it "
                                "(cycle service level). Higher = more safety stock.") / 100
    z = norm.ppf(target)

    rel_all = (bt_full["actual"] / bt_full["forecast"] - 1).to_numpy()
    d_ai, s_ai = ai_inputs(fc["point"], L, rel_all)
    ai_f = analytic(z, d_ai, s_ai, L_mean, sigma_L)
    ltd, ell = simulate_ltd(fc["point"], L, rel_all, days, p, 20_000, np.random.default_rng(seed))
    ai_mc = monte_carlo(ltd, target, d_ai, L_mean)
    d_cl, s_cl = classic_inputs(obs["demand"])
    cl = analytic(z, d_cl, s_cl, L_mean, sigma_L)
    formula_cover = float((ltd <= ai_f["rop"]).mean())

    demand_all = pd.concat([obs.set_index("date")["demand"], w.iloc[len(obs):].set_index("date")["demand"]])
    csl_grid = tuple(sorted({0.80, 0.85, 0.90, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999,
                             round(target, 4)}))
    front = ss_backtest(demand_all, bt_full, L, sigma_L_in, shape, csl_grid)
    at_t = front[np.isclose(front["csl"], target)].set_index("method")

    def ss_needed(key: str):
        f = front[front["method"] == key].sort_values("csl")
        svc, s = f["Achieved service %"].to_numpy() / 100, f["Avg safety stock"].to_numpy()
        return None if svc.max() < target else float(np.interp(target, np.maximum.accumulate(svc), s))

    # rows are keyed by stable method ids ('classic', 'ai_formula', 'ai_mc'); names are added for display only
    summary = at_t[["Avg safety stock", "Avg reorder point", "Achieved service %"]].copy()
    summary["Safety stock needed to truly hit target"] = [ss_needed(k) for k in summary.index]
    need = summary["Safety stock needed to truly hit target"]
    need_cl = need["classic"] if pd.notna(need["classic"]) else None
    need_ai = need["ai_mc"] if pd.notna(need["ai_mc"]) else None
    svc_cl = at_t.loc["classic", "Achieved service %"]
    svc_ai = at_t.loc["ai_mc", "Achieved service %"]

    msg = (f"**Key takeaway:** reorder when stock falls to **{ai_mc['rop']:,.0f} units** (of which "
           f"{ai_mc['ss']:,.0f} is safety stock). Over the last 6 months, the classic method promised "
           f"{target * 100:.1f} % but delivered **{svc_cl:.1f} %**; the AI method delivered **{svc_ai:.1f} %**.")
    if need_cl is not None and need_ai is not None:
        msg += (f" To truly reach the target, AI needs **{(1 - need_ai / need_cl) * 100:.0f} % less safety "
                "stock** than the classic method.")
    elif need_ai is not None:
        msg += " The classic method could not reach the target at all."
    else:
        msg += " Neither method fully reaches a target this high; try a lower one."
    takeaway2.info(msg)

    k1, k2, k3, k4 = st.columns(4)
    diff = cl["rop"] - ai_mc["rop"]
    k1.metric("Reorder when stock falls to", f"{ai_mc['rop']:,.0f} units",
              help=f"The reorder point: place an order when stock on hand + already on order reaches this level. "
                   f"Method: {METHODS['ai_mc']}, for an order placed on {last_day:%d %b %Y}.")
    k2.metric("Safety buffer included", f"{ai_mc['ss']:,.0f} units",
              help="The safety stock: the part of the reorder point above the demand expected during the "
                   "delivery time. It absorbs forecast errors and late deliveries.")
    k3.metric("Textbook method's reorder point", f"{cl['rop']:,.0f} units",
              delta=f"{abs(diff):,.0f} units {'below' if diff < 0 else 'above'} the AI level", delta_color="off",
              delta_arrow="off",          # the words say the direction; an inferred arrow contradicted them
              help=f"{METHODS['classic']}: the formula with a 28-day average and the raw spread of daily demand. "
                   "It does not know about upcoming promotions.")
    k4.metric("Service the formula really delivers", f"{formula_cover * 100:.1f} %",
              delta=f"{(formula_cover - target) * 100:+.1f} pts vs the {target * 100:.1f} % target",
              help=f"{METHODS['ai_formula']}: the share of the 20,000 simulated scenarios its reorder point would "
                   "really cover. Below target means it under-protects, usually because late deliveries are "
                   "more likely than the formula assumes.")

    with st.expander("Show the math: the formula with today's numbers"):
        st.markdown(r"$SS = Z\cdot\sqrt{\sigma_D^2\cdot \bar L + \bar D^2\cdot\sigma_L^2}$,  "
                    r"$\;ROP = \bar D\cdot\bar L + SS$,  "
                    f"with Z = {z:.3f} for {target * 100:.1f} % service.")
        c_left, c_right = st.columns(2)
        for col, name, r_, note in [
            (c_left, "Classic inputs", cl, "D̄ = average of the last 28 days; σD = spread of the last 90 days of daily "
                                         "demand, including predictable swings (weekdays, promotions)."),
            (c_right, "AI inputs", ai_f, f"D̄ = forecast for the next {L} days ÷ {L} (knows about planned "
                                       "promotions); σD from the model's past errors, so only the unpredictable "
                                       "part of demand is buffered."),
        ]:
            with col:
                st.markdown(f"**{name}**")
                st.latex(rf"SS = {z:.2f}\sqrt{{{r_['sigma_d']:.1f}^2\cdot{L_mean:.1f} + "
                         rf"{r_['d_bar']:.1f}^2\cdot{sigma_L:.2f}^2}} = {r_['ss']:,.0f}")
                st.caption(note)
        parts = pd.DataFrame({
            "From demand uncertainty  σD·√L̄": [cl["sigma_d"] * np.sqrt(L_mean), ai_f["sigma_d"] * np.sqrt(L_mean)],
            "From lead-time uncertainty  D̄·σL": [cl["d_bar"] * sigma_L, ai_f["d_bar"] * sigma_L],
        }, index=["Classic", "AI"])
        chart_title("Where the buffer comes from (units, before × Z)",
                    "the formula combines two sources of risk as √(a² + b²): uncertain demand and uncertain "
                    "delivery time.",
                    "the AI forecast shrinks the demand part. If the lead-time part is much bigger, a better "
                    "forecast can only help a little: lower the lead-time variability in the sidebar to see the "
                    "forecast's value grow.")
        st.dataframe(parts.style.format("{:,.0f}"), width="stretch")

    # ---- Monte Carlo
    st.markdown("#### Simulating 20,000 possible futures")
    c_left, c_right = st.columns([2, 1])
    with c_left:
        chart_title("Demand while waiting for a delivery",
                    "each bar counts scenarios with that much demand between ordering and receiving stock. Each "
                    "scenario combines a random delivery time with one of the model's real past errors. Gray = "
                    "average, green = formula's reorder point, orange = Monte Carlo reorder point (covers "
                    f"{target * 100:.1f} % of scenarios).",
                    "the long right tail comes from late deliveries. The formula assumes a symmetric bell curve, "
                    "so it sits too low. The bumps are whole-day lead times (4, 5, 6 days…).")
        counts, edges = np.histogram(ltd, bins=60)
        top_y = counts.max() * 1.05
        hfig = go.Figure(go.Histogram(x=ltd, marker_color=BLUE, opacity=0.85,
                                      xbins=dict(start=edges[0], end=edges[-1], size=edges[1] - edges[0]),
                                      name="Simulated scenarios", showlegend=False,
                                      hovertemplate="%{x} units: %{y} scenarios<extra></extra>"))
        # vertical markers as legend entries (plain names + value) instead of crowded text labels
        for x, label, color in [(d_ai * L_mean, "Expected demand during delivery", GRAY),
                                (ai_f["rop"], f"Reorder point: {METHODS['ai_formula']}", AQUA),
                                (ai_mc["rop"], f"Reorder point: {METHODS['ai_mc']}", ORANGE)]:
            hfig.add_trace(go.Scatter(x=[x, x], y=[0, top_y], mode="lines", line=dict(color=color, width=2.5),
                                      name=f"{label} ({x:,.0f})",
                                      hovertemplate=f"{label}: {x:,.0f} units<extra></extra>"))
        style(hfig, "Number of scenarios", 340)
        hfig.update_layout(hovermode="closest", bargap=0.05,     # legend in the empty top-right corner
                           legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.99))
        hfig.update_xaxes(title="Units demanded before the delivery arrives")
        st.plotly_chart(hfig, width="stretch")
    with c_right:
        chart_title("Delivery time",
                    "chance that a delivery takes each number of days. Set by the lead-time settings in the sidebar.",
                    "with right-skewed delays, very late deliveries (the right tail) are possible while very early "
                    "ones are not.")
        lfig = go.Figure(go.Bar(x=days, y=p * 100, marker_color=BLUE, name="Lead time",
                                hovertemplate="%{x} days: %{y:.1f} %<extra></extra>"))
        lfig.update_layout(hovermode="closest", showlegend=False)
        lfig.update_xaxes(title="Days until the delivery arrives", dtick=1 if days.max() <= 16 else 2)
        st.plotly_chart(style(lfig, "Chance (%)", 340), width="stretch")

    # ---- honest check over the last 6 months
    st.markdown("#### Did it work? 6 months of daily decisions, checked against real demand")
    chart_title("Promised vs delivered",
                "every day for 6 months, each method set a reorder point using only what was known that day. "
                "Achieved service is how often that would really have avoided a stockout, given the demand that "
                "followed. The last column is the safety stock each method needs to truly hit the target.",
                "the classic method under-delivers because it reacts to promotions only after they happen. "
                f"{METHODS['ai_mc']} lands on target.")
    st.dataframe(summary.rename(index=METHODS).rename_axis("Method").style.format({
                                       "Avg safety stock": "{:,.0f}", "Avg reorder point": "{:,.0f}",
                                       "Achieved service %": "{:.1f}",
                                       "Safety stock needed to truly hit target": "{:,.0f}"},
                                      na_rep="not reachable"), width="stretch")

    c_left, c_right = st.columns(2)
    with c_left:
        chart_title("Service bought per unit of safety stock",
                    "each dot is one service-level setting. Across = average stock held, up = service actually "
                    "achieved. Higher and further left is better.",
                    "the AI curves sit above the classic one: the same service with less stock. The two AI curves "
                    "overlap, so the formula is not wasteful, just mis-calibrated: it lands below the target.")
        ffig = go.Figure()
        for key, color in [("classic", GRAY), ("ai_formula", AQUA), ("ai_mc", ORANGE)]:
            f = front[front["method"] == key].sort_values("csl")
            ffig.add_trace(go.Scatter(x=f["Avg safety stock"], y=f["Achieved service %"], name=METHODS[key],
                                      mode="lines+markers", line=dict(color=color, width=2),
                                      marker=dict(size=8, line=dict(color="white", width=1)),
                                      customdata=f["csl"] * 100,
                                      hovertemplate="stock %{x:,.0f} → %{y:.1f} % achieved"
                                                    "<br>(asked for %{customdata:.1f} %)<extra></extra>"))
        ffig.add_hline(y=target * 100, line_color=GRAY, line_width=1,
                       annotation_text=f"target {target * 100:.1f} %", annotation_position="bottom right")
        ffig.update_layout(hovermode="closest")
        ffig.update_xaxes(title="Average safety stock (units)")
        st.plotly_chart(style(ffig, "Achieved service %", 360), width="stretch")
    with c_right:
        chart_title("Reorder point over time",
                    f"blue is the demand that actually followed over the next {round(L_mean)} days; the lines are "
                    "each method's reorder point on that day. When blue gets close to a line, that method was at "
                    "risk of running out.",
                    "the AI reorder point rises before the promotion peaks; the classic one only drifts upward "
                    "afterwards.")
        tl = ss_timeline(demand_all, bt_full, L, sigma_L_in, shape, target)
        d_series = demand_all.to_numpy(float)
        idx = pd.Index(demand_all.index)
        kk = int(round(L_mean))
        act = [d_series[idx.get_loc(t) + 1: idx.get_loc(t) + 1 + kk].sum() for t in tl["origin"]]
        tfig = go.Figure([
            line(tl["origin"], tl["rop_classic"], f"Reorder point: {METHODS['classic']}", GRAY),
            line(tl["origin"], tl["rop_ai_mc"], f"Reorder point: {METHODS['ai_mc']}", ORANGE),
            line(tl["origin"], act, f"Actual demand over the next {kk} days", BLUE, 1.5),
        ])
        style(tfig, "Units", 380)
        tfig.update_layout(margin=dict(t=64))                 # room for a two-row legend
        st.plotly_chart(tfig, width="stretch")
    st.caption("**Next → tab ③:** we fixed the service level here. What if we let costs decide both when and "
               "how much to order?")

# ================================================================= TAB 3
with tab3:
    st.markdown("#### What is the cheapest policy that still keeps customers served?")
    using_sidebar(f"world {seed}", model_name, lead_desc)
    takeaway3 = st.container()

    with tab_settings_box():
        c1, c2, c3 = st.columns(3)
        with c1:
            policy_label = st.radio("Reorder-point policy", ["AI dynamic", "Static"], horizontal=True,
                                    help="AI dynamic: the reorder point moves every day with the forecast "
                                         "(expected demand + k × its uncertainty). Static: one fixed number "
                                         "for the whole 6 months.")
            policy = "dynamic" if policy_label == "AI dynamic" else "static"
            budget = st.slider("Simulations allowed", 20, 150, 60, 5,
                               help="How many candidate policies each search method may try. Each one is a full "
                                    "6-month inventory simulation.")
        with c2:
            service_label = st.radio("Service rule", ["Fill rate", "Cycle service level"], horizontal=True,
                                     help="Fill rate: share of customer demand served from stock. Cycle service "
                                          "level: share of orders with no stockout while they were on the way.")
            service = "fill_rate" if service_label == "Fill rate" else "csl"
            svc_target = st.slider("Must be at least", 80.0, 99.5, 95.0, 0.5, format="%.1f %%",
                                   help="Policies below this are rejected, however cheap.") / 100
        with c3:
            h_cost = st.number_input("Holding cost (₹ per unit per day)", 0.01, 5.0, 0.10, 0.01, format="%.2f",
                                     help="Cost of keeping one unit on the shelf for a day (capital, space, "
                                          "spoilage). ₹0.10 ≈ 36 % a year on a ₹100 item.")
            p_cost = st.number_input("Lost-sale penalty (₹ per unit)", 0.0, 500.0, 20.0, 1.0,
                                     help="Lost margin and goodwill when a customer finds the shelf empty.")
            o_cost = st.number_input("Ordering cost (₹ per order)", 0.0, 20_000.0, 1_000.0, 100.0,
                                     help="Fixed cost of each order (transport, handling, admin). This is what "
                                          "makes ordering in bigger batches worthwhile.")
    st.caption("Each candidate (reorder point, order quantity) is tested by simulating 6 months of daily inventory "
               "with the real demand and 20 random delivery-time scenarios.")

    tl3 = ss_timeline(demand_all, bt_full, L, sigma_L_in, shape, target)
    mu_t = (tl3["rop_ai_formula"] - tl3["ss_ai_formula"]).to_numpy()
    sig_t = (tl3["ss_ai_formula"] / z).to_numpy()
    dem_t = demand_all.loc[tl3["origin"]].to_numpy(float)
    common = dict(service=service, target=svc_target, costs=(h_cost, p_cost, o_cost), budget=budget,
                  demand=dem_t, mu=mu_t, sigma=sig_t, L=L, sigma_L=sigma_L_in, shape=shape, seed=int(seed))
    res = run_optimization(policy, **common)
    other = run_optimization("static" if policy == "dynamic" else "dynamic", **common, with_grids=False)
    n1, n2 = res["names"]
    bo, grid, full = res["bo"], res["grid"], res["full"]
    b_bo, b_grid, b_full = best_feasible(bo), best_feasible(grid), best_feasible(full)
    b_other = best_feasible(other["bo"])

    if b_bo is None:
        takeaway3.warning("No simulated policy met the service rule. Lower the target or change the costs.")
    else:
        mu_today, sig_today = d_ai * L_mean, ai_f["ss"] / z
        r_today = b_bo["R"] if policy == "static" else mu_today + b_bo["k"] * sig_today
        q_days = b_bo["Q"] / dem_t.mean()
        gap = (b_bo["cost"] / b_full["cost"] - 1) * 100 if b_full is not None else np.nan
        msg = (f"**Key takeaway:** reorder at **{r_today:,.0f} units** and order **{b_bo['Q']:,.0f} units** "
               f"each time (about {q_days:.0f} days of demand). Cost: **₹{b_bo['cost']:,.0f} a day** with "
               f"{b_bo['service'] * 100:.1f} % {service_label.lower()}. Bayesian optimization found this in "
               f"{len(bo)} simulations, {gap:+.1f} % from the best of a 1,600-simulation brute-force search.")
        if b_other is not None:
            pol = {policy: b_bo, ("static" if policy == "dynamic" else "dynamic"): b_other}
            saving = 1 - pol["dynamic"]["cost"] / pol["static"]["cost"]
            msg += (f" Letting the reorder point follow the AI forecast is **{abs(saving) * 100:.1f} % "
                    f"{'cheaper' if saving >= 0 else 'more expensive'}** than keeping it fixed.")
        takeaway3.info(msg)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Reorder when stock falls to", f"{r_today:,.0f} units",
                  help="Order when stock on hand + on order falls to this. Static: one fixed number. "
                       "AI dynamic: today's forecast of lead-time demand + k × its uncertainty.")
        k2.metric("Order quantity", f"{b_bo['Q']:,.0f} units",
                  help=f"How much to order each time: about {q_days:.1f} days of average demand, "
                       f"{b_bo['orders_per_month']:.1f} orders a month.")
        k3.metric("Average daily cost", f"₹{b_bo['cost']:,.0f}",
                  help=f"Holding ₹{b_bo['holding']:,.0f} + lost sales ₹{b_bo['stockout']:,.0f} + "
                       f"ordering ₹{b_bo['ordering']:,.0f} per day.")
        k4.metric(f"Achieved {service_label.lower()}", f"{b_bo['service'] * 100:.1f} %",
                  delta=f"{(b_bo['service'] - svc_target) * 100:+.1f} pts vs rule", delta_color="off",
                  help="Above the rule by a lot = the rule is not what limits the policy; costs are.")
        notes = []
        if policy == "dynamic" and service == "csl":
            notes.append(f"Optimal safety factor k = {b_bo['k']:.2f}; the textbook formula would use "
                         f"Z = {norm.ppf(svc_target):.2f} for a {svc_target * 100:.1f} % cycle service level. "
                         "The optimizer weighs costs directly instead.")
        elif policy == "dynamic":
            notes.append(f"Optimal safety factor k = {b_bo['k']:.2f}: the buffer is {b_bo['k']:.2f} × the "
                         "forecast's uncertainty during the lead time.")
        if b_bo["service"] - svc_target < 0.005:
            notes.append("The service rule is **binding**: without it, the cheapest policy would serve fewer "
                         "customers.")
        else:
            notes.append("The service rule is **not binding**: a lost sale is expensive enough that the cheapest "
                         "policy already beats the target. Lower the lost-sale penalty to see it bind.")
        st.caption(" ".join(notes))

        # ---- method comparison
        st.markdown("#### How the best policy was found: Bayesian optimization vs grid search")
        n_side = int(round(np.sqrt(budget)))
        c_left, c_right = st.columns([3, 2])
        with c_left:
            chart_title("Cost map: every combination at once",
                        "each cell is the daily cost of one combination of reorder point (across) and order "
                        "quantity (up). Darker = more expensive; light gray = breaks the service rule. Gray "
                        "squares are where the grid search looked, orange dots where Bayesian optimization "
                        "looked, and the star is its best.",
                        "this map replaces one-at-a-time sensitivity sweeps. Bayesian trials start scattered, "
                        "then gather in the cheap (light) valley; the grid spends most of its runs in expensive "
                        "areas.")
            g1_ = np.linspace(*res["bounds"][0], 40)
            g2_ = np.linspace(*res["bounds"][1], 40)
            cost = full["cost"].to_numpy().reshape(40, 40).T          # rows = Q, cols = x1
            feas = full["feasible"].to_numpy().reshape(40, 40).T
            ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
            hm = go.Figure()
            hm.add_trace(go.Heatmap(x=g1_, y=g2_, z=np.where(feas, np.nan, 1), showscale=False, hoverinfo="skip",
                                    colorscale=[[0, "#e1e0d9"], [1, "#e1e0d9"]], name="Breaks the rule"))
            hm.add_trace(go.Heatmap(x=g1_, y=g2_, z=np.where(feas, cost, np.nan), colorscale=ramp,
                                    zmin=np.nanmin(np.where(feas, cost, np.nan)),
                                    zmax=np.nanquantile(np.where(feas, cost, np.nan), 0.9),
                                    colorbar=dict(title="₹/day", thickness=12), name="Daily cost",
                                    hovertemplate=f"{n1} %{{x:,.2f}}<br>Q %{{y:,.0f}}<br>₹%{{z:,.1f}}/day<extra></extra>"))
            hm.add_trace(go.Scatter(x=grid[n1], y=grid["Q"], mode="markers", name=f"Grid search ({len(grid)})",
                                    marker=dict(color=GRAY, size=7, symbol="square",
                                                line=dict(color="white", width=1))))
            hm.add_trace(go.Scatter(x=bo[n1], y=bo["Q"], mode="markers", name=f"Bayesian trials ({len(bo)})",
                                    marker=dict(color=ORANGE, size=8, line=dict(color="white", width=1)),
                                    customdata=np.arange(1, len(bo) + 1),
                                    hovertemplate="trial %{customdata}<extra></extra>"))
            hm.add_trace(go.Scatter(x=[b_bo[n1]], y=[b_bo["Q"]], mode="markers", name="Bayesian best",
                                    marker=dict(color=ORANGE, size=18, symbol="star",
                                                line=dict(color="#0b0b0b", width=1.5))))
            hm.update_layout(hovermode="closest")
            hm.update_xaxes(title="Safety factor k" if n1 == "k" else "Reorder point R (units)")
            st.plotly_chart(style(hm, "Order quantity Q (units)", 420), width="stretch")
        with c_right:
            chart_title("Best cost found so far",
                        "after each simulation, the cheapest policy found so far that keeps the service rule. "
                        "The blue line is the best of a 1,600-simulation brute-force search.",
                        "how quickly each method gets close to the blue line. The grid only improves when one of "
                        "its fixed points happens to land near the optimum; Bayesian optimization learns where to "
                        "try next.")
            cfig = go.Figure([
                line(np.arange(1, len(grid) + 1), best_so_far(grid), f"Grid search ({n_side}×{n_side})", GRAY),
                line(np.arange(1, len(bo) + 1), best_so_far(bo), "Bayesian optimization", ORANGE),
            ])
            if b_full is not None:
                cfig.add_hline(y=b_full["cost"], line_color=BLUE, line_width=1,
                               annotation_text="brute-force best", annotation_position="top right")
            cfig.update_xaxes(title="Simulations run")
            st.plotly_chart(style(cfig, "₹ per day", 420), width="stretch")

        def row(name, evals, b):
            if b is None:
                return {"Method": name, "Simulations": evals}
            return {"Method": name, "Simulations": evals, n1: b[n1], "Q": b["Q"], "Daily cost ₹": b["cost"],
                    "Gap vs brute force": (b["cost"] / b_full["cost"] - 1) * 100 if b_full is not None else np.nan,
                    "Service %": b["service"] * 100}

        chart_title("Results side by side",
                    "the best policy each search method found, its daily cost, and how far that is from the "
                    "brute-force best (0 % = equally good).",
                    f"Bayesian optimization uses about the same number of simulations as the {n_side}×{n_side} "
                    "grid, but is usually closer to the brute-force best, which needed 1,600.")
        comp = pd.DataFrame([row("Bayesian optimization", len(bo), b_bo),
                             row(f"Grid search ({n_side}×{n_side})", len(grid), b_grid),
                             row("Brute force (40×40, reference)", len(full), b_full)]).set_index("Method")
        fmt = {n1: "{:,.2f}" if n1 == "k" else "{:,.0f}", "Q": "{:,.0f}", "Daily cost ₹": "{:,.1f}",
               "Gap vs brute force": "{:+.1f} %", "Service %": "{:.1f}"}
        st.dataframe(comp.style.format(fmt, na_rep="none passed"), width="stretch")

        with st.expander("Is Bayesian optimization always better? What our tests showed"):
            st.markdown(
                "Over 10 random worlds, with two decisions to make (reorder point and order quantity):\n\n"
                "- With the **same number of simulations**, Bayesian optimization usually got closer to the best "
                "answer than a grid, but not always: a lucky grid can land right on it.\n"
                "- The grid is **erratic**: a 12×12 grid once did worse than a 10×10 one. Bayesian optimization "
                "improved steadily with more trials.\n"
                "- With 100–150 trials it matched or beat the **1,600-simulation** brute force, because it is not "
                "limited to fixed grid points.\n"
                "- Its advantage grows with more decisions: a 7-point grid needs 49 runs for 2 decisions but "
                "343 for 3 and 2,401 for 4.\n\n"
                "The method used is Optuna's TPE sampler: it builds a probability model of which settings tend to "
                "be cheap and samples there, and it treats the service rule as a constraint.")

        # ---- static vs AI dynamic
        if b_other is not None:
            st.markdown("#### Is the AI-driven reorder point worth it?")
            chart_title("Fixed vs forecast-driven reorder point",
                        "both policies were tuned by Bayesian optimization under the same costs and service rule. "
                        "Costs are per day; stock is the average number of units on the shelf.",
                        "the AI policy holds less stock in quiet weeks and more before promotions. Its saving is "
                        f"modest here because each order is large (about {pol['dynamic']['Q'] / dem_t.mean():.0f} "
                        "days of demand), so routine batch stock, not safety stock, is most of the inventory. "
                        "Lower the ordering cost to see the gap grow.")
            vs = pd.DataFrame({
                "Fixed reorder point": pol["static"], "AI forecast-driven": pol["dynamic"],
            }).T[["cost", "holding", "stockout", "ordering", "avg_on_hand", "service", "orders_per_month"]]
            vs.columns = ["Daily cost ₹", "Holding ₹", "Lost sales ₹", "Ordering ₹", "Avg units on hand",
                          f"{service_label} %", "Orders / month"]
            vs[f"{service_label} %"] *= 100
            st.dataframe(vs.astype(float).style.format("{:,.1f}"), width="stretch")
