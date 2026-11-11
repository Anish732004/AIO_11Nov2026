"""Reorder Point Lab - AI-driven demand forecasting, safety stock and reorder-point optimization."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_io import REQUIRED_UPLOAD, validate_new_rows
from scipy.stats import norm

from src.forecast import ALL_GROUPS, MODELS, forecast_next, metrics, walk_forward
from src.safety_stock import (METHODS, ai_inputs, analytic, classic_inputs, dynamic_backtest,
                              lead_time_pmf, monte_carlo, pmf_moments, simulate_ltd, summarize)
from src.simulate import DATA_COLUMNS, HOLIDAYS, SimConfig, generate_world

st.set_page_config(page_title="Reorder Point Lab", page_icon="📦", layout="wide")

HISTORY_DAYS, FUTURE_DAYS = 3 * 365, 365
MAX_REVEAL = FUTURE_DAYS - 30
# Categorical slots in fixed order (validated palette); gray = baseline / de-emphasised.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"


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


@st.cache_data(show_spinner=False)
def ss_timeline(demand: pd.Series, bt: pd.DataFrame, L: int, sigma_L: float, shape: str, csl: float):
    days, p = lead_time_pmf(L, sigma_L, shape)
    return dynamic_backtest(demand, bt, L, days, p, csl)


@st.cache_data(show_spinner="Refitting on all data and forecasting...")
def next_forecast(df_plan: pd.DataFrame, L: int, kind: str, bt: pd.DataFrame) -> dict:
    return forecast_next(df_plan, L, kind, bt)


def observed(w: pd.DataFrame, n: int, overrides: pd.DataFrame) -> pd.DataFrame:
    obs = w.iloc[:HISTORY_DAYS + n][DATA_COLUMNS].copy()
    if len(overrides):
        obs = obs.set_index("date")
        obs.update(overrides.set_index("date")[[c for c in DATA_COLUMNS if c != "date"]])
        obs = obs.reset_index()
    return obs


def with_plan(w: pd.DataFrame, obs: pd.DataFrame, L: int) -> pd.DataFrame:
    """Append the next L days of *known-in-advance* inputs (promo plan, holidays, weather forecast)."""
    plan = w.iloc[len(obs):len(obs) + L][DATA_COLUMNS].copy()
    plan[["demand", "temp"]] = np.nan
    return pd.concat([obs, plan], ignore_index=True)


# ---------------------------------------------------------------- chart helpers
def style(fig: go.Figure, title: str, ytitle: str, height: int = 340) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)), height=height,
                      margin=dict(l=8, r=8, t=56, b=8), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title=ytitle, zeroline=False)
    return fig


def line(x, y, name, color, width=2.0, **kw):
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=width), **kw)


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


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Scenario")
    seed = st.number_input("Simulation seed", 1, 9999, 7, help="A different seed = a different synthetic world.",
                           on_change=reset)
    L = st.slider("Average lead time L (days)", 3, 14, 5,
                  help="We forecast total demand over the next L days: the demand a reorder must cover.")
    sigma_L_in = st.slider("Lead-time variability σL (days)", 0.0, 4.0, 1.5, 0.5,
                           help="Standard deviation of supplier lead time.")
    shape_label = st.radio("Lead-time shape", ["Right-skewed (delays)", "Normal"], horizontal=True,
                           help="Real suppliers are late more often than early. The formula assumes normal.")
    shape = "skewed" if shape_label.startswith("Right") else "normal"
    model_name = st.radio("Forecast model", list(MODELS), index=1, horizontal=True)
    kind = MODELS[model_name]

    st.divider()
    st.header("New data feed")
    st.caption("Release the next days of the simulated world. The model retrains on everything it has seen.")
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

# ---------------------------------------------------------------- header
st.title("Reorder Point Lab")
st.caption("Predict lead-time demand with machine learning, size safety stock from the forecast's own errors, "
           "and let Bayesian optimization choose the reorder point.")

tab1, tab2, tab3 = st.tabs(["1 · Demand forecasting", "2 · Dynamic safety stock", "3 · Reorder-point optimization"])

# ================================================================= TAB 1
with tab1:
    # One year of walk-forward forecasts: the first half only builds up error history for the
    # safety-stock tab, the second half (last 182 days) is what tab 1 scores.
    bt_full = backtest(obs, L, kind, test_days=364)
    bt = bt_full.tail(182).reset_index(drop=True)
    bt_other = backtest(obs, L, "rf" if kind == "gb" else "gb", test_days=364).tail(182)
    fc = next_forecast(with_plan(w, obs, L), L, kind, bt_full)

    # forecast log: every forecast made at an origin, scored once its window has happened
    log = ss.log.setdefault(f"{seed}-{L}-{kind}", {})
    log.setdefault(fc["origin"], (fc["point"], fc["low"], fc["high"]))

    m_model = metrics(bt["actual"], bt["forecast"])
    m_ma = metrics(bt["actual"], bt["moving_avg"])
    win = f"{(last_day + pd.Timedelta(days=1)):%d %b} – {(last_day + pd.Timedelta(days=L)):%d %b %Y}"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Forecast demand, next {L} days", f"{fc['point']:,.0f} units",
              help=f"Window {win}. Range is the 80 % interval from back-test errors.")
    k1.caption(f"{win} · 80 % range {fc['low']:,.0f}–{fc['high']:,.0f}")
    k2.metric(f"{model_name} error (WAPE)", f"{m_model['WAPE %']:.1f} %",
              delta=f"{m_model['WAPE %'] - m_ma['WAPE %']:+.1f} pts vs moving average", delta_color="inverse")
    k3.metric("Moving-average baseline (WAPE)", f"{m_ma['WAPE %']:.1f} %")
    k4.metric("Days of data", f"{len(obs):,}", delta=f"+{ss.n} new" if ss.n else None)

    # ---- data
    st.subheader("The data")
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
    st.plotly_chart(style(fig, "Daily demand (drag the slider below to see all history)", "Units / day", 380),
                    width="stretch")

    c_left, c_right = st.columns([1, 1])
    with c_left:
        tfig = go.Figure(line(view["date"], view["temp"], "Actual temperature", BLUE, 1.2))
        tfig.update_xaxes(range=[last_day - pd.Timedelta(days=365), last_day])
        st.plotly_chart(style(tfig, "External factor: temperature", "°C", 260), width="stretch")
    with c_right:
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
        st.markdown("**How demand is generated**")
        st.dataframe(drivers[["Driver", "Rule", "Range"]], hide_index=True, width="stretch")
        st.caption("demand = 100 × product of multipliers, plus noise with sd = 10 % of the level.")

    # ---- back-test
    st.subheader(f"Walk-forward back-test: forecasting {L}-day demand")
    st.caption("Each point is a forecast made with only the data available on that day. "
               "The model is retrained every 28 days, as it would be in production.")
    bfig = go.Figure([
        line(bt["origin"], bt["moving_avg"], "Moving average (28 d)", GRAY, 1.5),
        line(bt["origin"], bt["actual"], f"Actual {L}-day demand", BLUE),
        line(bt["origin"], bt["forecast"], f"{model_name} forecast", ORANGE),
    ])
    st.plotly_chart(style(bfig, f"Actual vs forecast {L}-day demand (x = forecast date)", "Units", 360),
                    width="stretch")

    other_name = next(n for n, k in MODELS.items() if k != kind)
    table = pd.DataFrame({
        model_name: m_model,
        other_name: metrics(bt_other["actual"], bt_other["forecast"]),
        "Moving average (28 d)": m_ma,
        f"Naïve (last {L} days)": metrics(bt["actual"], bt["naive"]),
    }).T
    c_left, c_right = st.columns([3, 2])
    with c_left:
        st.markdown("**Accuracy over the back-test** (lower is better)")
        st.dataframe(table.style.format("{:.1f}"), width="stretch")
        st.caption("WAPE = total absolute error ÷ total demand. Bias > 0 means over-forecasting.")
    with c_right:
        imp = fc["importances"].head(8)[::-1]
        ifig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=BLUE,
                                hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        ifig.update_layout(hovermode="closest")
        st.plotly_chart(style(ifig, "What the model relies on", "", 300), width="stretch")

    with st.expander("Which drivers matter? Drop-one feature ablation (about 10 s)"):
        if st.toggle("Run the ablation on the current data"):
            sets = {"All features": tuple(ALL_GROUPS)}
            sets |= {f"Without {g}": tuple(x for x in ALL_GROUPS if x != g) for g in ALL_GROUPS[1:]}
            res = {k: metrics(*backtest(obs, L, kind, v)[["actual", "forecast"]].T.values)["WAPE %"]
                   for k, v in sets.items()}
            r = pd.Series(res)
            afig = go.Figure(go.Bar(x=r.index, y=r.values, text=[f"{v:.1f} %" for v in r.values],
                                    textposition="outside",
                                    marker_color=[BLUE] + [GRAY] * (len(r) - 1),
                                    hovertemplate="%{x}: %{y:.2f} % WAPE<extra></extra>"))
            afig.update_layout(hovermode="closest")
            afig.update_yaxes(range=[0, r.max() * 1.2])
            st.plotly_chart(style(afig, "Back-test WAPE when one driver group is removed", "WAPE %", 320),
                            width="stretch")
            st.caption("The bigger the jump from 'All features', the more that driver group is worth. "
                       "Lag features are always kept.")

    st.subheader("Forecast log: continuous updating")
    st.caption("Every time new data arrives (sidebar), the model retrains and a new forecast is logged. "
               "Once its window has passed, the actual demand is filled in.")
    d = obs.set_index("date")["demand"]
    rows = []
    for origin, (pt, lo, hi) in sorted(log.items(), reverse=True):
        end = origin + pd.Timedelta(days=L)
        known = end <= last_day
        actual = d.loc[origin + pd.Timedelta(days=1):end].sum() if known else None
        rows.append({"Forecast made": origin.date(), "Window": f"{origin + pd.Timedelta(days=1):%d %b} – {end:%d %b}",
                     "Forecast": f"{pt:,.0f}", "80 % range": f"{lo:,.0f} – {hi:,.0f}",
                     "Actual": f"{actual:,.0f}" if known else "—",
                     "Error %": f"{(pt - actual) / actual * 100:+.1f}" if known else "—",
                     "Inside range": ("✅ yes" if lo <= actual <= hi else "❌ no") if known else "pending"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if len(rows) == 1:
        st.caption("Press **+7 d** in the sidebar a few times to watch forecasts get scored.")

# ================================================================= TAB 2 / 3 (next steps)
with tab2:
    days, p = lead_time_pmf(L, sigma_L_in, shape)
    L_mean, sigma_L = pmf_moments(days, p)          # moments of the whole-day distribution actually used
    target = st.slider("Target cycle service level", 80.0, 99.5, 95.0, 0.5, format="%.1f %%",
                       help="Probability of not running out between placing an order and receiving it.") / 100
    z = norm.ppf(target)

    rel_all = (bt_full["actual"] / bt_full["forecast"] - 1).to_numpy()
    d_ai, s_ai = ai_inputs(fc["point"], L, rel_all)
    ai_f = analytic(z, d_ai, s_ai, L_mean, sigma_L)
    ltd, ell = simulate_ltd(fc["point"], L, rel_all, days, p, 20_000, np.random.default_rng(seed))
    ai_mc = monte_carlo(ltd, target, d_ai, L_mean)
    d_cl, s_cl = classic_inputs(obs["demand"])
    cl = analytic(z, d_cl, s_cl, L_mean, sigma_L)
    formula_cover = float((ltd <= ai_f["rop"]).mean())

    st.subheader("Today's recommendation")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Reorder point (AI + Monte Carlo)", f"{ai_mc['rop']:,.0f} units",
              help="Place an order when on-hand + on-order inventory falls to this level.")
    k2.metric("of which safety stock", f"{ai_mc['ss']:,.0f} units",
              help="Reorder point minus expected demand during the average lead time.")
    k3.metric("Classic reorder point", f"{cl['rop']:,.0f} units",
              delta=f"{cl['rop'] - ai_mc['rop']:+,.0f} vs AI", delta_color="off")
    k4.metric("Real protection of the AI formula", f"{formula_cover * 100:.1f} %",
              delta=f"{(formula_cover - target) * 100:+.1f} pts vs target",
              help="Share of Monte Carlo scenarios the formula's reorder point would have covered.")
    st.caption(f"For an order placed on {last_day:%d %b %Y}. Lead time: average {L_mean:.1f} days, "
               f"σL {sigma_L:.2f} days, {shape_label.lower()}.")

    # ---- the formula with today's numbers
    st.subheader("The formula, with today's numbers")
    st.markdown(r"$SS = Z\cdot\sqrt{\sigma_D^2\cdot \bar L + \bar D^2\cdot\sigma_L^2}$,  "
                r"$\;ROP = \bar D\cdot\bar L + SS$,  "
                f"with Z = {z:.3f} for {target * 100:.1f} % service.")
    c_left, c_right = st.columns(2)
    for col, name, res, note in [
        (c_left, "Classic inputs", cl, f"D̄ = average of the last 28 days; σD = spread of the last 90 days of daily "
                                      "demand, including predictable swings (weekdays, promotions)."),
        (c_right, "AI inputs", ai_f, f"D̄ = forecast for the next {L} days ÷ {L} (knows about planned promotions); "
                                    "σD from the model's past errors, so only unpredictable demand is buffered."),
    ]:
        with col:
            st.markdown(f"**{name}**")
            st.latex(rf"SS = {z:.2f}\sqrt{{{res['sigma_d']:.1f}^2\cdot{L_mean:.1f} + "
                     rf"{res['d_bar']:.1f}^2\cdot{sigma_L:.2f}^2}} = {res['ss']:,.0f}")
            st.caption(note)
    parts = pd.DataFrame({
        "Demand uncertainty  σD·√L̄": [cl["sigma_d"] * np.sqrt(L_mean), ai_f["sigma_d"] * np.sqrt(L_mean)],
        "Lead-time uncertainty  D̄·σL": [cl["d_bar"] * sigma_L, ai_f["d_bar"] * sigma_L],
    }, index=["Classic", "AI"])
    st.markdown("**Where the buffer comes from** (units, before multiplying by Z; combined as √(a² + b²))")
    st.dataframe(parts.style.format("{:,.0f}"), width="stretch")
    if sigma_L > 0 and (parts.iloc[1, 1] > parts.iloc[1, 0]):
        st.caption("Lead-time uncertainty is the bigger part here, so a better forecast can only shrink "
                   "the buffer a little. Lower σL in the sidebar to see the forecast's value grow.")

    # ---- Monte Carlo
    st.subheader("Monte Carlo: 20,000 possible lead-time scenarios")
    c_left, c_right = st.columns([2, 1])
    with c_left:
        hfig = go.Figure(go.Histogram(x=ltd, nbinsx=80, marker_color=BLUE, opacity=0.85, name="Scenarios",
                                      hovertemplate="%{x} units: %{y} scenarios<extra></extra>"))
        # labels at different heights; the formula label sits left of its line so it never meets the MC one
        mc_right = ai_mc["rop"] >= ai_f["rop"]
        for x, label, color, y, side in [(d_ai * L_mean, "Expected", GRAY, 1.0, "left"),
                                         (ai_f["rop"], "Formula ROP", AQUA, 0.84, "right" if mc_right else "left"),
                                         (ai_mc["rop"], "Monte Carlo ROP", ORANGE, 1.0, "left" if mc_right else "right")]:
            hfig.add_vline(x=x, line_color=color, line_width=2)
            hfig.add_annotation(x=x, y=y, yref="paper", text=f"{label}<br>{x:,.0f}", showarrow=False,
                                xanchor=side, yanchor="top", xshift=4 if side == "left" else -4,
                                font=dict(size=11), bgcolor="rgba(255,255,255,0.7)")
        hfig.update_layout(hovermode="closest", showlegend=False, bargap=0.05)
        st.plotly_chart(style(hfig, "Demand during the lead time", "Scenarios", 340), width="stretch")
        st.caption(f"Each scenario draws a lead time and one of the model's past forecast errors. The Monte Carlo "
                   f"reorder point is the {target * 100:.1f}th percentile, so it covers that share of scenarios "
                   "whatever the distribution's shape. The bumps are whole-day lead times (4, 5, 6 days...).")
    with c_right:
        lfig = go.Figure(go.Bar(x=days, y=p * 100, marker_color=BLUE, name="Lead time",
                                hovertemplate="%{x} days: %{y:.1f} %<extra></extra>"))
        lfig.update_layout(hovermode="closest", showlegend=False)
        lfig.update_xaxes(title="Days", dtick=1 if days.max() <= 16 else 2)
        st.plotly_chart(style(lfig, "Lead-time distribution", "Probability %", 340), width="stretch")

    # ---- honest check over the last 6 months
    st.subheader("Did it work? Six months of daily decisions, checked against actual demand")
    st.caption("Every day the reorder point is recomputed with only the information available that day, then "
               "scored on the demand that really followed, averaged over the lead-time distribution.")
    demand_all = pd.concat([obs.set_index("date")["demand"],
                            w.iloc[len(obs):].set_index("date")["demand"]])
    grid = (0.80, 0.85, 0.90, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999)
    grid = tuple(sorted(set(grid) | {round(target, 4)}))
    front = ss_backtest(demand_all, bt_full, L, sigma_L_in, shape, grid)
    at_t = front[np.isclose(front["csl"], target)].set_index("method")

    def ss_needed(label: str):
        f = front[front["method"] == label].sort_values("csl")
        svc, ss = f["Achieved service %"].to_numpy() / 100, f["Avg safety stock"].to_numpy()
        if svc.max() < target:
            return None
        return float(np.interp(target, np.maximum.accumulate(svc), ss))

    summary = at_t[["Avg safety stock", "Avg reorder point", "Achieved service %"]].rename_axis("Method").copy()
    summary["SS needed to truly reach target"] = [ss_needed(m) for m in summary.index]
    need = summary["SS needed to truly reach target"]
    need_cl = need[METHODS["classic"]] if pd.notna(need[METHODS["classic"]]) else None
    need_ai = need[METHODS["ai_mc"]] if pd.notna(need[METHODS["ai_mc"]]) else None
    st.dataframe(summary.style.format({"Avg safety stock": "{:,.0f}", "Avg reorder point": "{:,.0f}",
                                       "Achieved service %": "{:.1f}",
                                       "SS needed to truly reach target": "{:,.0f}"}, na_rep="not reachable"),
                 width="stretch")
    if need_ai is not None and need_cl is not None:
        st.success(f"To really deliver {target * 100:.1f} % service, the classic method needs about "
                   f"{need_cl:,.0f} units of safety stock; AI + Monte Carlo needs {need_ai:,.0f} "
                   f"({(need_ai / need_cl - 1) * 100:+.0f} %).")
    elif need_ai is not None:
        st.success(f"The classic method could not reach {target * 100:.1f} % in practice at any tested setting; "
                   f"AI + Monte Carlo does it with about {need_ai:,.0f} units.")
    else:
        st.warning(f"Neither method reached {target * 100:.1f} % in practice at any tested setting.")

    c_left, c_right = st.columns(2)
    with c_left:
        ffig = go.Figure()
        for key, color in [("classic", GRAY), ("ai_formula", AQUA), ("ai_mc", ORANGE)]:
            f = front[front["method"] == METHODS[key]].sort_values("csl")
            ffig.add_trace(go.Scatter(x=f["Avg safety stock"], y=f["Achieved service %"], name=METHODS[key],
                                      mode="lines+markers", line=dict(color=color, width=2),
                                      marker=dict(size=8, line=dict(color="white", width=1)),
                                      customdata=f["csl"] * 100,
                                      hovertemplate="SS %{x:,.0f} → %{y:.1f} % achieved"
                                                    "<br>(asked for %{customdata:.1f} %)<extra></extra>"))
        ffig.add_hline(y=target * 100, line_color=GRAY, line_width=1,
                       annotation_text=f"target {target * 100:.1f} %", annotation_position="bottom right")
        ffig.update_layout(hovermode="closest")
        ffig.update_xaxes(title="Average safety stock (units)")
        st.plotly_chart(style(ffig, "Service bought per unit of safety stock (up and left is better)",
                              "Achieved service %", 380), width="stretch")
    with c_right:
        tl = ss_timeline(demand_all, bt_full, L, sigma_L_in, shape, target)
        d_series = demand_all.to_numpy(float)
        idx = pd.Index(demand_all.index)
        k = int(round(L_mean))
        act = [d_series[idx.get_loc(t) + 1: idx.get_loc(t) + 1 + k].sum() for t in tl["origin"]]
        tfig = go.Figure([
            line(tl["origin"], act, f"Actual {k}-day demand", BLUE, 1.5),
            line(tl["origin"], tl["rop_classic"], "Classic ROP", GRAY),
            line(tl["origin"], tl["rop_ai_mc"], "AI ROP", ORANGE),
        ])
        st.plotly_chart(style(tfig, "Reorder point over time vs demand that followed", "Units", 380),
                        width="stretch")
    st.caption("Left: each point is one nominal service level; the AI curve reaching higher service with less "
               "stock is the value of the forecast. The two AI curves overlap: the formula is not less efficient, "
               "it is mis-calibrated, landing below the target because it assumes a normal lead time. "
               "Right: the classic reorder point reacts to promotions only after they happen, while the AI one "
               "rises before them.")
with tab3:
    st.info("Next step: Bayesian optimization (Optuna) of the reorder point under a service-level constraint, "
            "compared with a brute-force grid sweep.")
