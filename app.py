"""Volatility dashboard: equity indices and commodities.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # yfinance triggers pandas deprecation noise
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vol_dashboard import charts, crude, events, forecast, hedging, metrics as m, signal as sg  # noqa: E402
from vol_dashboard.vix import replicate_vix  # noqa: E402
from vol_dashboard.data import futures_curve, load_closes, load_ohlc, option_chain  # noqa: E402
from vol_dashboard.options import clean_chain, skew_metrics  # noqa: E402
from vol_dashboard.universe import BY_NAME, UNIVERSE, VIX_EXTRAS, VIX_TERM  # noqa: E402

st.set_page_config(page_title="Vol Dashboard", page_icon="📈", layout="wide")

# Shared hosts get rate-limited by Yahoo, so there the app opens on the CSV mirrors committed to the repo.
# Streamlit Community Cloud mounts the repo under /mount/src, Hugging Face Spaces sets SPACE_ID.
HOSTED = (str(Path(__file__).resolve()).startswith("/mount/src/") or "SPACE_ID" in os.environ
          or os.environ.get("OFFLINE_DEFAULT") == "1")

# ---------- cached loaders ----------

@st.cache_data(ttl=3600, show_spinner=False)
def ohlc(ticker: str, start: str, offline: bool) -> pd.DataFrame:
    return load_ohlc(ticker, start=start, offline=offline)


@st.cache_data(ttl=3600, show_spinner=False)
def closes(tickers: tuple[str, ...], start: str, offline: bool) -> pd.DataFrame:
    return load_closes(list(tickers), start=start, offline=offline)


@st.cache_data(ttl=900, show_spinner=False)
def chain(ticker: str, target_days: int):
    return option_chain(ticker, target_days)


@st.cache_data(ttl=3600, show_spinner=False)
def curve(root: str, exchange: str, offline: bool) -> pd.DataFrame:
    return futures_curve(root, exchange, offline=offline)


@st.cache_data(ttl=900, show_spinner=False)
def vix_replication(ticker: str) -> dict:
    return replicate_vix(ticker)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def forecasts(ticker: str, vol_index: str | None, start: str, offline: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = load_ohlc(ticker, start=start, offline=offline)
    iv = load_ohlc(vol_index, start=start, offline=offline)["Close"] if vol_index else None
    fc = forecast.run_forecasts(px, iv)
    return fc, forecast.evaluate(fc)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def crude_contracts(root: str, exchange: str, offline: bool) -> dict:
    return crude.load_contracts(root, exchange, offline=offline)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def straddle_backtest(ticker: str, vol_index: str, offline: bool, cost_bps: float, spread_pts: float) -> pd.DataFrame:
    px = load_ohlc(ticker, start="2005-01-01", offline=offline)["Close"]
    iv = load_ohlc(vol_index, start="2005-01-01", offline=offline)["Close"]
    return hedging.backtest_short_straddle(px, iv, hedge_cost_bps=cost_bps, option_spread_vol_pts=spread_pts)


@st.cache_data(ttl=900, show_spinner=False)
def live_event_move(ticker: str, event: pd.Timestamp) -> dict:
    return events.implied_move_from_chain(ticker, event)


def skew_history_df() -> pd.DataFrame:
    path = Path(__file__).parent / "data" / "skew_history.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def fmt(x: float, suffix: str = "%", digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}f}{suffix}"


# ---------- sidebar ----------

with st.sidebar:
    st.title("Vol Dashboard")
    st.caption("Equity indices and commodities. Realized vs implied, cone, term structure, skew.")
    asset_name = st.selectbox("Asset", list(BY_NAME), index=0)
    start = st.date_input("History from", pd.Timestamp("2015-01-01")).strftime("%Y-%m-%d")
    rv_window = st.slider("Realized vol window (days)", 5, 126, 21)
    offline = st.toggle("Offline (use cached CSV extracts)", value=HOSTED,
                        help="Off: live Yahoo prices and option chains. On: the CSV extracts in data/.")
    if HOSTED and offline:
        st.caption("Hosted demo: opens on the CSV snapshot committed to the repo. Switch offline off for live data and option chains.")
    st.divider()
    st.caption("Data: Yahoo Finance via yfinance, mirrored to `data/*.csv`. "
               "Implied vols are rebuilt from option mids with Black-76 on the implied forward.")

asset = BY_NAME[asset_name]
px = ohlc(asset.underlying, start, offline)
implied = ohlc(asset.vol_index, start, offline)["Close"].rename(asset.vol_index) if asset.vol_index else None

st.title(f"{asset.name}  ·  {asset.asset_class}")
if px.empty:
    st.error(f"No price data for {asset.underlying}.")
    st.stop()

# ---------- headline tiles ----------

rv_now = m.rv_close_to_close(px, rv_window).dropna()
st.caption(f"Last close {px.index[-1].strftime('%Y-%m-%d')}. Vols annualized, in %.")
cols = st.columns(4)
cols[0].metric("Spot", f"{px['Close'].iloc[-1]:,.2f}", f"{px['Close'].pct_change().iloc[-1]*100:+.2f}% 1d")
cols[1].metric(f"Realized {rv_window}d", fmt(rv_now.iloc[-1]), f"{m.percentile_rank(rv_now):.0f}th pct 1y", delta_color="off")
if implied is not None and not implied.dropna().empty:
    iv_now = float(implied.dropna().iloc[-1])
    cols[2].metric(f"Implied 30d", fmt(iv_now), f"{m.percentile_rank(implied):.0f}th pct 1y", delta_color="off")
    cols[3].metric("IV - RV (pts)", fmt(iv_now - rv_now.iloc[-1], ""),
                   "options rich" if iv_now > rv_now.iloc[-1] else "options cheap", delta_color="off")
else:
    iv_now = np.nan
    cols[2].metric("Implied 30d", "no CBOE index")

tabs = st.tabs(["Overview", "Realized vs implied", "Vol cone", "Term structure", "Skew", "Commodity curve",
                "VIX replication", "Forecasting", "Crude research", "Harvesting vol"])

# ---------- Overview ----------

with tabs[0]:
    st.subheader("Cross-asset snapshot")
    rows = []
    for a in UNIVERSE:
        p = ohlc(a.underlying, start, offline)
        if p.empty:
            continue
        iv = ohlc(a.vol_index, start, offline)["Close"] if a.vol_index else None
        rows.append(m.summary_row(a.name, a.asset_class, p, iv))
    table = pd.DataFrame(rows).set_index("Asset")
    pct = lambda label: st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.0f")
    st.dataframe(
        table,
        use_container_width=True,
        column_config={
            "Spot": st.column_config.NumberColumn(format="%.2f"),
            "1d %": st.column_config.NumberColumn(format="%+.2f%%"),
            "RV 21d": st.column_config.NumberColumn(format="%.1f%%"),
            "RV 63d": st.column_config.NumberColumn(format="%.1f%%"),
            "IV 30d": st.column_config.NumberColumn(format="%.1f%%"),
            "IV - RV": st.column_config.NumberColumn(format="%+.1f"),
            "RV 21d pct (1y)": pct("RV pct (1y)"),
            "IV pct (1y)": pct("IV pct (1y)"),
        },
    )
    st.caption("Vols annualized in %. Percentiles over the trailing year. IV from the CBOE 30-day index when one exists.")
    st.plotly_chart(charts.price_series(px["Close"], asset.name), use_container_width=True)

# ---------- Realized vs implied ----------

with tabs[1]:
    rv_table = m.realized_vol_table(px, rv_window)
    st.plotly_chart(charts.realized_vs_implied(rv_table, implied, asset.vol_index), use_container_width=True)
    with st.expander("About the estimators"):
        st.markdown(
            "- **Close-to-close**: standard deviation of log returns. Noisy, captures overnight gaps.\n"
            "- **Parkinson**: uses the daily high-low range. About 5x more efficient, ignores gaps.\n"
            "- **Garman-Klass**: adds open and close to the range. Assumes no drift, no gaps.\n"
            "- **Yang-Zhang**: combines overnight, open-to-close and Rogers-Satchell. Robust to both."
        )
    if implied is not None and not implied.dropna().empty:
        vrp = m.vol_risk_premium(implied, px, 21)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg VRP (vol pts)", fmt(vrp["vrp"].mean(), "", 2))
        c2.metric("VRP positive", fmt((vrp["vrp"] > 0).mean() * 100, "%", 0))
        c3.metric("Worst month", fmt(vrp["vrp"].min(), "", 1), vrp["vrp"].idxmin().strftime("%b %Y"), delta_color="off")
        c4.metric("Avg implied / realized", fmt((vrp["implied"] / vrp["forward_realized"]).mean(), "x", 2))
        st.plotly_chart(charts.vrp_history(vrp, asset.vol_index), use_container_width=True)
        st.plotly_chart(charts.vrp_spread(vrp), use_container_width=True)
        st.caption("Positive on average: sellers of index variance are paid a premium, with sharp losses in crises. "
                   "The series ends 21 trading days before the last date because the forward window must be complete.")
    else:
        st.info("No CBOE implied vol index for this asset on free data. See the Skew tab for a live implied vol from the option chain.")

# ---------- Vol cone ----------

with tabs[2]:
    est_name = st.radio("Estimator", list(m.ESTIMATORS), horizontal=True, index=0)
    cone, current = m.vol_cone(px, estimator=m.ESTIMATORS[est_name])
    st.plotly_chart(charts.vol_cone(cone, current, iv_now), use_container_width=True)
    show = cone.copy()
    show.insert(0, "current", current)
    st.dataframe(show.round(1), use_container_width=True)
    st.caption("Short windows are wider because they are noisier. Current realized near the top of the cone "
               "with implied below it means the market expects vol to mean-revert.")

# ---------- Term structure ----------

with tabs[3]:
    st.subheader("S&P 500 implied vol curve (CBOE VIX family)")
    ts_closes = closes(tuple(VIX_TERM) + tuple(VIX_EXTRAS), start, offline)
    if ts_closes.empty:
        st.warning("VIX term structure unavailable.")
    else:
        snap = m.term_structure_snapshot(ts_closes, VIX_TERM)
        labels = {"^VIX9D": "VIX9D", "^VIX": "VIX", "^VIX3M": "VIX3M", "^VIX6M": "VIX6M", "^VIX1Y": "VIX1Y"}
        c1, c2 = st.columns([1, 1])
        c1.plotly_chart(charts.term_structure(snap, labels), use_container_width=True)
        ratio = m.term_structure_history(ts_closes, "^VIX", "^VIX3M")
        c2.plotly_chart(charts.ratio_history(ratio, "VIX / VIX3M: above 1 the curve is inverted"), use_container_width=True)
        slope = snap.set_index("index")["vol"]
        if "^VIX3M" in slope and "^VIX" in slope:
            state = "contango (normal)" if slope["^VIX3M"] > slope["^VIX"] else "backwardation (stress)"
            st.markdown(f"**Curve state:** {state}. VIX3M - VIX = {slope['^VIX3M'] - slope['^VIX']:+.2f} pts. "
                        f"Inverted on {(ratio > 1).mean()*100:.0f}% of days since {ratio.index[0].year}.")
        c3, c4 = st.columns(2)
        if "^VVIX" in ts_closes:
            c3.plotly_chart(charts.single_series(ts_closes["^VVIX"].dropna(), "VVIX: implied vol of the VIX"), use_container_width=True)
        if "^SKEW" in ts_closes:
            c4.plotly_chart(charts.single_series(ts_closes["^SKEW"].dropna(), "CBOE SKEW: tail-risk pricing (100 = lognormal)"), use_container_width=True)
        st.caption("VIX1Y has no history on Yahoo, so only its latest print is used in the curve.")

# ---------- Skew ----------

with tabs[4]:
    if not asset.option_proxy:
        st.info("No optionable proxy configured.")
    else:
        target = st.select_slider("Target expiry (days)", [14, 30, 60, 90, 180], value=30)
        if offline:
            raw, expiry, spot = pd.DataFrame(), pd.NaT, float("nan")
        else:
            raw, expiry, spot = chain(asset.option_proxy, target)
        cleaned, info = clean_chain(raw, spot, expiry) if not raw.empty else (pd.DataFrame(), {})
        if offline:
            st.info(f"Live {asset.option_proxy} smile: switch offline off in the sidebar to read the chain. "
                    "The daily snapshot history below is committed to the repo.")
        elif cleaned.empty:
            st.warning(f"No usable quotes on {asset.option_proxy} right now.")
        else:
            sk = skew_metrics(cleaned)
            if info["stale"]:
                st.warning("Market closed: most quotes have no bid/ask, IVs are computed from last trades and may be stale.")
            st.markdown(f"**{asset.option_proxy}** spot {info['spot']:,.2f}, implied forward {info['forward']:,.2f}, "
                        f"expiry {info['expiry'].date()} ({info['days']} days), {len(cleaned)} OTM quotes.")
            c = st.columns(5)
            c[0].metric("ATM IV", fmt(sk.get("atm_iv")))
            c[1].metric("25d put", fmt(sk.get("put25")))
            c[2].metric("25d call", fmt(sk.get("call25")))
            c[3].metric("25d RR (pts)", fmt(sk.get("rr25"), ""), "call - put", delta_color="off")
            c[4].metric("25d BF (pts)", fmt(sk.get("bf25"), ""), "wings - ATM", delta_color="off")
            c1, c2 = st.columns(2)
            c1.plotly_chart(charts.smile_by_moneyness(cleaned, info), use_container_width=True)
            c2.plotly_chart(charts.smile_by_delta(cleaned), use_container_width=True)
            st.caption("Equity indices show put skew (negative risk reversal): crash protection is bid. "
                       "Crude and gold often show the opposite or a symmetric smile, since supply shocks push prices up. "
                       "IVs are inverted from mid prices with Black-76 on the put-call-parity forward, not Yahoo's field.")
            with st.expander("Chain"):
                st.dataframe(cleaned[["type", "strike", "moneyness", "mid", "bid", "ask", "iv_pct", "delta", "volume", "openInterest"]]
                             .rename(columns={"iv_pct": "iv %"}).round(3), use_container_width=True, hide_index=True)
    hist = skew_history_df()
    if not hist.empty:
        st.subheader("Skew history from daily snapshots")
        h30 = hist[hist["tenor_target"] == 30]
        metric = st.radio("Metric", ["rr25", "bf25", "atm_iv", "skew_90_110"], horizontal=True)
        st.plotly_chart(charts.skew_history(h30, metric), use_container_width=True)
        st.caption(f"{hist['date'].nunique()} snapshot day(s) so far, captured by `scripts/snapshot_skew.py` "
                   "(GitHub Actions, weekdays after the US close). Nobody sells this history for free, so it is built here.")

# ---------- Commodity curve ----------

with tabs[5]:
    commo = [a for a in UNIVERSE if a.futures_root]
    pick = st.selectbox("Commodity", [a.name for a in commo],
                        index=[a.name for a in commo].index(asset.name) if asset.futures_root else 0)
    a = BY_NAME[pick]
    cv = curve(a.futures_root, a.exchange, offline)
    if cv.empty or len(cv) < 2:
        st.warning("No cached curve for this root: switch offline off to read it from Yahoo." if offline
                   else "Futures curve unavailable on Yahoo for this root.")
    else:
        front, second = cv.iloc[0], cv.iloc[1]
        months = max((second["expiry_month"] - front["expiry_month"]).days / 30.4, 1)
        roll = (front["price"] / second["price"]) ** (12 / months) - 1
        state = "backwardation" if second["price"] < front["price"] else "contango"
        c1, c2, c3 = st.columns(3)
        c1.metric("Front contract", f"{front['price']:,.2f}", front["contract"])
        c2.metric("Curve", state.capitalize())
        c3.metric("Roll yield M1/M2 (ann.)", f"{roll*100:+.1f}%",
                  "long futures earn the roll" if roll > 0 else "long futures pay the roll", delta_color="off")
        st.plotly_chart(charts.futures_curve(cv, a.name), use_container_width=True)
        st.caption("Backwardation: spot scarcity, front month above deferred, positive roll yield for longs. "
                   "In crude, backwardation usually comes with elevated OVX; contango with calmer vol.")

# ---------- VIX replication ----------

with tabs[6]:
    st.subheader("Rebuilding the VIX from SPY options with the CBOE methodology")
    st.markdown(
        "The VIX is the square root of a 30-day **variance swap rate**, not an average of implied vols: "
        "a strip of OTM options weighted by $1/K^2$, for two expiries bracketing 30 days, interpolated in total variance. "
        "$\\sigma^2 = \\frac{2}{T}\\sum_i \\frac{\\Delta K_i}{K_i^2} e^{RT} Q(K_i) - \\frac{1}{T}\\left(\\frac{F}{K_0}-1\\right)^2$"
    )
    res = {"error": "Live SPY chain: switch offline off in the sidebar to run the replication."} if offline else vix_replication("SPY")
    if offline:
        st.info(res["error"])
    elif "error" in res:
        st.warning(res["error"])
    else:
        vix_series = ohlc("^VIX", start, offline)["Close"]
        official = float(vix_series.iloc[-1])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Replicated from SPY", f"{res['vix']:.2f}")
        c2.metric("Official VIX (last close)", f"{official:.2f}", f"{res['vix'] - official:+.2f} gap", delta_color="off")
        c3.metric("Near term", f"{100*np.sqrt(res['near'].variance):.2f}%", f"{res['near'].expiry.date()} ({res['near'].days}d)", delta_color="off")
        c4.metric("Next term", f"{100*np.sqrt(res['next'].variance):.2f}%", f"{res['next'].expiry.date()} ({res['next'].days}d)", delta_color="off")
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.vix_strip(res["near"].strip, f"Near-term strip, F={res['near'].forward:.2f}, K0={res['near'].k0:.0f}, {len(res['near'].strip)} strikes"), use_container_width=True)
        c2.plotly_chart(charts.vix_strip(res["next"].strip, f"Next-term strip, F={res['next'].forward:.2f}, K0={res['next'].k0:.0f}, {len(res['next'].strip)} strikes"), use_container_width=True)
        st.markdown(
            f"**Puts carry {res['near'].put_share*100:.0f}% of the near-term variance** although they are half the strikes: "
            "that is the skew inside the VIX. Residual gap vs the official print comes from SPY vs SPX (dividends, "
            "delayed quotes, slightly different expiries), and the official number is the previous close while the chain is live."
        )

# ---------- Forecasting ----------

with tabs[7]:
    st.subheader("Does the option market forecast vol better than the past does?")
    st.caption("Expanding window, refit monthly, 21-day horizon, from two years after the start date. "
               "HAR-RV (Corsi 2009) on log Parkinson / 5d / 22d realized; GARCH(1,1) by MLE; the CBOE index as a direct forecast; HAR with log implied added.")
    with st.spinner("Fitting models..."):
        fc, ev = forecasts(asset.underlying, asset.vol_index, start, offline)
    if fc.empty:
        st.warning("Not enough history.")
    else:
        best_rmse, best_qlike = ev["RMSE (pts)"].idxmin(), ev["QLIKE"].idxmin()
        st.dataframe(ev.style.format({"Bias (pts)": "{:+.2f}", "RMSE (pts)": "{:.2f}", "QLIKE": "{:.3f}",
                                      "MZ alpha": "{:+.2f}", "MZ beta": "{:.2f}", "MZ R2": "{:.3f}"})
                     .highlight_min(subset=["RMSE (pts)", "QLIKE"], color="#cde2fb"), use_container_width=True)
        models = [c for c in fc.columns if c != "target"]
        st.plotly_chart(charts.forecast_lines(fc, models), use_container_width=True)
        pick = st.selectbox("Mincer-Zarnowitz plot", models, index=models.index("Implied") if "Implied" in models else 0)
        st.plotly_chart(charts.mz_scatter(fc, pick), use_container_width=True)
        msg = [f"Lowest RMSE: **{best_rmse}**. Lowest QLIKE: **{best_qlike}**."]
        if "Implied" in ev.index:
            b = ev.loc["Implied", "Bias (pts)"]
            msg.append(f"Implied overshoots realized by **{b:+.1f} pts** on average: that bias is the variance risk premium, "
                       "and the MZ beta below 1 says a point of implied is worth less than a point of future realized. "
                       "Correcting the bias (HAR+IV) is what a desk's forecast actually does.")
        st.markdown(" ".join(msg))
        st.caption("Targets overlap day to day, so treat differences of a few hundredths as noise. QLIKE is the loss "
                   "used in the forecasting literature (Patton 2011) because it is robust to a noisy realized proxy.")

# ---------- Crude research ----------

with tabs[8]:
    st.subheader("WTI: the Samuelson effect and what the curve says about vol")
    with st.spinner("Loading contract histories..."):
        cs = crude_contracts("CL", "NYM", offline)
    if len(cs) < 5:
        st.warning("Not enough contract histories.")
    else:
        today_cs = crude.samuelson_same_date(cs)
        panel = crude.samuelson_panel(cs)
        pooled = crude.samuelson_curve(panel)
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.samuelson_today(today_cs), use_container_width=True)
        c2.plotly_chart(charts.samuelson_pooled(pooled), use_container_width=True)
        f, b = today_cs.iloc[0], today_cs.iloc[min(11, len(today_cs) - 1)]
        st.markdown(
            f"**Samuelson effect:** today the front contract ({f['contract']}, {f['ttm_months']:.1f} months) realizes "
            f"**{f['rv']:.0f}%** vs **{b['rv']:.0f}%** for {b['contract']} at {b['ttm_months']:.0f} months. "
            "The front of the curve absorbs inventory and OPEC news; the back is anchored by the marginal cost of production. "
            "Pooled across every listed contract since 2018 the same decay appears, with a bump at the long end "
            "where quotes are thin and 2020 dominates the sample."
        )
        with st.expander("Pooled table"):
            st.dataframe(pooled.round(1), use_container_width=True)

        st.subheader("Curve shape vs implied vol and the crude VRP")
        spread, near, far = crude.dec_dec_spread(cs)
        ovx = ohlc("^OVX", start, offline)["Close"]
        clpx = ohlc("CL=F", start, offline)
        res = crude.curve_vs_vol(spread, ovx, clpx)
        st.plotly_chart(charts.single_series(spread, f"{near[:5]} / {far[:5]} calendar spread (positive = backwardation)", "%"), use_container_width=True)
        c1, c2 = st.columns([1, 1])
        c1.plotly_chart(charts.spread_vs_vol(res["panel"]), use_container_width=True)
        with c2:
            st.markdown("**By curve regime**")
            st.dataframe(res["by_regime"].rename(columns={"days": "Days", "avg_implied": "Avg OVX", "avg_fwd_rv": "Avg fwd RV 21d",
                                                           "avg_vrp": "Avg VRP", "vrp_positive": "VRP > 0 (%)"}).round(1), use_container_width=True)
            by = res["by_regime"]
            st.markdown(
                f"Correlation between spread and OVX: **{res['corr_spread_implied']:+.2f}**. "
                "The textbook link (backwardation = scarcity = high vol) does not survive this sample: 2020 put the curve in "
                "super-contango with OVX above 100, and that episode dominates. What does hold is the **premium**: "
                f"selling crude vol earned **{by.loc['Backwardation', 'avg_vrp']:+.1f} pts** on average in backwardation vs "
                f"**{by.loc['Contango', 'avg_vrp']:+.1f} pts** in contango, positive {by.loc['Backwardation', 'vrp_positive']:.0f}% vs "
                f"{by.loc['Contango', 'vrp_positive']:.0f}% of days. Contango is when the crude vol seller gets hurt."
            )

# ---------- Harvesting vol ----------

with tabs[9]:
    st.subheader("How the premium is actually earned, what it costs, and when it is priced")
    if not asset.vol_index:
        st.info("This section needs a CBOE implied vol index. Select the S&P 500, Nasdaq 100, WTI or gold.")
    else:
        # --- 1. delta-hedged straddle
        st.markdown("#### 1. Short 1-month ATM straddle, delta-hedged daily at the implied vol")
        c1, c2 = st.columns(2)
        cost_bps = c1.slider("Hedge cost (bps per rebalance)", 0.0, 5.0, 1.0, 0.5)
        spread_pts = c2.slider("Option bid-ask (vol pts below mid when selling)", 0.0, 2.0, 0.5, 0.25)
        tr = straddle_backtest(asset.underlying, asset.vol_index, offline, cost_bps, spread_pts)
        ss = hedging.straddle_stats(tr)
        c = st.columns(6)
        c[0].metric("Trades", ss["trades"])
        c[1].metric("Avg P&L / trade", fmt(ss["avg_pnl_pct"], "%", 2), f"{ss['avg_pnl_vol_pts']:+.1f} vol pts", delta_color="off")
        c[2].metric("Hit ratio", fmt(ss["hit_ratio"], "%", 0))
        c[3].metric("Sharpe (net)", fmt(ss["sharpe"], "", 2))
        c[4].metric("Worst trade", fmt(ss["worst_pct"], "%", 1), ss["worst_date"].strftime("%b %Y"), delta_color="off")
        c[5].metric("Max drawdown", fmt(ss["max_drawdown_pct"], "%", 1))
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.straddle_cum(tr), use_container_width=True)
        c2.plotly_chart(charts.straddle_vs_gamma_theta(tr), use_container_width=True)
        st.plotly_chart(charts.pnl_vs_realized(tr), use_container_width=True)
        st.markdown(
            f"The straddle sold at the {asset.vol_index} level earns on average **{ss['avg_premium_pct']:.1f}% of spot in premium** and keeps "
            f"**{ss['avg_pnl_pct']:.2f}%** after the hedge and {ss['avg_cost_pct']:.2f}% of costs. The gamma-theta identity "
            f"$\\tfrac{{1}}{{2}}\\sum_t \\Gamma_t S_t^2(\\sigma_{{impl}}^2\\,dt - r_t^2)$ explains **{ss['gamma_theta_corr']*100:.0f}%** of the trade-by-trade P&L: "
            "the seller is paid the gap between implied and realized variance, weighted by gamma. The scatter against a variance swap of "
            "the same vega shows the path dependence: same realized-minus-implied, very different P&L depending on where spot sat when gamma was large."
        )

        # --- 2. forward vol and FOMC
        st.markdown("#### 2. Forward vol and the price of the next FOMC")
        ts = closes(tuple(VIX_TERM), "2011-01-01", offline)
        spx_close = ohlc("^GSPC", "2011-01-01", offline)["Close"]
        fc_curve = events.forward_curve_history(ts)
        ev_hist = events.event_premium_history(ts, spx_close)
        ev_tab = events.event_table(ev_hist, spx_close)
        nxt = events.next_event(pd.Timestamp.today().normalize())
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(charts.forward_curve(fc_curve), use_container_width=True)
        with c2:
            st.plotly_chart(charts.event_implied_vs_realized(ev_tab), use_container_width=True)
        summ = events.event_premium_summary(ev_hist)
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg implied FOMC move (term structure)", fmt(ev_tab["implied_move_pct"].mean(), "%", 2), f"{len(ev_tab)} meetings since 2015", delta_color="off")
        c2.metric("Avg realized |move| on FOMC day", fmt(ev_tab["realized_abs_move_pct"].mean(), "%", 2))
        c3.metric("VIX9D − forward 9d-30d, FOMC inside vs not", f"{summ.loc['FOMC within 9 days', 'avg_excess_vol_pts']:+.2f} vs {summ.loc['No FOMC within 9 days', 'avg_excess_vol_pts']:+.2f} pts")
        st.markdown(
            "Forward vol between two tenors is $\\sigma_f^2 = (T_2\\sigma_2^2 - T_1\\sigma_1^2)/(T_2 - T_1)$. When a FOMC sits inside the "
            "9-day window, VIX9D carries its variance on top of the forward: the excess is the market's price for the meeting. "
            "**Event vol is not overpriced the way generic vol is**: FOMC days realize about what is implied, while the month as a whole "
            "realizes several points less than the VIX. The carry lives in the days with no scheduled news. "
            "The term-structure read is floored at zero when the 9-day tenor sits below the forward (steep contango), so it understates in calm years."
        )
        if nxt is not None and asset.option_proxy and offline:
            st.caption(f"Live implied move for the {nxt.date()} FOMC: switch offline off to read the {asset.option_proxy} chain.")
        elif nxt is not None and asset.option_proxy:
            with st.spinner(f"Reading {asset.option_proxy} expiries around the {nxt.date()} FOMC..."):
                live = live_event_move(asset.option_proxy, nxt)
            if "error" in live:
                st.caption(f"Live read unavailable: {live['error']}")
            else:
                c = st.columns(4)
                c[0].metric(f"Next FOMC {nxt.date()}", f"{live['event_move_base_30d_pct']:.2f}%", "implied move, base = 30d ATM", delta_color="off")
                c[1].metric("Same, base = next-expiry forward", f"{live['event_move_base_fwd_pct']:.2f}%", "polluted by other events", delta_color="off")
                c[2].metric(f"ATM IV expiring {live['expiry_1'].date()}", f"{live['atm_iv_1']:.1f}%", f"vs {live['atm_iv_30d']:.1f}% at 30d", delta_color="off")
                c[3].metric("Normal daily move (30d ATM)", f"{live['base_daily_move_30d_pct']:.2f}%")
                st.caption("From the ATM vols of the two expiries on and after the event: total variance to the first expiry minus the normal days "
                           "leaves the event's variance. Delayed quotes; rerun during US hours.")

        # --- 3. the gap signal
        st.markdown("#### 3. Is the implied-minus-model gap a timing signal?")
        fc, _ = forecasts(asset.underlying, asset.vol_index, start, offline)
        if "HAR+IV" in fc:
            cond = sg.conditional_premium(fc)
            twg = sg.trades_with_gap(tr, fc)
            stats_tbl, paths = sg.strategy_comparison(twg)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Forward VRP by gap bucket (daily)**")
                st.dataframe(cond.round(2), use_container_width=True)
            with c2:
                st.markdown("**Straddle P&L by gap at trade date**")
                st.dataframe(sg.pnl_by_gap(twg).round(2), use_container_width=True)
            st.plotly_chart(charts.strategy_paths(paths), use_container_width=True)
            st.dataframe(stats_tbl.round(2), use_container_width=True)
            best = stats_tbl["sharpe"].idxmax()
            st.markdown(
                f"The premium grows with the gap ({cond['avg_vrp_pts'].iloc[2]:.1f} pts in the 2-4 bucket, {cond['avg_vrp_pts'].iloc[-1]:.1f} pts above 6) "
                f"but the best risk-adjusted rule is **{best}**: skipping months loses more carry than it avoids losses. "
                "The gap tells you how much you are paid, not when the crash comes. That is the honest answer, and the reason systematic "
                "vol sellers size by regime rather than switch on and off."
            )
        else:
            st.info("Gap signal needs a CBOE implied index with history.")
