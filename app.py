"""Volatility dashboard: equity indices and commodities.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vol_dashboard import charts, metrics as m  # noqa: E402
from vol_dashboard.data import futures_curve, load_closes, load_ohlc, option_chain  # noqa: E402
from vol_dashboard.options import clean_chain, skew_metrics  # noqa: E402
from vol_dashboard.universe import BY_NAME, UNIVERSE, VIX_EXTRAS, VIX_TERM  # noqa: E402

st.set_page_config(page_title="Vol Dashboard", page_icon="📈", layout="wide")

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
def curve(root: str, exchange: str) -> pd.DataFrame:
    return futures_curve(root, exchange)


def fmt(x: float, suffix: str = "%", digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}f}{suffix}"


# ---------- sidebar ----------

with st.sidebar:
    st.title("Vol Dashboard")
    st.caption("Equity indices and commodities. Realized vs implied, cone, term structure, skew.")
    asset_name = st.selectbox("Asset", list(BY_NAME), index=0)
    start = st.date_input("History from", pd.Timestamp("2015-01-01")).strftime("%Y-%m-%d")
    rv_window = st.slider("Realized vol window (days)", 5, 126, 21)
    offline = st.toggle("Offline (use cached CSV extracts)", value=False)
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

tabs = st.tabs(["Overview", "Realized vs implied", "Vol cone", "Term structure", "Skew", "Commodity curve"])

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
        raw, expiry, spot = chain(asset.option_proxy, target)
        cleaned, info = clean_chain(raw, spot, expiry)
        if cleaned.empty:
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

# ---------- Commodity curve ----------

with tabs[5]:
    commo = [a for a in UNIVERSE if a.futures_root]
    pick = st.selectbox("Commodity", [a.name for a in commo],
                        index=[a.name for a in commo].index(asset.name) if asset.futures_root else 0)
    a = BY_NAME[pick]
    cv = curve(a.futures_root, a.exchange)
    if cv.empty or len(cv) < 2:
        st.warning("Futures curve unavailable on Yahoo for this root.")
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
