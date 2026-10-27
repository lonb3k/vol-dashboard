"""Commodity-specific volatility: the Samuelson effect and the curve / vol link on WTI.

Samuelson (1965): a futures contract's volatility rises as it approaches expiry, because
news about near-term supply and demand moves the front of the curve much more than the back.
Expired contracts are not available on free data, but every listed contract carries its full
history, so pooling realized vol by time-to-maturity across ~30 live contracts gives the curve.

Curve shape: the December-to-December calendar spread (e.g. CLZ26 vs CLZ27) is the trader's
standard gauge of the curve, and both legs have years of history. Backwardation (near > far)
means physical scarcity; it is usually accompanied by higher OVX.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .data import DATA_DIR
from .metrics import ANN, forward_realized
from .universe import FUTURES_MONTH_CODES

CONTRACT_DIR = DATA_DIR / "contracts"
CONTRACT_DIR.mkdir(exist_ok=True)

TTM_BUCKETS = [0, 1, 2, 3, 6, 12, 24, 48, 120]  # months to expiry


def contract_expiry(root: str, year: int, month: int) -> pd.Timestamp:
    """Last trading day. CL: 3 business days before the 25th of the month preceding delivery."""
    prev = pd.Timestamp(year=year, month=month, day=1) - pd.offsets.MonthBegin(1)
    d25 = pd.Timestamp(year=prev.year, month=prev.month, day=25)
    if d25.weekday() >= 5:
        d25 = d25 - pd.offsets.BDay(1)
    if root == "CL":
        return d25 - pd.offsets.BDay(3)
    return pd.Timestamp(year=year, month=month, day=1) - pd.offsets.BDay(1)  # generic fallback


def list_contracts(root: str, exchange: str, n_months: int = 48) -> list[tuple[str, pd.Timestamp]]:
    today = dt.date.today()
    year, month = today.year, today.month
    out = []
    for _ in range(n_months):
        month += 1
        if month > 12:
            month, year = 1, year + 1
        code = f"{root}{FUTURES_MONTH_CODES[month - 1]}{str(year)[-2:]}.{exchange}"
        out.append((code, contract_expiry(root, year, month)))
    return out


def load_contract(code: str, offline: bool = False) -> pd.DataFrame:
    path = CONTRACT_DIR / f"{code.replace('.', '_')}.csv"
    if offline and path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    try:
        h = yf.Ticker(code).history(period="max", auto_adjust=False)
    except Exception:
        h = pd.DataFrame()
    if h.empty:
        return pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()
    h = h[["Open", "High", "Low", "Close", "Volume"]].copy()
    h.index = pd.to_datetime(h.index).tz_localize(None)
    h.index.name = "Date"
    h.to_csv(path)
    return h


def load_contracts(root: str, exchange: str, n_months: int = 48, offline: bool = False,
                   max_contracts: int = 30) -> dict[str, tuple[pd.Timestamp, pd.DataFrame]]:
    """{code: (expiry, ohlcv)} for every listed contract that has data."""
    out = {}
    for code, exp in list_contracts(root, exchange, n_months):
        h = load_contract(code, offline)
        if len(h) > 60:
            out[code] = (exp, h)
        if len(out) >= max_contracts:
            break
    return out


# ---------- Samuelson ----------

def samuelson_panel(contracts: dict, window: int = 21) -> pd.DataFrame:
    """Long panel: one row per contract-day with realized vol and months to expiry.
    Days with zero volume are dropped: a stale settlement looks like zero vol."""
    frames = []
    for code, (exp, h) in contracts.items():
        h = h[h["Volume"] > 0] if "Volume" in h else h
        r = np.log(h["Close"] / h["Close"].shift(1))
        rv = r.rolling(window).std() * np.sqrt(ANN) * 100
        ttm = (exp - h.index).days / 30.4
        frames.append(pd.DataFrame({"contract": code, "rv": rv, "ttm_months": ttm}, index=h.index).dropna())
    panel = pd.concat(frames)
    return panel[panel["ttm_months"] > 0]


def samuelson_curve(panel: pd.DataFrame) -> pd.DataFrame:
    labels = [f"{a}-{b}m" for a, b in zip(TTM_BUCKETS[:-1], TTM_BUCKETS[1:])]
    b = pd.cut(panel["ttm_months"], TTM_BUCKETS, labels=labels, right=False)
    g = panel.groupby(b, observed=True)["rv"]
    out = pd.DataFrame({"median_rv": g.median(), "mean_rv": g.mean(), "q25": g.quantile(0.25),
                        "q75": g.quantile(0.75), "obs": g.size(), "contracts": panel.groupby(b, observed=True)["contract"].nunique()})
    out.index.name = "months_to_expiry"
    return out


def samuelson_same_date(contracts: dict, lookback_days: int = 60) -> pd.DataFrame:
    """Cross-section today: realized vol over the last `lookback_days` for each contract vs its
    time to maturity. Removes the time-period effect that pooling across years leaves in."""
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)
    rows = []
    for code, (exp, h) in contracts.items():
        h = h[(h.index >= cutoff) & (h["Volume"] > 0)] if "Volume" in h else h[h.index >= cutoff]
        if len(h) < 15:
            continue
        r = np.log(h["Close"] / h["Close"].shift(1)).dropna()
        rows.append({"contract": code, "expiry": exp, "ttm_months": (exp - pd.Timestamp.today().normalize()).days / 30.4,
                     "rv": float(r.std() * np.sqrt(ANN) * 100), "days": len(r)})
    return pd.DataFrame(rows).sort_values("ttm_months").reset_index(drop=True)


# ---------- curve vs vol ----------

def dec_dec_spread(contracts: dict) -> tuple[pd.Series, str, str]:
    """(near/far - 1) in %, for the first two December contracts. Positive = backwardation."""
    decs = sorted([c for c in contracts if c[2] == "Z"], key=lambda c: contracts[c][0])
    if len(decs) < 2:
        return pd.Series(dtype=float), "", ""
    near, far = decs[0], decs[1]
    a, b = contracts[near][1]["Close"], contracts[far][1]["Close"]
    s = ((a / b - 1) * 100).dropna().rename("spread_pct")
    return s, near, far


def curve_vs_vol(spread: pd.Series, implied: pd.Series, px: pd.DataFrame, horizon: int = 21) -> dict:
    """Join curve state with OVX and the crude VRP; summarize by curve regime."""
    fwd = forward_realized(px, horizon)
    df = pd.DataFrame({"spread": spread, "implied": implied, "fwd_rv": fwd}).dropna(subset=["spread", "implied"])
    df["vrp"] = df["implied"] - df["fwd_rv"]
    df["regime"] = np.where(df["spread"] > 0, "Backwardation", "Contango")
    by = df.groupby("regime").agg(days=("spread", "size"), avg_implied=("implied", "mean"),
                                  avg_fwd_rv=("fwd_rv", "mean"), avg_vrp=("vrp", "mean"),
                                  vrp_positive=("vrp", lambda s: (s.dropna() > 0).mean() * 100))
    corr = float(df[["spread", "implied"]].corr().iloc[0, 1])
    return {"panel": df, "by_regime": by, "corr_spread_implied": corr}
