"""Volatility metrics: realized vol estimators, vol cone, vol risk premium, term structure.

All vols are annualized and expressed in percent (so 20.0 means 20% vol), matching how
CBOE indices are quoted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252
CONE_WINDOWS = [10, 21, 42, 63, 126, 252]
CONE_QUANTILES = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]


# ---------- realized vol estimators ----------

def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).dropna()


def rv_close_to_close(df: pd.DataFrame, window: int = 21) -> pd.Series:
    r = log_returns(df["Close"])
    return r.rolling(window).std() * np.sqrt(ANN) * 100


def rv_parkinson(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Uses the daily high-low range. More efficient than close-to-close, ignores overnight gaps."""
    hl = np.log(df["High"] / df["Low"]) ** 2
    var = hl.rolling(window).mean() / (4 * np.log(2))
    return np.sqrt(var * ANN) * 100


def rv_garman_klass(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Range-based estimator using open, high, low, close."""
    hl = np.log(df["High"] / df["Low"]) ** 2
    co = np.log(df["Close"] / df["Open"]) ** 2
    var = (0.5 * hl - (2 * np.log(2) - 1) * co).rolling(window).mean()
    return np.sqrt(var.clip(lower=0) * ANN) * 100


def rv_yang_zhang(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Handles overnight jumps and intraday drift. Standard choice on a vol desk."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    overnight = np.log(o / c.shift(1))
    open_close = np.log(c / o)
    rs = np.log(h / o) * np.log(h / c) + np.log(l / o) * np.log(l / c)  # Rogers-Satchell
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = (
        overnight.rolling(window).var()
        + k * open_close.rolling(window).var()
        + (1 - k) * rs.rolling(window).mean()
    )
    return np.sqrt(var.clip(lower=0) * ANN) * 100


ESTIMATORS = {
    "Close-to-close": rv_close_to_close,
    "Parkinson": rv_parkinson,
    "Garman-Klass": rv_garman_klass,
    "Yang-Zhang": rv_yang_zhang,
}


def realized_vol_table(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """All estimators side by side."""
    return pd.DataFrame({name: fn(df, window) for name, fn in ESTIMATORS.items()}).dropna(how="all")


# ---------- vol cone ----------

def vol_cone(df: pd.DataFrame, estimator=rv_close_to_close, windows=CONE_WINDOWS,
             lookback_days: int | None = 252 * 5) -> tuple[pd.DataFrame, pd.Series]:
    """Distribution of realized vol by horizon.

    Returns (cone, current): `cone` indexed by window with one column per quantile;
    `current` is the latest realized vol for each window.
    """
    if lookback_days:
        df = df.iloc[-(lookback_days + max(windows)):]
    rows, current = {}, {}
    for w in windows:
        s = estimator(df, w).dropna()
        if lookback_days:
            s = s.iloc[-lookback_days:]
        if s.empty:
            continue
        rows[w] = s.quantile(CONE_QUANTILES).values
        current[w] = s.iloc[-1]
    cone = pd.DataFrame(rows, index=[f"q{int(q*100)}" for q in CONE_QUANTILES]).T
    cone.index.name = "window"
    return cone, pd.Series(current, name="current")


# ---------- implied vs realized ----------

def forward_realized(df: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Realized vol over the *next* `horizon` days, aligned on the start date.

    This is what an implied vol quoted today is trying to forecast.
    """
    rv = rv_close_to_close(df, horizon)
    return rv.shift(-horizon)


def vol_risk_premium(implied: pd.Series, df: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """VRP = implied vol today minus realized vol over the following `horizon` days.

    Positive on average: option sellers are paid to carry variance risk.
    """
    fwd = forward_realized(df, horizon)
    out = pd.DataFrame({"implied": implied, "forward_realized": fwd}).dropna()
    out["vrp"] = out["implied"] - out["forward_realized"]
    return out


def percentile_rank(s: pd.Series, lookback: int = 252) -> float:
    """Where the last value sits within its trailing `lookback` distribution, in [0, 100]."""
    window = s.dropna().iloc[-lookback:]
    if window.empty:
        return float("nan")
    return float((window < window.iloc[-1]).mean() * 100)


# ---------- term structure ----------

def term_structure_snapshot(closes: pd.DataFrame, tenors: dict[str, int]) -> pd.DataFrame:
    """Latest level of each vol index with its tenor, sorted by tenor."""
    last = closes.ffill().iloc[-1]
    rows = [{"index": k, "tenor_days": d, "vol": last[k]} for k, d in tenors.items() if k in last.index]
    return pd.DataFrame(rows).sort_values("tenor_days").reset_index(drop=True)


def term_structure_history(closes: pd.DataFrame, short: str = "^VIX", long: str = "^VIX3M") -> pd.Series:
    """Ratio short / long tenor. Above 1 = inverted curve (backwardation), a stress signal."""
    if short not in closes or long not in closes:
        return pd.Series(dtype=float)
    return (closes[short] / closes[long]).dropna().rename("ratio")


def summary_row(name: str, asset_class: str, px: pd.DataFrame, implied: pd.Series | None) -> dict:
    """One line of the overview table."""
    rv21 = rv_close_to_close(px, 21)
    rv63 = rv_close_to_close(px, 63)
    row = {
        "Asset": name,
        "Class": asset_class,
        "Spot": float(px["Close"].iloc[-1]),
        "1d %": float(px["Close"].pct_change().iloc[-1] * 100),
        "RV 21d": float(rv21.iloc[-1]),
        "RV 63d": float(rv63.iloc[-1]),
        "RV 21d pct (1y)": percentile_rank(rv21),
        "IV 30d": np.nan,
        "IV - RV": np.nan,
        "IV pct (1y)": np.nan,
    }
    if implied is not None and not implied.dropna().empty:
        iv = float(implied.dropna().iloc[-1])
        row["IV 30d"] = iv
        row["IV - RV"] = iv - row["RV 21d"]
        row["IV pct (1y)"] = percentile_rank(implied)
    return row
