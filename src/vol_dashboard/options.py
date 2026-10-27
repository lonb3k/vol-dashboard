"""Option-chain analytics.

Yahoo's `impliedVolatility` field is computed against spot with no forward adjustment, so
calls and puts at the same strike disagree by several vol points. We rebuild the surface the
way a desk does:

1. Mid price from bid/ask (falls back to last trade when the market is closed).
2. Implied forward from put-call parity on the strikes nearest the money: F = K + (C - P) / DF.
3. Implied vol per OTM option by inverting Black-76 on that forward.
4. Forward delta from Black-76, then the usual desk quotes: ATM, 25d risk reversal, 25d butterfly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

RISK_FREE = 0.04  # flat rate, good enough for 1-3 month tenors on ETF options


# ---------- Black-76 ----------

def black76_price(F: float, K: float, t: float, sigma: float, is_call: bool, df: float) -> float:
    if sigma <= 0 or t <= 0:
        return df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)
    sq = sigma * np.sqrt(t)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * t) / sq
    d2 = d1 - sq
    if is_call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_delta(F: float, K: np.ndarray, t: float, sigma: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Forward (undiscounted) delta."""
    sq = sigma * np.sqrt(t)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * t) / sq
    nd1 = norm.cdf(d1)
    return np.where(is_call, nd1, nd1 - 1)


def implied_vol(price: float, F: float, K: float, t: float, is_call: bool, df: float) -> float:
    intrinsic = df * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    if not np.isfinite(price) or price <= intrinsic + 1e-6:
        return float("nan")
    f = lambda s: black76_price(F, K, t, s, is_call, df) - price
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-6, maxiter=100)
    except ValueError:
        return float("nan")


# ---------- chain construction ----------

def _mid(df: pd.DataFrame) -> tuple[pd.Series, bool]:
    """Mid from bid/ask; falls back to last trade when there are no quotes. Returns (mid, stale)."""
    has_quote = (df["bid"] > 0) & (df["ask"] > 0)
    mid = pd.Series(np.where(has_quote, (df["bid"] + df["ask"]) / 2, df["last"]), index=df.index)
    stale = has_quote.mean() < 0.5
    return mid, stale


def implied_forward(chain: pd.DataFrame, mid: pd.Series, spot: float, df: float, band: float = 0.03) -> float:
    """Put-call parity on near-the-money strikes: C - P = DF * (F - K)."""
    tmp = chain.assign(mid=mid)
    tmp = tmp[(tmp["mid"] > 0) & tmp["strike"].between(spot * (1 - band), spot * (1 + band))]
    pivot = tmp.pivot_table(index="strike", columns="type", values="mid")
    if not {"call", "put"} <= set(pivot.columns):
        return spot * np.exp(RISK_FREE * 0)  # no pairs: assume forward = spot
    pairs = pivot.dropna()
    if pairs.empty:
        return spot
    fwd = pairs.index.values + (pairs["call"].values - pairs["put"].values) / df
    return float(np.median(fwd))


def clean_chain(chain: pd.DataFrame, spot: float, expiry: pd.Timestamp,
                moneyness_band: tuple[float, float] = (0.7, 1.3)) -> tuple[pd.DataFrame, dict]:
    """OTM options with recomputed IV and delta.

    Returns (chain, info) where info holds forward, time to expiry, discount factor and a
    `stale` flag set when most quotes are missing (market closed).
    """
    empty = pd.DataFrame(), {}
    if chain.empty or not np.isfinite(spot):
        return empty

    t = max((expiry - pd.Timestamp.today().normalize()).days, 1) / 365
    disc = float(np.exp(-RISK_FREE * t))
    mid, stale = _mid(chain)
    F = implied_forward(chain, mid, spot, disc)

    df = chain.assign(mid=mid)
    df = df[df["mid"] > 0]
    df["moneyness"] = df["strike"] / F
    df = df[df["moneyness"].between(*moneyness_band)]
    is_put, is_call = df["type"] == "put", df["type"] == "call"
    df = df[(is_put & (df["strike"] <= F)) | (is_call & (df["strike"] >= F))].copy()
    if df.empty:
        return empty

    df["iv"] = [
        implied_vol(p, F, k, t, c, disc)
        for p, k, c in zip(df["mid"], df["strike"], df["type"] == "call")
    ]
    df = df.dropna(subset=["iv"])
    df = df[(df["iv"] > 0.01) & (df["iv"] < 3.0)].copy()
    if df.empty:
        return empty

    df["iv_pct"] = df["iv"] * 100
    df["delta"] = black76_delta(F, df["strike"].values, t, df["iv"].values, (df["type"] == "call").values)
    df["spread_pct"] = np.where(df["bid"] > 0, (df["ask"] - df["bid"]) / df["mid"] * 100, np.nan)
    df = df.sort_values("strike").reset_index(drop=True)
    info = {"forward": F, "spot": spot, "t_years": t, "discount": disc, "stale": stale,
            "expiry": expiry, "days": int(round(t * 365))}
    return df, info


# ---------- desk skew quotes ----------

def _iv_at_delta(df: pd.DataFrame, target: float) -> float:
    d = df.dropna(subset=["delta"]).sort_values("delta")
    if len(d) < 2 or not (d["delta"].min() <= target <= d["delta"].max()):
        return float("nan")
    return float(np.interp(target, d["delta"].values, d["iv_pct"].values))


def _iv_at_moneyness(df: pd.DataFrame, m: float) -> float:
    d = df.sort_values("moneyness")
    if len(d) < 2 or not (d["moneyness"].min() <= m <= d["moneyness"].max()):
        return float("nan")
    return float(np.interp(m, d["moneyness"].values, d["iv_pct"].values))


def skew_metrics(df: pd.DataFrame) -> dict:
    """
    - atm_iv: IV at the forward (moneyness 1.0), interpolated between the nearest put and call.
    - rr25: 25-delta risk reversal = IV(25d call) - IV(25d put). Negative = put skew (equities).
    - bf25: 25-delta butterfly = mean(wings) - ATM. Smile curvature / tail richness.
    - skew_90_110: IV(90% put) - IV(110% call), the simple moneyness skew.
    """
    if df.empty:
        return {}
    puts, calls = df[df["type"] == "put"], df[df["type"] == "call"]
    put25, call25 = _iv_at_delta(puts, -0.25), _iv_at_delta(calls, 0.25)
    put10, call10 = _iv_at_delta(puts, -0.10), _iv_at_delta(calls, 0.10)
    # ATM = IV at the forward (moneyness 1.0), interpolated across the put/call boundary
    both = df.sort_values("moneyness")
    atm = float(np.interp(1.0, both["moneyness"].values, both["iv_pct"].values))
    return {
        "atm_iv": atm,
        "put25": put25, "call25": call25,
        "put10": put10, "call10": call10,
        "rr25": call25 - put25,
        "bf25": 0.5 * (call25 + put25) - atm,
        "skew_90_110": _iv_at_moneyness(puts, 0.90) - _iv_at_moneyness(calls, 1.10),
    }
