"""Replicate the VIX from an option chain with the CBOE methodology.

The VIX is not an average of implied vols. It is the square root of the 30-day variance swap
rate, obtained from a strip of OTM options weighted by 1/K^2 (the model-free implied variance
of Demeterfi, Derman, Kamal and Zou, 1999). The CBOE white paper spells out the recipe:

    sigma^2 = (2/T) * sum_i (dK_i / K_i^2) * e^{RT} * Q(K_i)  -  (1/T) * (F/K0 - 1)^2

for each of two expiries bracketing 30 days, then a linear interpolation in total variance
to exactly 30 days. We apply it to SPY options (free data) rather than SPX, so expect a
small gap from the official print: SPY carries dividends, quotes are delayed, and the two
expiries differ slightly from the CBOE's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from .options import RISK_FREE


@dataclass
class ExpiryVariance:
    expiry: pd.Timestamp
    days: int
    t_years: float
    forward: float
    k0: float
    variance: float            # annualized sigma^2
    strip: pd.DataFrame        # per-strike contributions
    put_share: float           # share of the strip from puts (skew footprint)


def _strip_variance(calls: pd.DataFrame, puts: pd.DataFrame, t: float, r: float) -> ExpiryVariance | None:
    """CBOE steps 1-3 for one expiry. `calls` and `puts` indexed by strike with bid/ask columns."""
    calls, puts = calls.sort_index(), puts.sort_index()
    both = calls.join(puts, how="inner", suffixes=("_c", "_p"))
    both = both[(both["bid_c"] > 0) & (both["bid_p"] > 0)]
    if both.empty:
        return None
    mid_c = (both["bid_c"] + both["ask_c"]) / 2
    mid_p = (both["bid_p"] + both["ask_p"]) / 2
    diff = (mid_c - mid_p).abs()
    k_star = diff.idxmin()
    F = k_star + np.exp(r * t) * (mid_c[k_star] - mid_p[k_star])
    strikes = np.sort(both.index.values)
    below = strikes[strikes < F]
    if len(below) == 0:
        return None
    K0 = below[-1]

    def select(df: pd.DataFrame, side: str) -> pd.DataFrame:
        """OTM strikes walking away from K0; stop after two consecutive zero bids."""
        if side == "put":
            cand = df[df.index < K0].sort_index(ascending=False)
        else:
            cand = df[df.index > K0].sort_index()
        keep, zeros = [], 0
        for k, row in cand.iterrows():
            if row["bid"] <= 0:
                zeros += 1
                if zeros >= 2:
                    break
                continue
            zeros = 0
            keep.append((k, (row["bid"] + row["ask"]) / 2, side))
        return pd.DataFrame(keep, columns=["strike", "mid", "side"])

    puts_sel = select(puts, "put")
    calls_sel = select(calls, "call")
    atm_mid = (mid_c[K0] + mid_p[K0]) / 2 if K0 in both.index else np.nan
    atm = pd.DataFrame([(K0, atm_mid, "atm")], columns=["strike", "mid", "side"])
    strip = pd.concat([puts_sel, atm, calls_sel]).dropna().sort_values("strike").reset_index(drop=True)
    if len(strip) < 5:
        return None

    k = strip["strike"].values
    dk = np.empty_like(k)
    dk[1:-1] = (k[2:] - k[:-2]) / 2
    dk[0] = k[1] - k[0]
    dk[-1] = k[-1] - k[-2]
    strip["dK"] = dk
    strip["contribution"] = dk / k**2 * np.exp(r * t) * strip["mid"]
    var = (2 / t) * strip["contribution"].sum() - (1 / t) * (F / K0 - 1) ** 2
    put_share = float(strip.loc[strip["side"] == "put", "contribution"].sum() / strip["contribution"].sum())
    return ExpiryVariance(pd.NaT, 0, t, float(F), float(K0), float(var), strip, put_share)


def _pick_expiries(expiries: list[pd.Timestamp], today: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Nearest expiry with >= 23 days (near term) and the next one after 30 days (next term)."""
    days = [(e, (e - today).days) for e in expiries]
    near = [e for e, d in days if 23 <= d <= 30]
    nxt = [e for e, d in days if 30 < d <= 45]
    if not near or not nxt:
        return None
    return max(near, key=lambda e: (e - today).days), min(nxt, key=lambda e: (e - today).days)


def replicate_vix(ticker: str = "SPY", r: float = RISK_FREE) -> dict:
    """Returns {'vix': float, 'near': ExpiryVariance, 'next': ExpiryVariance, 'spot': float} or {'error': ...}."""
    tk = yf.Ticker(ticker)
    today = pd.Timestamp.today().normalize()
    exps = [pd.Timestamp(e) for e in tk.options]
    picked = _pick_expiries(exps, today)
    if picked is None:
        return {"error": "No pair of expiries bracketing 30 days."}

    results = []
    for exp in picked:
        chain = tk.option_chain(exp.strftime("%Y-%m-%d"))
        calls = chain.calls.set_index("strike")[["bid", "ask"]]
        puts = chain.puts.set_index("strike")[["bid", "ask"]]
        days = (exp - today).days
        t = days / 365
        ev = _strip_variance(calls, puts, t, r)
        if ev is None:
            return {"error": f"Not enough quotes for {exp.date()} (market closed?)."}
        ev.expiry, ev.days = exp, days
        results.append(ev)
    near, nxt = results

    n1, n2, n30 = near.days, nxt.days, 30
    w1 = (n2 - n30) / (n2 - n1)
    w2 = (n30 - n1) / (n2 - n1)
    var30 = (near.t_years * near.variance * w1 + nxt.t_years * nxt.variance * w2) * (365 / 30)
    hist = tk.history(period="5d")
    spot = float(hist["Close"].iloc[-1]) if not hist.empty else float("nan")
    return {"vix": float(100 * np.sqrt(max(var30, 0))), "near": near, "next": nxt, "spot": spot,
            "weights": (w1, w2)}


def synthetic_chain(spot: float, sigma: float, t: float, r: float = 0.0, step: float = 1.0,
                    width: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flat-vol Black-Scholes chain for testing: the strip must return sigma^2."""
    from scipy.stats import norm
    F = spot * np.exp(r * t)
    strikes = np.arange(spot * (1 - width), spot * (1 + width), step)
    d1 = (np.log(F / strikes) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    disc = np.exp(-r * t)
    c = disc * (F * norm.cdf(d1) - strikes * norm.cdf(d2))
    p = disc * (strikes * norm.cdf(-d2) - F * norm.cdf(-d1))
    calls = pd.DataFrame({"bid": c, "ask": c}, index=pd.Index(strikes, name="strike"))
    puts = pd.DataFrame({"bid": p, "ask": p}, index=pd.Index(strikes, name="strike"))
    return calls, puts
