"""How the volatility risk premium is actually harvested: a short straddle, delta-hedged daily.

Sell a one-month ATM straddle on the index at the implied vol (the VIX), hedge the delta every
close with Black-Scholes at that same vol, hold to expiry. The realized P&L is

    premium - payoff + sum_t hedge_t * (S_{t+1} - S_t) - costs

and, to second order, the gamma-theta identity (El Karoui, Jeanblanc-Picque, Shreve 1998):

    P&L  ~  sum_t  1/2 * Gamma_t * S_t^2 * (sigma_impl^2 * dt - r_t^2)

so the seller earns the spread between implied and realized variance, weighted by gamma.
Gamma is largest when spot sits at the strike near expiry, which is why the same implied-minus-
realized gap can give very different P&Ls: the straddle is path dependent, a variance swap is not.

Units: everything in % of the initial spot, and in "vol points" (P&L divided by the straddle's
vega at inception), which is how a desk thinks about it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .metrics import ANN

HORIZON = 21


def _d1(S, K, sigma, tau):
    return (np.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))


def straddle_price(S, K, sigma, tau):
    d1 = _d1(S, K, sigma, tau)
    d2 = d1 - sigma * np.sqrt(tau)
    call = S * norm.cdf(d1) - K * norm.cdf(d2)
    put = K * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call + put


def straddle_delta(S, K, sigma, tau):
    return 2 * norm.cdf(_d1(S, K, sigma, tau)) - 1


def straddle_gamma(S, K, sigma, tau):
    return 2 * norm.pdf(_d1(S, K, sigma, tau)) / (S * sigma * np.sqrt(tau))


def straddle_vega(S, K, sigma, tau):
    """Per 1.00 of vol (multiply by 0.01 for one vol point)."""
    return 2 * S * norm.pdf(_d1(S, K, sigma, tau)) * np.sqrt(tau)


def hedge_one_trade(path: np.ndarray, sigma_impl: float, hedge_cost_bps: float = 1.0,
                    option_spread_vol_pts: float = 0.5) -> dict:
    """Short straddle on `path[0]`, expiring at `path[-1]`, hedged at each close.

    path: closes from trade date to expiry (HORIZON + 1 points). sigma_impl in decimal.
    Costs: proportional cost on every hedge trade, and the straddle sold `option_spread_vol_pts`
    below mid (half the bid-ask in vol).
    """
    n = len(path) - 1
    S0, K = path[0], path[0]
    T = n / ANN
    dt = 1 / ANN
    sigma_sold = sigma_impl - option_spread_vol_pts / 100
    premium = straddle_price(S0, K, sigma_sold, T)
    vega = straddle_vega(S0, K, sigma_impl, T) * 0.01  # per vol point

    hedge_pnl, cost, gamma_theta = 0.0, 0.0, 0.0
    pos = 0.0
    for t in range(n):
        tau = (n - t) / ANN
        S = path[t]
        target = straddle_delta(S, K, sigma_impl, tau)  # short straddle -> hold +delta of index
        cost += abs(target - pos) * S * hedge_cost_bps / 1e4
        pos = target
        dS = path[t + 1] - S
        hedge_pnl += pos * dS
        r = np.log(path[t + 1] / S)
        gamma_theta += 0.5 * straddle_gamma(S, K, sigma_impl, tau) * S**2 * (sigma_impl**2 * dt - r**2)
    cost += abs(pos) * path[-1] * hedge_cost_bps / 1e4  # unwind
    payoff = abs(path[-1] - K)
    pnl = premium - payoff + hedge_pnl - cost
    realized = float(np.std(np.diff(np.log(path))) * np.sqrt(ANN))
    return {
        "premium_pct": premium / S0 * 100,
        "payoff_pct": payoff / S0 * 100,
        "hedge_pnl_pct": hedge_pnl / S0 * 100,
        "cost_pct": cost / S0 * 100,
        "pnl_pct": pnl / S0 * 100,
        "pnl_vol_pts": pnl / vega,
        "gamma_theta_pct": gamma_theta / S0 * 100,
        "implied": sigma_impl * 100,
        "realized": realized * 100,
        "var_swap_pts": (sigma_impl**2 - realized**2) / (2 * sigma_impl) * 100,  # variance swap P&L in vega-equivalent vol pts
    }


def backtest_short_straddle(close: pd.Series, implied: pd.Series, start: str | None = None,
                            hedge_cost_bps: float = 1.0, option_spread_vol_pts: float = 0.5,
                            horizon: int = HORIZON) -> pd.DataFrame:
    """One trade every `horizon` trading days: sell the ATM straddle at the implied index level,
    hedge daily to expiry. Returns one row per trade, indexed by trade date."""
    df = pd.DataFrame({"S": close, "iv": implied}).dropna()
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    rows = []
    for i in range(0, len(df) - horizon, horizon):
        path = df["S"].values[i:i + horizon + 1]
        res = hedge_one_trade(path, df["iv"].iloc[i] / 100, hedge_cost_bps, option_spread_vol_pts)
        res["date"] = df.index[i]
        res["expiry"] = df.index[i + horizon]
        rows.append(res)
    out = pd.DataFrame(rows).set_index("date")
    out["cum_pnl_pct"] = out["pnl_pct"].cumsum()
    out["drawdown_pct"] = out["cum_pnl_pct"] - out["cum_pnl_pct"].cummax()
    return out


def straddle_stats(trades: pd.DataFrame) -> dict:
    p = trades["pnl_pct"]
    return {
        "trades": int(len(p)),
        "avg_pnl_pct": float(p.mean()),
        "median_pnl_pct": float(p.median()),
        "hit_ratio": float((p > 0).mean() * 100),
        "sharpe": float(p.mean() / p.std() * np.sqrt(12)) if p.std() > 0 else np.nan,
        "worst_pct": float(p.min()),
        "worst_date": p.idxmin(),
        "best_pct": float(p.max()),
        "max_drawdown_pct": float(trades["drawdown_pct"].min()),
        "avg_premium_pct": float(trades["premium_pct"].mean()),
        "avg_cost_pct": float(trades["cost_pct"].mean()),
        "avg_pnl_vol_pts": float(trades["pnl_vol_pts"].mean()),
        "gamma_theta_corr": float(trades[["pnl_pct", "gamma_theta_pct"]].corr().iloc[0, 1]),
    }


def simulate_gbm_path(sigma: float, n: int = HORIZON, S0: float = 100.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(-0.5 * sigma**2 / ANN, sigma / np.sqrt(ANN), n)
    return S0 * np.exp(np.concatenate([[0], np.cumsum(r)]))
