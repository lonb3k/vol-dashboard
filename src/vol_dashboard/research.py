"""Research on implied vol where a history exists (CBOE indices, or Bloomberg extracts in local mode).

- variance premium: implied 1-month vol against the vol realized over the next 21 sessions;
- implied term structure: how often 1-month implied sits above 3-month, and what 1-month implied does next;
- proxy audit: how a free-data contract tracks the benchmark a European client hedges (weekly returns, vol);
- calibration: the typical ratio of implied to realized vol, to estimate implied vol where no history is free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def realized(close: pd.Series, window: int = 21) -> pd.Series:
    r = np.log(close / close.shift(1))
    return r.rolling(window).std() * np.sqrt(ANN) * 100


def forward_realized(close: pd.Series, horizon: int = 21) -> pd.Series:
    return realized(close, horizon).shift(-horizon)


def variance_premium(iv: pd.Series, close: pd.Series, horizon: int = 21) -> dict:
    df = pd.concat({"iv": iv, "fwd": forward_realized(close, horizon)}, axis=1, sort=True).dropna()
    prem = df["iv"] - df["fwd"]
    terc = pd.qcut(df["iv"].rank(method="first"), 3, labels=["low implied", "mid implied", "high implied"])
    by = prem.groupby(terc, observed=True).mean()
    return {"mean": float(prem.mean()), "median": float(prem.median()), "hit": float((prem > 0).mean() * 100),
            "worst": float(prem.min()), "n": int(len(prem)), "start": df.index[0], "by_tercile": by.round(2).to_dict()}


def term_structure(iv1: pd.Series, iv3: pd.Series, horizon: int = 21) -> dict:
    df = pd.concat({"iv1": iv1, "iv3": iv3}, axis=1, sort=True).dropna()
    df["inverted"] = df["iv1"] > df["iv3"]
    df["next"] = df["iv1"].shift(-horizon) - df["iv1"]
    d = df.dropna()
    inv, norm = d[d["inverted"]], d[~d["inverted"]]
    return {"share_inverted": float(df["inverted"].mean() * 100), "now_inverted": bool(df["inverted"].iloc[-1]),
            "spread_now": float(df["iv1"].iloc[-1] - df["iv3"].iloc[-1]),
            "next_change_inverted": float(inv["next"].mean()) if len(inv) else np.nan,
            "next_change_normal": float(norm["next"].mean()) if len(norm) else np.nan,
            "fell_when_inverted": float((inv["next"] < 0).mean() * 100) if len(inv) else np.nan, "n": int(len(d))}


def proxy_audit(benchmark: pd.Series, proxy: pd.Series, since: str = "2021-11-01") -> dict:
    """Weekly (Friday) returns avoid the time-zone gap between London and New York settlements."""
    b, p = benchmark[benchmark.index >= since], proxy[proxy.index >= since]
    wk = pd.concat({"b": b.resample("W-FRI").last(), "p": p.resample("W-FRI").last()}, axis=1).dropna()
    r = np.log(wk / wk.shift(1)).dropna()
    rb, rp = realized(benchmark).dropna(), realized(proxy).dropna()
    both = pd.concat({"b": rb, "p": rp}, axis=1, sort=True).dropna()
    both = both[both.index >= since]
    def pct(s):
        w = s.iloc[-1260:]
        return float((w <= w.iloc[-1]).mean() * 100)
    return {"weekly_corr": float(r.corr().iloc[0, 1]), "vol_ratio": float((both["p"] / both["b"]).median()),
            "vol_gap_now": float(both["p"].iloc[-1] - both["b"].iloc[-1]),
            "pct_benchmark": pct(rb), "pct_proxy": pct(rp)}


def calibrate(iv: pd.Series, close: pd.Series) -> dict:
    """Implied vol as a multiple of 21-day realized: median ratio, and the error of estimating implied that way."""
    df = pd.concat({"iv": iv, "rv": realized(close)}, axis=1, sort=True).dropna()
    ratio = df["iv"] / df["rv"]
    k = float(ratio.median())
    err = (k * df["rv"] - df["iv"]).abs()
    return {"ratio": round(k, 3), "mae_pts": round(float(err.median()), 2), "p90_err_pts": round(float(err.quantile(0.9)), 2),
            "n": int(len(df)), "start": str(df.index[0].date()), "end": str(df.index[-1].date())}
