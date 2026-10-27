"""Turning the forecasting race into a signal: is vol rich when implied sits above the model?

gap = implied vol - HAR+IV forecast (vol points). HAR+IV is the model that debiases the implied,
so the gap is "how much more than its usual premium the market is charging today". The test:
does a bigger gap mean a bigger premium over the next month, and does selling vol only on a big
gap beat selling vol every month?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP_EDGES = [-np.inf, 0, 2, 4, 6, np.inf]
GAP_LABELS = ["< 0", "0 to 2", "2 to 4", "4 to 6", "> 6"]


def gap_series(fc: pd.DataFrame) -> pd.Series:
    if "Implied" not in fc or "HAR+IV" not in fc:
        return pd.Series(dtype=float)
    return (fc["Implied"] - fc["HAR+IV"]).rename("gap")


def conditional_premium(fc: pd.DataFrame) -> pd.DataFrame:
    """Forward VRP (implied - realized next 21d) by gap bucket, daily observations."""
    d = fc.dropna(subset=["target", "Implied", "HAR+IV"]).copy()
    d["gap"] = d["Implied"] - d["HAR+IV"]
    d["vrp"] = d["Implied"] - d["target"]
    b = pd.cut(d["gap"], GAP_EDGES, labels=GAP_LABELS)
    g = d.groupby(b, observed=True)
    return pd.DataFrame({"days": g.size(), "share_%": g.size() / len(d) * 100, "avg_implied": g["Implied"].mean(),
                         "avg_model": g["HAR+IV"].mean(), "avg_fwd_realized": g["target"].mean(),
                         "avg_vrp_pts": g["vrp"].mean(), "vrp_positive_%": g["vrp"].apply(lambda s: (s > 0).mean() * 100)})


def trades_with_gap(trades: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Attach the gap observed on each straddle trade date."""
    gap = gap_series(fc)
    out = trades.copy()
    out["gap"] = gap.reindex(out.index, method="ffill")
    out["gap_bucket"] = pd.cut(out["gap"], GAP_EDGES, labels=GAP_LABELS)
    return out


def pnl_by_gap(trades: pd.DataFrame) -> pd.DataFrame:
    d = trades.dropna(subset=["gap"])
    g = d.groupby("gap_bucket", observed=True)
    return pd.DataFrame({"trades": g.size(), "avg_pnl_pct": g["pnl_pct"].mean(), "hit_%": g["pnl_pct"].apply(lambda s: (s > 0).mean() * 100),
                         "worst_pct": g["pnl_pct"].min(), "avg_vol_pts": g["pnl_vol_pts"].mean()})


def strategy_comparison(trades: pd.DataFrame, thresholds=(None, 0.0, 2.0, 4.0)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sell every month vs sell only when gap > threshold. Returns (stats table, cumulative P&L paths)."""
    d = trades.dropna(subset=["gap"])
    stats, paths = [], {}
    for th in thresholds:
        name = "Always" if th is None else f"gap > {th:g}"
        active = pd.Series(True, index=d.index) if th is None else d["gap"] > th
        p = d["pnl_pct"].where(active, 0.0)
        taken = p[active]
        cum = p.cumsum()
        dd = cum - cum.cummax()
        stats.append({"Strategy": name, "trades": int(active.sum()), "avg_pnl_per_trade_pct": float(taken.mean()) if len(taken) else np.nan,
                      "total_pnl_pct": float(p.sum()), "hit_%": float((taken > 0).mean() * 100) if len(taken) else np.nan,
                      "worst_pct": float(taken.min()) if len(taken) else np.nan, "max_dd_pct": float(dd.min()),
                      "sharpe": float(p.mean() / p.std() * np.sqrt(12)) if p.std() > 0 else np.nan})
        paths[name] = cum
    return pd.DataFrame(stats).set_index("Strategy"), pd.DataFrame(paths)
