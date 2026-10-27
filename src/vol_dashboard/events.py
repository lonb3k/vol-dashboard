"""Forward volatility and the price of scheduled events.

Two tenors of implied vol pin down the variance the market assigns to the window between them:

    sigma_fwd^2 (T1 -> T2) = (T2 sigma_2^2 - T1 sigma_1^2) / (T2 - T1)

With the CBOE term structure (VIX9D, VIX, VIX3M, VIX6M, VIX1Y) this gives a daily history of
forward vols. When a scheduled event (FOMC) sits inside the first 9 days, the short tenor
carries the event's variance on top of the "normal" daily variance, which we take from the
forward vol just after it. The implied event move is the square root of that excess variance.

Live, the same logic on consecutive option expiries around the event gives the market's
implied move for a specific date, the number every desk quotes before a Fed decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import ANN
from .universe import VIX_TERM

# FOMC decision days (second day of the meeting), including the 2020 emergency moves.
FOMC_DATES = pd.to_datetime([
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-03", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
])


def forward_vol(closes: pd.DataFrame, short: str, long: str) -> pd.Series:
    """Forward vol between two CBOE tenors, in vol points. NaN where the curve is so inverted
    that the forward variance would be negative."""
    t1, t2 = VIX_TERM[short], VIX_TERM[long]
    v = (t2 * closes[long] ** 2 - t1 * closes[short] ** 2) / (t2 - t1)
    return np.sqrt(v.where(v > 0)).rename(f"fwd {t1}d-{t2}d")


def forward_curve_history(closes: pd.DataFrame) -> pd.DataFrame:
    pairs = [("^VIX9D", "^VIX"), ("^VIX", "^VIX3M"), ("^VIX3M", "^VIX6M")]
    out = {}
    for s, l in pairs:
        if s in closes and l in closes:
            out[f"fwd {VIX_TERM[s]}d-{VIX_TERM[l]}d"] = forward_vol(closes, s, l)
    return pd.DataFrame(out).dropna(how="all")


def next_event(date: pd.Timestamp, events: pd.DatetimeIndex = FOMC_DATES) -> pd.Timestamp | None:
    nxt = events[events > date]
    return nxt[0] if len(nxt) else None


def event_premium_history(closes: pd.DataFrame, spx_close: pd.Series,
                          events: pd.DatetimeIndex = FOMC_DATES) -> pd.DataFrame:
    """For each trading day: is a FOMC inside the next 9 calendar days, the excess variance in the
    9-day tenor over the 9d-30d forward, and the implied event move it corresponds to.

    implied_move = sqrt( 9/365 * (VIX9D^2 - fwd^2) ) in % of spot, floored at zero.
    """
    df = pd.DataFrame({"vix9d": closes["^VIX9D"], "vix": closes["^VIX"]}).dropna()
    fwd = forward_vol(closes, "^VIX9D", "^VIX").reindex(df.index)
    df["fwd_9_30"] = fwd
    excess_var = (9 / 365) * ((df["vix9d"] / 100) ** 2 - (fwd / 100) ** 2)
    df["implied_move_pct"] = np.sqrt(excess_var.clip(lower=0)) * 100
    df["excess_vol_pts"] = df["vix9d"] - fwd
    nxt = pd.Series([next_event(d, events) for d in df.index], index=df.index)
    df["next_event"] = nxt
    df["days_to_event"] = (nxt - df.index.to_series()).dt.days
    df["event_in_window"] = df["days_to_event"].between(1, 9)
    r = np.log(spx_close / spx_close.shift(1))
    df["spx_abs_ret_pct"] = r.abs().reindex(df.index) * 100
    return df


def event_table(hist: pd.DataFrame, spx_close: pd.Series, events: pd.DatetimeIndex = FOMC_DATES) -> pd.DataFrame:
    """One row per event: implied move read on the last close before the event vs the realized
    absolute close-to-close move on the decision day."""
    r = np.log(spx_close / spx_close.shift(1)) * 100
    rows = []
    for e in events:
        before = hist[hist.index < e]
        if before.empty or e not in r.index:
            continue
        last = before.iloc[-1]
        if last["days_to_event"] > 9:
            continue
        rows.append({"event": e, "read_on": before.index[-1], "vix9d": last["vix9d"], "fwd_9_30": last["fwd_9_30"],
                     "implied_move_pct": last["implied_move_pct"], "realized_abs_move_pct": abs(r[e]),
                     "realized_move_pct": r[e]})
    out = pd.DataFrame(rows).set_index("event")
    out["ratio_realized_implied"] = out["realized_abs_move_pct"] / out["implied_move_pct"].replace(0, np.nan)
    return out


def event_premium_summary(hist: pd.DataFrame) -> pd.DataFrame:
    g = hist.dropna(subset=["excess_vol_pts"]).groupby("event_in_window")
    out = pd.DataFrame({"days": g.size(), "avg_vix9d": g["vix9d"].mean(), "avg_fwd_9_30": g["fwd_9_30"].mean(),
                        "avg_excess_vol_pts": g["excess_vol_pts"].mean(), "avg_implied_move_pct": g["implied_move_pct"].mean()})
    out.index = out.index.map({True: "FOMC within 9 days", False: "No FOMC within 9 days"})
    return out


# ---------- live: implied move for a date from the option chain ----------

def implied_move_from_chain(ticker: str, event_date: pd.Timestamp) -> dict:
    """Implied move for `event_date` from the two listed expiries on and after it.

    V(E1) = (n1 - 1) b + (b + e), V(E2) = (n2 - 1) b + (b + e)  ->  b = (V2 - V1)/(n2 - n1), e = V1 - n1 b
    where V = sigma_atm^2 * n / 252 and n counts trading days from today to the expiry inclusive.
    """
    from .data import listed_expiries, option_chain_for_expiry
    from .options import clean_chain, skew_metrics
    today = pd.Timestamp.today().normalize()
    exps = [e for e in listed_expiries(ticker) if e >= event_date and e > today]
    if len(exps) < 2:
        return {"error": "need two listed expiries after the event"}
    e1, e2 = exps[0], exps[1]
    out = {"event": event_date, "expiry_1": e1, "expiry_2": e2}
    vols, ns = [], []
    for e in (e1, e2):
        raw, _, spot = option_chain_for_expiry(ticker, e)
        cleaned, info = clean_chain(raw, spot, e)
        if cleaned.empty:
            return {"error": f"no quotes for {e.date()}"}
        vols.append(skew_metrics(cleaned)["atm_iv"] / 100)
        ns.append(len(pd.bdate_range(today + pd.Timedelta(days=1), e)))
        out["spot"] = info["spot"]
        out["stale"] = info["stale"]
    (s1, s2), (n1, n2) = vols, ns
    v1, v2 = s1**2 * n1 / ANN, s2**2 * n2 / ANN
    # base daily variance, two readings: the forward between the two expiries (can be polluted by
    # other events on those days, e.g. mega-cap earnings) and the 30-day ATM vol (cleaner)
    b_fwd = (v2 - v1) / (n2 - n1) if n2 > n1 else np.nan
    from .data import option_chain
    raw30, exp30, spot30 = option_chain(ticker, 30)
    c30, _ = clean_chain(raw30, spot30, exp30)
    s30 = skew_metrics(c30)["atm_iv"] / 100 if not c30.empty else np.nan
    b_30 = s30**2 / ANN
    out.update({"atm_iv_1": s1 * 100, "atm_iv_2": s2 * 100, "atm_iv_30d": s30 * 100, "n1": n1, "n2": n2,
                "base_daily_move_fwd_pct": float(np.sqrt(max(b_fwd, 0)) * 100),
                "base_daily_move_30d_pct": float(np.sqrt(max(b_30, 0)) * 100),
                "event_move_base_fwd_pct": float(np.sqrt(max(v1 - n1 * b_fwd, 0)) * 100),
                "event_move_base_30d_pct": float(np.sqrt(max(v1 - n1 * b_30, 0)) * 100),
                "straddle_1_move_pct": float(0.8 * s1 * np.sqrt(n1 / ANN) * 100)})
    return out
