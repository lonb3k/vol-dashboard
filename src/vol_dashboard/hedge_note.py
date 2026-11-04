"""Weekly corporate-hedging note: what a hedge costs, market by market, in the words a sales desk uses.

For every commodity in the universe with a futures curve (energy, base and precious metals, grains):

- Curve. Listed contracts from Yahoo (expired ones dropped; gold and silver on their active delivery months). The 6- and 12-month points are the contracts
  closest to 6 and 12 months after the front; the 12-month comparison is same-season for gas and grains
  (December against December). Backwardation means the deferred contract is below the front: a producer
  selling forward locks in less than today's front, a consumer buying forward pays less.
- Volatility. 21-day close-to-close realized vol of the front-month series, and its percentile over the
  last one and five years. Implied vol has a free history only for WTI (OVX) and gold (GVZ); elsewhere the
  realized percentile stands in for how expensive options are, on the assumption that implied tracks realized.
- Moves. One-week and one-month change of the front-month series (which rolls, so it includes roll gaps).

Every sentence is a rule applied to those numbers; nothing is a recommendation.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import charts, metrics as m
from .data import DATA_DIR, futures_curve, load_ohlc
from .sheet import _fig, _num, tile
from .universe import UNIVERSE

GROUPS = {"Energy": ["Brent Crude", "WTI Crude", "Natural Gas"],
          "Metals": ["Copper", "Aluminium", "Gold", "Silver"],
          "Agriculture": ["Wheat", "Corn", "Soybeans"]}
UNITS = {"Brent Crude": "$/bbl", "WTI Crude": "$/bbl", "Natural Gas": "$/MMBtu", "Copper": "$/lb", "Aluminium": "$/t",
         "Gold": "$/oz", "Silver": "$/oz", "Wheat": "c/bu", "Corn": "c/bu", "Soybeans": "c/bu"}
CURVE_HISTORY = DATA_DIR / "curve_history.csv"
# Precious metals list every month on Yahoo, but only these delivery months trade: the others carry
# stale settlements that make the curve zigzag. Every other market keeps all its listed months.
ACTIVE_MONTHS = {"GC": {2, 4, 6, 8, 10, 12}, "SI": {3, 5, 7, 9, 12}}
FLAT = 2.0          # a 12-month curve within ±2% of the front reads as flat
HIGH, LOW = 80, 20  # realized-vol percentile bands


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return (b.year - a.year) * 12 + (b.month - a.month)


def curve_point(curve: pd.DataFrame, months: int) -> pd.Series | None:
    """The contract closest to `months` after the front, if one sits within two months of it."""
    if len(curve) < 2:
        return None
    front = curve.iloc[0]
    gap = curve["expiry_month"].apply(lambda e: abs(months_between(front["expiry_month"], e) - months))
    i = gap.idxmin()
    return curve.loc[i] if gap.loc[i] <= 2 and i != curve.index[0] else None


def curve_metrics(curve: pd.DataFrame) -> dict:
    front = curve.iloc[0]
    out = {"front": front["contract"].split(".")[0], "front_price": float(front["price"]), "front_month": front["expiry_month"]}
    for k in (6, 12):
        pt = curve_point(curve, k)
        out[f"m{k}_contract"] = pt["contract"].split(".")[0] if pt is not None else None
        out[f"m{k}_pct"] = (pt["price"] / front["price"] - 1) * 100 if pt is not None else np.nan
    # the shape is read 12 months out, or on the furthest listed contract when the feed stops earlier
    far = curve_point(curve, 12)
    out["far_fallback"] = far is None
    if far is None:
        far = curve.iloc[-1]
    out["far_contract"] = far["contract"].split(".")[0]
    out["far_months"] = int(months_between(front["expiry_month"], far["expiry_month"]))
    out["far_pct"] = (far["price"] / front["price"] - 1) * 100
    s = out["far_pct"]
    out["shape"] = "flat" if abs(s) < FLAT else ("backwardation" if s < 0 else "contango")
    return out


def ordinal(x: float) -> str:
    n = int(round(x))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def percentile(series: pd.Series, value: float) -> float:
    s = series.dropna()
    return float((s <= value).mean() * 100) if len(s) else np.nan


def vol_metrics(px: pd.DataFrame, implied: pd.Series | None) -> dict:
    rv = m.rv_close_to_close(px, 21).dropna()
    now = float(rv.iloc[-1])
    out = {"rv21": now, "pct_1y": percentile(rv.iloc[-252:], now), "pct_5y": percentile(rv.iloc[-1260:], now),
           "iv": np.nan, "iv_rv": np.nan}
    if implied is not None and not implied.dropna().empty:
        out["iv"] = float(implied.dropna().iloc[-1])
        out["iv_rv"] = out["iv"] - now
    c = px["Close"].dropna()
    out["chg_1w"] = (c.iloc[-1] / c.iloc[-6] - 1) * 100 if len(c) > 6 else np.nan
    out["chg_1m"] = (c.iloc[-1] / c.iloc[-22] - 1) * 100 if len(c) > 22 else np.nan
    out["last_close"] = c.index[-1]
    return out


def vol_band(pct: float) -> str:
    return "high" if pct >= HIGH else "low" if pct <= LOW else "normal"


def curve_sentence(name: str, c: dict) -> str:
    s, far, k = c["far_pct"], c["far_contract"], c["far_months"]
    horizon = f"{k} months" + (" (the furthest contract on the free feed)" if c.get("far_fallback") else "")
    if c["shape"] == "flat":
        return f"{name}: flat curve, {far} within {abs(s):.1f}% of {c['front']}: forward hedges lock in roughly today's price."
    if c["shape"] == "backwardation":
        return (f"{name}: backwardation, {far} {abs(s):.1f}% below {c['front']}. A producer selling {horizon} forward locks in "
                f"{abs(s):.1f}% less than the front; a consumer buying forward pays {abs(s):.1f}% less.")
    return (f"{name}: contango, {far} {s:.1f}% above {c['front']}. A consumer buying {horizon} forward pays {s:.1f}% more than "
            f"the front; a producer selling forward locks in {s:.1f}% more.")


def vol_sentence(name: str, v: dict) -> str:
    band = vol_band(v["pct_5y"])
    base = f"{name}: realized vol {v['rv21']:.0f}%, {ordinal(v['pct_5y'])} percentile of five years"
    if band == "high":
        out = base + ". Options-based hedges (puts, collars) are expensive against history; forwards and swaps carry no vol premium."
    elif band == "low":
        out = base + ". Optionality is cheap against history: the moment to price puts, collars or participating forwards."
    else:
        out = base + ", in line with history."
    if not np.isnan(v["iv"]):
        rich = "well above" if v["iv_rv"] > 5 else "close to" if abs(v["iv_rv"]) <= 5 else "below"
        out += (f" Implied vol, which is what an option actually costs, is {v['iv']:.0f}%: {rich} what has been realized "
                f"({v['iv_rv']:+.1f} points).")
    return out


def snapshot_curves(curves: dict, as_of: pd.Timestamp) -> None:
    """Append today's curves to data/curve_history.csv, one row per contract, so later notes can show the shift."""
    rows = [{"date": as_of.date(), "market": n, "contract": r["contract"], "expiry_month": r["expiry_month"].date(),
             "price": r["price"]} for n, cv in curves.items() for _, r in cv.iterrows()]
    new = pd.DataFrame(rows)
    if CURVE_HISTORY.exists():
        old = pd.read_csv(CURVE_HISTORY)
        new = pd.concat([old[old["date"] != str(as_of.date())], new], ignore_index=True)
    new.to_csv(CURVE_HISTORY, index=False)


def _curves_chart(group: str, names: list[str], curves: dict) -> go.Figure:
    fig = charts._base(f"{group}: futures curve, % against the front contract", height=360)
    fig.update_layout(yaxis=dict(ticksuffix="%", zeroline=True, zerolinecolor=charts.MUTED),
                      xaxis=dict(title="months after the front contract"))
    for i, n in enumerate(names):
        cv = curves.get(n)
        if cv is None or len(cv) < 2:
            continue
        f = cv.iloc[0]
        x = [months_between(f["expiry_month"], e) for e in cv["expiry_month"]]
        y = (cv["price"] / f["price"] - 1) * 100
        fig.add_trace(go.Scatter(x=x, y=y, name=n, mode="lines+markers", line=dict(width=2, color=charts.SERIES[i]),
                                 marker=dict(size=8), hovertemplate="%{y:+.1f}%"))
    return fig


def _vol_chart(table: pd.DataFrame) -> go.Figure:
    t = table.sort_values("pct_5y")
    fig = charts._base("Realized vol today, percentile of its own five-year history", height=380)
    fig.update_layout(yaxis=dict(ticksuffix="", range=[0, 100]), hovermode="closest", showlegend=False)
    fig.add_trace(go.Bar(x=t.index, y=t["pct_5y"], marker_color=charts.C["blue"], hovertemplate="%{x}: %{y:.0f}th<extra></extra>"))
    for lvl in (LOW, HIGH):
        fig.add_hline(y=lvl, line=dict(color=charts.MUTED, width=1, dash="dot"))
    return fig


def build_hedge_note(offline: bool = False, start: str = "2015-01-01") -> dict:
    markets = [a for a in UNIVERSE if a.asset_class == "Commodity" and a.futures_root]
    curves, rows, warnings_ = {}, {}, []
    for a in markets:
        cv = futures_curve(a.futures_root, a.exchange, offline=offline)
        px = load_ohlc(a.underlying, start=start, offline=offline)
        if cv is None or len(cv) < 2 or px.empty:
            warnings_.append(f"{a.name}: curve or prices unavailable")
            continue
        cv = cv.sort_values("expiry_month").reset_index(drop=True)
        if a.futures_root in ACTIVE_MONTHS:
            cv = cv[cv["expiry_month"].dt.month.isin(ACTIVE_MONTHS[a.futures_root])].reset_index(drop=True)
        iv = load_ohlc(a.vol_index, start=start, offline=offline)["Close"] if a.vol_index else None
        curves[a.name] = cv
        rows[a.name] = {**curve_metrics(cv), **vol_metrics(px, iv)}
    table = pd.DataFrame(rows).T
    as_of = max(r["last_close"] for r in rows.values())
    if not offline:
        snapshot_curves(curves, as_of)

    order = [n for g in GROUPS.values() for n in g if n in table.index]
    table = table.loc[order]
    back = table[table["shape"] == "backwardation"].index.tolist()
    cont = table[table["shape"] == "contango"].index.tolist()
    hi = table[table["pct_5y"].astype(float) >= HIGH].index.tolist()
    lo = table[table["pct_5y"].astype(float) <= LOW].index.tolist()
    headline = (f"Week of {as_of:%d %B %Y}. Backwardation in {', '.join(back) or 'none'}; contango in {', '.join(cont) or 'none'}. "
                f"Realized vol is high against five years in {', '.join(hi) or 'no market'} and low in {', '.join(lo) or 'no market'}: "
                f"that is where option hedges are dear and cheap.")

    tile_names = ["Brent Crude", "Natural Gas", "Copper", "Aluminium", "Wheat", "Corn"]
    tiles = []
    for n in tile_names:
        if n not in table.index:
            continue
        r = table.loc[n]
        px = load_ohlc(BY(n).underlying, start=start, offline=offline)["Close"]
        tiles.append(tile(f"{n}, {r['far_months']}M vs front", r["far_pct"], "%",
                          f"{r['shape']}; vol {r['rv21']:.0f}%, {ordinal(r['pct_5y'])} pct 5y", px.iloc[-120:],
                          "neutral", 1))

    # talking points: the most extreme curves and vols
    def score(r):
        c = abs(float(r["far_pct"])) / 15
        v = abs(float(r["pct_5y"]) - 50) / 50
        return max(c, v)
    ranked = sorted(table.index, key=lambda n: -score(table.loc[n]))[:4]
    ideas = []
    for n in ranked:
        r = table.loc[n]
        band = vol_band(float(r["pct_5y"]))
        if r["shape"] == "backwardation":
            angle = "producers face a discount to lock in forward; consumers are offered one"
        elif r["shape"] == "contango":
            angle = "consumers pay up to lock in forward; producers are paid to hedge"
        else:
            angle = "forward hedges sit at today's price"
        opt = {"high": "options dear against history", "low": "options cheap against history", "normal": "options in line with history"}[band]
        ideas.append({
            "rank": len(ideas) + 1, "title": f"{n}: {r['shape']}, {opt}", "direction": r["shape"],
            "instrument": f"{r['front']} to {r['far_contract']} ({UNITS.get(n, '')})",
            "condition": f"{r['far_months']}M {float(r['far_pct']):+.1f}% vs front, realized vol {float(r['rv21']):.0f}% at the {ordinal(r['pct_5y'])} percentile of 5y",
            "thesis": curve_sentence(n, r) + " " + vol_sentence(n, r).split(". ", 1)[-1] if band != "normal" else curve_sentence(n, r),
            "stats": {"expected": _num(r["far_pct"], 1), "expected_unit": f"% vs the front, {r['far_months']} months out",
                      "hit": _num(r["pct_5y"], 0), "worst": _num(r["chg_1m"], 1), "worst_unit": "% over one month",
                      "n": int(len(curves[n])), "sample": "contracts on the curve",
                      "labels": {"expected": "Curve", "hit": "Vol percentile", "hit_unit": "of five years",
                                 "worst": "1-month move", "sample": "Curve points"}},
            "score": _num(score(r), 2)})

    sections = []
    for g, names in GROUPS.items():
        names = [n for n in names if n in table.index]
        sections.append({"title": f"{g}: what a hedge locks in, and what optionality costs",
                         "paragraphs": [curve_sentence(n, table.loc[n]) for n in names] +
                                       [" ".join(vol_sentence(n, table.loc[n]) for n in names)]})
    sections.append({"title": "Curves", "paragraphs": ["Contracts closest to 6 and 12 months after the front; gas and grains compare "
                                                       "the same season. Where the free feed stops before 12 months, the shape is read on the furthest contract."],
                     "table": {"columns": ["Market", "Unit", "Front", "Price", "6M vs front", "12M vs front", "Shape read on", "Shape"],
                               "rows": [[n, UNITS.get(n, ""), r["front"], _num(r["front_price"], 2), _num(r["m6_pct"], 1),
                                         _num(r["m12_pct"], 1), f"{r['far_contract']} ({r['far_months']}M)", r["shape"]]
                                        for n, r in table.iterrows()]}})
    sections.append({"title": "Volatility and moves", "paragraphs": ["Realized vol of the front-month series over 21 sessions, "
                                                                      "with its percentile over one and five years. Implied only where CBOE publishes it."],
                     "table": {"columns": ["Market", "RV 21d", "Pct 1y", "Pct 5y", "Implied", "IV - RV", "1 week", "1 month"],
                               "rows": [[n, _num(r["rv21"], 1), _num(r["pct_1y"], 0), _num(r["pct_5y"], 0), _num(r["iv"], 1),
                                         _num(r["iv_rv"], 1), _num(r["chg_1w"], 1), _num(r["chg_1m"], 1)] for n, r in table.iterrows()]}})

    figs = [_fig(_curves_chart(g, [n for n in names if n in curves], curves), f"curve_{g.lower()}") for g, names in GROUPS.items()]
    figs.append(_fig(_vol_chart(table.astype({"pct_5y": float})), "vol_percentiles"))
    return {
        "project": "vol_dashboard", "slug": "hedging", "title": "Commodity hedging",
        "subtitle": "Weekly note for corporate hedging conversations: energy, metals, grains",
        "as_of": f"{as_of:%Y-%m-%d}", "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "headline": headline, "tiles": tiles, "sections": sections, "ideas": ideas,
        "ideas_title": "Talking points, the most extreme curves and vols first",
        "charts": figs,
        "method": "Talking points rank markets by how far their 12-month curve or their realized-vol percentile sits from normal. "
                  "Context for client conversations, not a recommendation.",
        "caveats": [
            "Free data: Yahoo Finance closes of NYMEX, COMEX and CBOT futures. Copper and aluminium are COMEX contracts, not LME; "
            "Brent is the NYMEX last-day financial contract. No TTF, power, coal, softs or pulp on the free feed.",
            "Implied vol history exists only for WTI (OVX) and gold (GVZ). Elsewhere the realized-vol percentile stands in for what "
            "options cost, which holds only if implied tracks realized.",
            "The front-month series rolls from contract to contract, so weekly and monthly moves include roll gaps.",
            "Curves come from listed contracts only (12 to 13 months out, fewer for grains and aluminium), not from swap curves. "
            "Deferred metal contracts trade thinly, so intermediate points can be noisy; the 12-month comparison uses the same "
            "delivery month a year later, usually the most liquid.",
        ],
        "warnings": warnings_,
        "links": {"repo": "https://github.com/lonb3k/vol-dashboard", "streamlit": "https://lonb3k-vol-dashboard.streamlit.app"},
    }


def BY(name: str):
    return next(a for a in UNIVERSE if a.name == name)
