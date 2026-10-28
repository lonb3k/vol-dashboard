"""Morning sheet: the dashboard read for you, as a desk note.

`build_sheet()` runs the existing research functions (metrics, forecast, hedging, crude, events,
signal) on today's data and returns a plain dict: headline tiles with sparklines, sections of
sentences with the numbers filled in, two to four trade ideas ranked by historical conditional
expectation against historical worst case, and the existing Plotly figures as JSON for an annex.

The dict is serialized to `sheet/latest.json` by `scripts/build_sheet.py` and rendered by the
morning-sheet front end. Nothing here changes the financial logic: every number comes from a
function that already exists in this package.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.io as pio

from . import atm, charts, crude, events, forecast, hedging, metrics as m, signal as sg
from .data import DATA_DIR, load_closes, load_ohlc
from .universe import BY_NAME, UNIVERSE, VIX_EXTRAS, VIX_TERM

REPO_URL = "https://github.com/lonb3k/vol-dashboard"
SPARK_POINTS = 60
CHART_YEARS = 5
MAX_POINTS = 500


# ---------- small helpers (shared layout with the two sister projects) ----------

def _num(x: Any, digits: int = 2) -> float | None:
    """JSON-safe float: NaN / inf become None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, digits)


def _spark(s: pd.Series, points: int = SPARK_POINTS) -> list[float]:
    s = s.dropna()
    if s.empty:
        return []
    tail = s.iloc[-points * 2:]
    if len(tail) > points:
        idx = np.linspace(0, len(tail) - 1, points).round().astype(int)
        tail = tail.iloc[idx]
    return [round(float(v), 3) for v in tail.values]


def tile(label: str, value: float | None, unit: str = "", sub: str = "", spark: pd.Series | None = None,
         tone: str = "neutral", digits: int = 1) -> dict:
    return {"label": label, "value": _num(value, digits), "unit": unit, "sub": sub, "tone": tone,
            "spark": _spark(spark) if spark is not None else []}


def _compact(obj):
    """Shrink a Plotly JSON dict: midnight timestamps become dates, floats are rounded to 3 decimals."""
    if isinstance(obj, list):
        if obj and all(isinstance(v, str) and v.endswith("T00:00:00") for v in obj):
            return [v[:10] for v in obj]
        if obj and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
            return [round(v, 3) if isinstance(v, float) else v for v in obj]
        return [_compact(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items()}
    return obj


def _fig(fig, chart_id: str, title: str | None = None) -> dict:
    d = _compact(json.loads(pio.to_json(fig, validate=False)))
    d["layout"].pop("template", None)  # the front end applies its own theme
    return {"id": chart_id, "title": title or fig.layout.title.text, "figure": d}


def _recent(s: pd.Series | pd.DataFrame, years: int = CHART_YEARS, max_points: int = MAX_POINTS):
    """Last `years` of a daily series, thinned to at most `max_points` rows (the last row is kept)."""
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    s = s[s.index >= cutoff]
    if len(s) > max_points:
        step = math.ceil(len(s) / max_points)
        keep = list(range(len(s) - 1, -1, -step))[::-1]
        s = s.iloc[keep]
    return s


def _pct_word(p: float) -> str:
    if p >= 90:
        return "very high"
    if p >= 70:
        return "high"
    if p >= 30:
        return "mid-range"
    if p >= 10:
        return "low"
    return "very low"


def _score(expected: float | None, worst: float | None) -> float:
    """Rank ideas by expected P&L per unit of historical worst case. Simple and honest: a
    desk cares about how much it is paid for the tail it carries."""
    if expected is None or worst is None or worst >= 0:
        return expected or -1e9
    return expected / abs(worst)


# ---------- the sheet ----------

def build_sheet(offline: bool = False, start: str = "2015-01-01") -> dict:
    today = pd.Timestamp.today().normalize()
    warnings_: list[str] = []
    ideas: list[dict] = []
    sections: list[dict] = []
    chart_list: list[dict] = []

    def guard(name: str, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as exc:  # one failed source must not kill the note
            warnings_.append(f"{name}: {type(exc).__name__}: {str(exc)[:120]}")
            return None

    # ---------- S&P 500 core ----------
    spx = BY_NAME["S&P 500"]
    px = load_ohlc(spx.underlying, start=start, offline=offline)
    vix = load_ohlc(spx.vol_index, start=start, offline=offline)["Close"].rename("VIX")
    as_of = px.index[-1]
    rv21 = m.rv_close_to_close(px, 21).dropna()
    rv63 = m.rv_close_to_close(px, 63).dropna()
    iv_now, rv_now = float(vix.dropna().iloc[-1]), float(rv21.iloc[-1])
    iv_pct, rv_pct = m.percentile_rank(vix), m.percentile_rank(rv21)
    spread = iv_now - rv_now
    ret_1d = float(px["Close"].pct_change().iloc[-1] * 100)
    vrp = m.vol_risk_premium(vix, px, 21)
    vrp_avg, vrp_pos = float(vrp["vrp"].mean()), float((vrp["vrp"] > 0).mean() * 100)

    # ---------- term structure ----------
    ts = load_closes(list(VIX_TERM) + list(VIX_EXTRAS), start=start, offline=offline)
    snap = m.term_structure_snapshot(ts, VIX_TERM)
    level = snap.set_index("index")["vol"]
    ratio = m.term_structure_history(ts, "^VIX", "^VIX3M")
    ratio_now = float(ratio.iloc[-1])
    inverted_share = float((ratio > 1).mean() * 100)
    curve_state = "contango" if ratio_now < 1 else "backwardation"
    fwd = events.forward_curve_history(ts)
    fwd_30_91 = float(fwd["fwd 30d-91d"].dropna().iloc[-1]) if "fwd 30d-91d" in fwd else float("nan")
    vvix = ts["^VVIX"].dropna() if "^VVIX" in ts else pd.Series(dtype=float)
    skew_idx = ts["^SKEW"].dropna() if "^SKEW" in ts else pd.Series(dtype=float)

    # ---------- forecasting race and the gap ----------
    fc = forecast.run_forecasts(px, vix)
    ev = forecast.evaluate(fc)
    last_fc = fc.iloc[-1]
    har_iv = float(last_fc["HAR+IV"]) if "HAR+IV" in fc else float("nan")
    har = float(last_fc["HAR-RV"])
    gap = iv_now - har_iv if math.isfinite(har_iv) else float("nan")
    cond = sg.conditional_premium(fc)
    gap_bucket = str(pd.cut([gap], sg.GAP_EDGES, labels=sg.GAP_LABELS)[0]) if math.isfinite(gap) else None
    implied_bias = float(ev.loc["Implied", "Bias (pts)"]) if "Implied" in ev.index else float("nan")

    # ---------- straddle backtest + gap conditioning ----------
    px_long = load_ohlc(spx.underlying, start="2005-01-01", offline=offline)["Close"]
    vix_long = load_ohlc(spx.vol_index, start="2005-01-01", offline=offline)["Close"]
    skew_long = load_ohlc("^SKEW", start="2005-01-01", offline=offline)["Close"]
    atm_long, _ = atm.atm_vol(spx.vol_index, vix_long, atm.load_vxo(offline=offline), skew_long)
    trades = hedging.backtest_short_straddle(px_long, atm_long, varswap=vix_long)  # an ATM straddle sells at ATM vol, not at the VIX
    ss = hedging.straddle_stats(trades)
    twg = sg.trades_with_gap(trades, fc)
    by_gap = sg.pnl_by_gap(twg)

    # ---------- headline tiles ----------
    tiles = [
        tile("S&P 500", px["Close"].iloc[-1], "", f"{ret_1d:+.2f}% on the day", px["Close"], "neutral", 0),
        tile("VIX", iv_now, "%", f"{iv_pct:.0f}th pct 1y, {_pct_word(iv_pct)}", vix, "neg" if iv_pct > 70 else "neutral"),
        tile("Realized 21d", rv_now, "%", f"{rv_pct:.0f}th pct 1y, {_pct_word(rv_pct)}", rv21, "neutral"),
        tile("IV - RV", spread, " pts", "options rich" if spread > 0 else "options cheap", (vix - rv21).dropna(),
             "pos" if spread > 0 else "neg"),
        tile("VIX / VIX3M", ratio_now, "", curve_state, ratio, "neg" if ratio_now > 1 else "pos", 3),
        tile("VIX - HAR+IV gap", gap, " pts", f"bucket {gap_bucket}" if gap_bucket else "model unavailable",
             sg.gap_series(fc), "pos" if gap > 2 else "neutral"),
    ]
    if not vvix.empty:
        tiles.append(tile("VVIX", vvix.iloc[-1], "", f"{m.percentile_rank(vvix):.0f}th pct 1y", vvix, "neutral", 1))
    if not skew_idx.empty:
        tiles.append(tile("CBOE SKEW", skew_idx.iloc[-1], "", f"{m.percentile_rank(skew_idx):.0f}th pct 1y", skew_idx, "neutral", 1))

    # ---------- section 1: the read ----------
    rich_word = "rich" if spread > 0 else "cheap"
    sections.append({
        "title": "S&P 500 volatility",
        "paragraphs": [
            f"The VIX closed at {iv_now:.1f}, the {iv_pct:.0f}th percentile of its trailing year ({_pct_word(iv_pct)}). "
            f"The S&P 500 realized {rv_now:.1f}% over the last 21 sessions ({rv_pct:.0f}th percentile) and {float(rv63.iloc[-1]):.1f}% over 63. "
            f"Implied sits {abs(spread):.1f} points {'above' if spread > 0 else 'below'} realized: options are {rich_word} against recent history.",
            f"Since {vrp.index[0].year} the one-month variance premium (implied today minus realized over the next 21 days) averaged "
            f"{vrp_avg:+.1f} points and was positive on {vrp_pos:.0f}% of days. Today's {spread:+.1f} spread is "
            f"{'above' if spread > vrp_avg else 'below'} that average: the seller is paid {'more' if spread > vrp_avg else 'less'} than usual going in.",
            f"The HAR+IV model, which debiases the VIX with past realized vol, forecasts {har_iv:.1f}% for the next month "
            f"(pure HAR-RV: {har:.1f}%). The gap between market and model is {gap:+.1f} points, in the '{gap_bucket}' bucket. "
            f"Historically in that bucket the forward premium averaged {float(cond.loc[gap_bucket, 'avg_vrp_pts']):+.1f} points and was positive "
            f"{float(cond.loc[gap_bucket, 'vrp_positive_%']):.0f}% of the time."
            if gap_bucket else "The HAR+IV model is unavailable today.",
        ],
    })

    # ---------- section 2: term structure ----------
    ts_labels = {"^VIX9D": "VIX9D", "^VIX": "VIX", "^VIX3M": "VIX3M", "^VIX6M": "VIX6M", "^VIX1Y": "VIX1Y"}
    curve_txt = ", ".join(f"{ts_labels.get(i, i)} {v:.1f}" for i, v in zip(snap["index"], snap["vol"]))
    sections.append({
        "title": "Term structure",
        "paragraphs": [
            f"Curve: {curve_txt}. VIX / VIX3M = {ratio_now:.3f}, {curve_state}"
            f"{' (normal: the market expects vol to mean-revert up)' if curve_state == 'contango' else ' (stress: near-term vol above deferred)'}. "
            f"The curve has been inverted on {inverted_share:.0f}% of days since {ratio.index[0].year}.",
            f"The forward vol between 30 and 91 days is {fwd_30_91:.1f}%: that is what the market charges for month two and three, "
            f"{'above' if fwd_30_91 > iv_now else 'below'} spot VIX."
            + (f" VVIX at {float(vvix.iloc[-1]):.0f} and CBOE SKEW at {float(skew_idx.iloc[-1]):.0f} ({m.percentile_rank(skew_idx):.0f}th pct) "
               f"describe the vol-of-vol and tail pricing." if not vvix.empty and not skew_idx.empty else ""),
        ],
        "table": {"columns": ["Tenor", "Days", "Vol"], "rows": [[ts_labels.get(i, i), int(d), _num(v, 2)] for i, d, v in
                                                                 zip(snap["index"], snap["tenor_days"], snap["vol"])]},
    })

    # ---------- section 3: cross-asset ----------
    rows = []
    for a in UNIVERSE:
        p = load_ohlc(a.underlying, start=start, offline=offline)
        if p.empty:
            continue
        iv = load_ohlc(a.vol_index, start=start, offline=offline)["Close"] if a.vol_index else None
        rows.append(m.summary_row(a.name, a.asset_class, p, iv))
    table = pd.DataFrame(rows)
    with_iv = table.dropna(subset=["IV - RV"]).sort_values("IV - RV", ascending=False)
    richest, cheapest = with_iv.iloc[0], with_iv.iloc[-1]
    sections.append({
        "title": "Cross-asset",
        "paragraphs": [
            f"Among the assets with a CBOE implied index, {richest['Asset']} has the richest options (IV {richest['IV 30d']:.1f}% vs realized "
            f"{richest['RV 21d']:.1f}%, {richest['IV - RV']:+.1f} points) and {cheapest['Asset']} the cheapest ({cheapest['IV - RV']:+.1f} points). "
            f"Realized vol percentiles over one year: " + ", ".join(f"{r['Asset']} {r['RV 21d pct (1y)']:.0f}th" for _, r in table.iterrows()) + ".",
        ],
        "table": {"columns": ["Asset", "Class", "Spot", "1d %", "RV 21d", "RV 63d", "RV pct 1y", "IV 30d", "IV - RV", "IV pct 1y"],
                  "rows": [[r["Asset"], r["Class"], _num(r["Spot"], 2), _num(r["1d %"], 2), _num(r["RV 21d"], 1), _num(r["RV 63d"], 1),
                            _num(r["RV 21d pct (1y)"], 0), _num(r["IV 30d"], 1), _num(r["IV - RV"], 1), _num(r["IV pct (1y)"], 0)]
                           for _, r in table.iterrows()]},
    })

    # ---------- section 4: crude ----------
    crude_ctx = None

    def crude_section():
        cs = crude.load_contracts("CL", "NYM", offline=offline)
        if len(cs) < 5:
            raise RuntimeError("not enough contract histories")
        today_cs = crude.samuelson_same_date(cs)
        spread_dec, near, far = crude.dec_dec_spread(cs)
        ovx = load_ohlc("^OVX", start=start, offline=offline)["Close"]
        clpx = load_ohlc("CL=F", start=start, offline=offline)
        res = crude.curve_vs_vol(spread_dec, ovx, clpx)
        f, b = today_cs.iloc[0], today_cs.iloc[min(11, len(today_cs) - 1)]
        s_now = float(spread_dec.iloc[-1])
        regime = "Backwardation" if s_now > 0 else "Contango"
        by = res["by_regime"]
        panel = res["panel"]
        in_regime = panel[panel["regime"] == regime].dropna(subset=["vrp"])
        ovx_now = float(ovx.dropna().iloc[-1])
        cl_rv = float(m.rv_close_to_close(clpx, 21).dropna().iloc[-1])
        sections.append({
            "title": "WTI crude",
            "paragraphs": [
                f"OVX at {ovx_now:.1f} ({m.percentile_rank(ovx):.0f}th pct 1y) against {cl_rv:.1f}% realized on the front future. "
                f"The {near[:5]} / {far[:5]} calendar spread is {s_now:+.1f}%: the curve is in {regime.lower()}.",
                f"Samuelson effect today: {f['contract'][:5]} ({f['ttm_months']:.1f} months out) realizes {f['rv']:.0f}% against {b['rv']:.0f}% for "
                f"{b['contract'][:5]} at {b['ttm_months']:.0f} months. News hits the front of the curve; the back is anchored by marginal cost.",
                f"Selling crude vol at the OVX level earned {float(by.loc[regime, 'avg_vrp']):+.1f} points on average in {regime.lower()} "
                f"(positive {float(by.loc[regime, 'vrp_positive']):.0f}% of days, worst {float(in_regime['vrp'].min()):+.1f}) versus "
                f"{float(by.loc['Contango' if regime == 'Backwardation' else 'Backwardation', 'avg_vrp']):+.1f} in the other regime.",
            ],
        })
        chart_list.append(_fig(charts.samuelson_today(today_cs), "crude_samuelson"))
        chart_list.append(_fig(charts.spread_vs_vol(_recent(panel)), "crude_spread_vs_vol"))
        return {"regime": regime, "avg_vrp": float(by.loc[regime, "avg_vrp"]), "pos": float(by.loc[regime, "vrp_positive"]),
                "worst": float(in_regime["vrp"].min()), "worst_date": in_regime["vrp"].idxmin(), "n": int(len(in_regime)),
                "ovx": ovx_now, "spread": s_now, "cl_rv": cl_rv}

    crude_ctx = guard("crude", crude_section)

    # ---------- section 5: events ----------
    ts_long = load_closes(list(VIX_TERM), start="2011-01-01", offline=offline)
    spx_long = load_ohlc("^GSPC", start="2011-01-01", offline=offline)["Close"]
    ev_hist = events.event_premium_history(ts_long, spx_long)
    ev_tab = events.event_table(ev_hist, spx_long)
    nxt = events.next_event(today)
    days_to = (nxt - today).days if nxt is not None else None
    last_ev = ev_hist.iloc[-1]
    avg_impl, avg_real = float(ev_tab["implied_move_pct"].mean()), float(ev_tab["realized_abs_move_pct"].mean())
    ratio_ri = ev_tab["ratio_realized_implied"].dropna()
    event_ctx = {"next": nxt, "days_to": days_to, "implied_move_pct": float(last_ev["implied_move_pct"]),
                 "excess_vol_pts": float(last_ev["excess_vol_pts"]), "avg_impl": avg_impl, "avg_real": avg_real,
                 "ratio_mean": float(ratio_ri.mean()), "ratio_max": float(ratio_ri.max()),
                 "ratio_max_date": ratio_ri.idxmax(), "n": int(len(ev_tab)),
                 "share_realized_above": float((ratio_ri > 1).mean() * 100)}
    event_para = [
        f"Next FOMC decision: {nxt.date()} ({days_to} days). " if nxt is not None else "No FOMC scheduled in the calendar. ",
    ]
    if days_to is not None and 1 <= days_to <= 9:
        if float(last_ev["implied_move_pct"]) > 0:
            event_para[0] += (f"It sits inside the 9-day tenor: VIX9D carries {float(last_ev['excess_vol_pts']):+.1f} points over the 9d-30d forward, "
                              f"an implied move of {float(last_ev['implied_move_pct']):.2f}% for the day.")
        else:
            event_para[0] += (f"It sits inside the 9-day tenor, yet VIX9D is {abs(float(last_ev['excess_vol_pts'])):.1f} points below the 9d-30d forward: "
                              f"the contango is steep enough to hide the event premium, so the term-structure read is floored at zero. "
                              f"VIX9D alone implies a {float(ts['^VIX9D'].dropna().iloc[-1]) / 100 / math.sqrt(252) * 100:.2f}% daily move.")
    else:
        event_para[0] += "It is outside the 9-day window, so VIX9D carries no identifiable event premium today."
    event_para.append(
        f"Over {len(ev_tab)} meetings since 2015 the term structure priced an average {avg_impl:.2f}% move and the S&P realized {avg_real:.2f}% "
        f"(ratio {float(ratio_ri.mean()):.2f}, realized above implied on {float((ratio_ri > 1).mean() * 100):.0f}% of meetings, max ratio "
        f"{float(ratio_ri.max()):.1f}x on {ratio_ri.idxmax().date()}). Event vol is fairly priced; the generic month is not: the carry lives in the days with no scheduled news.")
    sections.append({"title": "Scheduled events", "paragraphs": event_para})

    # ---------- section 6: skew from snapshots ----------
    hist_path = DATA_DIR / "skew_history.csv"
    if hist_path.exists():
        sh = pd.read_csv(hist_path)
        spy = sh[(sh["proxy"] == "SPY") & (sh["tenor_target"] == 30)].sort_values("date")
        if not spy.empty:
            r = spy.iloc[-1]
            sections.append({
                "title": "Skew (SPY 30d, last snapshot)",
                "paragraphs": [
                    f"Snapshot {r['date']}: ATM {r['atm_iv']:.1f}%, 25-delta put {r['put25']:.1f}%, 25-delta call {r['call25']:.1f}%, "
                    f"risk reversal {r['rr25']:+.2f} points, butterfly {r['bf25']:+.2f}. {int(sh['date'].nunique())} snapshot day(s) in the history so far"
                    + (" (quotes stale: read after the close)." if bool(r["stale"]) else "."),
                ],
            })

    # ---------- ideas ----------
    # 1. short delta-hedged straddle, conditional on today's gap bucket
    if gap_bucket and gap_bucket in by_gap.index and int(by_gap.loc[gap_bucket, "trades"]) >= 5:
        row = by_gap.loc[gap_bucket]
        in_b = twg[twg["gap_bucket"].astype(str) == gap_bucket]
        worst_date = in_b["pnl_pct"].idxmin()
        ideas.append({
            "title": "Sell the 1-month ATM S&P straddle, delta-hedged daily",
            "direction": "short vol", "instrument": "SPX / SPY 1m ATM straddle at the ATM vol (about 0.8 x VIX), hedged at each close",
            "condition": f"VIX - HAR+IV gap in the '{gap_bucket}' bucket ({gap:+.1f} pts today)",
            "thesis": f"The market charges {gap:+.1f} points over the debiased model. In this bucket the straddle kept "
                      f"{float(row['avg_pnl_pct']):+.2f}% of spot per trade after hedge costs, {float(row['hit_%']):.0f}% winners.",
            "stats": {"expected": _num(row["avg_pnl_pct"], 2), "expected_unit": "% of spot per month", "hit": _num(row["hit_%"], 0),
                      "worst": _num(row["worst_pct"], 2), "worst_unit": "% of spot", "worst_date": worst_date.strftime("%b %Y"),
                      "n": int(row["trades"]), "sample": f"{int(row['trades'])} trades in this bucket since {in_b.index[0].year}",
                      "unconditional": f"all {ss['trades']} trades since 2005: {ss['avg_pnl_pct']:+.2f}% avg, {ss['hit_ratio']:.0f}% hit, worst {ss['worst_pct']:+.1f}% ({ss['worst_date'].strftime('%b %Y')})"},
        })
    # 2. carry conditional on the curve state (VIX / VIX3M bucket)
    ratio_b = pd.cut(ratio, [0, 0.8, 0.9, 1.0, 10], labels=["< 0.80", "0.80-0.90", "0.90-1.00", "> 1 (inverted)"])
    joined = pd.DataFrame({"vrp": vrp["vrp"], "bucket": ratio_b}).dropna()
    rb_now = str(pd.cut([ratio_now], [0, 0.8, 0.9, 1.0, 10], labels=["< 0.80", "0.80-0.90", "0.90-1.00", "> 1 (inverted)"])[0])
    in_rb = joined[joined["bucket"].astype(str) == rb_now]
    if len(in_rb) >= 20:
        exp_rb, worst_rb = float(in_rb["vrp"].mean()), float(in_rb["vrp"].min())
        direction = "short vol" if exp_rb > 0 else "long vol"
        ideas.append({
            "title": "Sell 1-month S&P variance at the VIX" if exp_rb > 0 else "Own 1-month S&P variance",
            "direction": direction, "instrument": "1m variance swap at the VIX, held to expiry (the premium sits in the wings, not at the money)",
            "condition": f"VIX / VIX3M = {ratio_now:.3f}, bucket '{rb_now}'",
            "thesis": f"With the curve in {curve_state}, the forward premium (VIX today minus realized over the next 21 days) averaged "
                      f"{exp_rb:+.1f} points in this bucket and was positive {float((in_rb['vrp'] > 0).mean() * 100):.0f}% of the time.",
            "stats": {"expected": _num(exp_rb, 2), "expected_unit": "vol pts over the month", "hit": _num((in_rb["vrp"] > 0).mean() * 100, 0),
                      "worst": _num(worst_rb, 1), "worst_unit": "vol pts", "worst_date": in_rb["vrp"].idxmin().strftime("%b %Y"),
                      "n": int(len(in_rb)), "sample": f"{len(in_rb)} daily observations since {joined.index[0].year}"},
        })
    # 3. crude vol by curve regime
    if crude_ctx:
        c = crude_ctx
        short = c["avg_vrp"] > 0
        ideas.append({
            "title": "Sell WTI vol at the OVX level" if short else "Stay away from short crude vol",
            "direction": "short vol" if short else "flat / long vol", "instrument": "USO / CL 1m variance at the OVX (crude ATM vol sits about 5% below OVX)",
            "condition": f"Dec/Dec spread {c['spread']:+.1f}%: curve in {c['regime'].lower()}",
            "thesis": f"OVX {c['ovx']:.1f} against {c['cl_rv']:.1f}% realized. In {c['regime'].lower()} the crude variance premium averaged "
                      f"{c['avg_vrp']:+.1f} points, positive {c['pos']:.0f}% of days. {'Contango is when the crude vol seller gets hurt.' if c['regime'] == 'Contango' else 'Scarcity regimes have paid the seller.'}",
            "stats": {"expected": _num(c["avg_vrp"], 2), "expected_unit": "vol pts over the month", "hit": _num(c["pos"], 0),
                      "worst": _num(c["worst"], 1), "worst_unit": "vol pts", "worst_date": c["worst_date"].strftime("%b %Y"),
                      "n": c["n"], "sample": f"{c['n']} daily observations in {c['regime'].lower()} since 2018"},
        })
    # 4. FOMC inside the window: do not sell the event
    if days_to is not None and 1 <= days_to <= 9:
        e = event_ctx
        ideas.append({
            "title": f"Do not sell the FOMC day ({nxt.date()}); sell the days after it",
            "direction": "event vol: flat", "instrument": "SPY weekly straddle expiring on or after the decision",
            "condition": f"FOMC in {days_to} day(s); term-structure implied move {e['implied_move_pct']:.2f}%" + (" (floored at zero by the contango)" if e['implied_move_pct'] <= 0 else ""),
            "thesis": f"Over {e['n']} meetings the market priced {e['avg_impl']:.2f}% and the index realized {e['avg_real']:.2f}%: the event is "
                      f"fairly priced on average and realized above implied {e['share_realized_above']:.0f}% of the time. The premium is in the generic days.",
            "stats": {"expected": _num(e["avg_impl"] - e["avg_real"], 2), "expected_unit": "% of spot (implied - realized move)",
                      "hit": _num(100 - e["share_realized_above"], 0), "worst": _num(-(e["ratio_max"] - 1) * e["avg_impl"], 2),
                      "worst_unit": "% of spot vs implied", "worst_date": e["ratio_max_date"].strftime("%b %Y"), "n": e["n"],
                      "sample": f"{e['n']} FOMC meetings since 2015"},
        })

    for i in ideas:
        i["score"] = _num(_score(i["stats"].get("expected"), i["stats"].get("worst")), 3)
    ideas.sort(key=lambda d: (d["score"] if d["score"] is not None else -1e9), reverse=True)
    for k, i in enumerate(ideas, 1):
        i["rank"] = k

    # ---------- charts annex ----------
    rv_table = m.realized_vol_table(px, 21)
    chart_list.insert(0, _fig(charts.realized_vs_implied(_recent(rv_table), _recent(vix), "VIX"), "rv_vs_iv"))
    chart_list.insert(1, _fig(charts.vrp_spread(_recent(vrp)), "vrp"))
    chart_list.insert(2, _fig(charts.term_structure(snap, ts_labels), "term_structure"))
    chart_list.insert(3, _fig(charts.ratio_history(_recent(ratio), "VIX / VIX3M: above 1 the curve is inverted"), "ratio"))
    models = [c for c in fc.columns if c != "target"]
    chart_list.append(_fig(charts.forecast_lines(_recent(fc), models), "forecasts"))
    chart_list.append(_fig(charts.straddle_cum(trades), "straddle_cum"))
    chart_list.append(_fig(charts.event_implied_vs_realized(ev_tab), "fomc"))
    cone, current = m.vol_cone(px)
    chart_list.append(_fig(charts.vol_cone(cone, current, iv_now), "cone"))

    # ---------- headline ----------
    lede = (f"{as_of.strftime('%A %d %B %Y')}. VIX {iv_now:.1f} ({_pct_word(iv_pct)}), realized {rv_now:.1f}%, spread {spread:+.1f} points, "
            f"curve in {curve_state}. The market charges {gap:+.1f} points over the HAR+IV model"
            + (f"; FOMC in {days_to} day(s)." if days_to is not None and days_to <= 9 else "."))
    if ideas:
        lede += f" Top idea: {ideas[0]['title'][0].lower() + ideas[0]['title'][1:]}."

    return {
        "project": "vol_dashboard", "slug": "vol", "title": "Volatility", "subtitle": "Equity indices and commodities: realized vs implied",
        "as_of": as_of.strftime("%Y-%m-%d"), "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "headline": lede, "tiles": tiles, "sections": sections, "ideas": ideas, "charts": chart_list,
        "method": "Ideas are ranked by historical conditional expectation divided by the historical worst case in the same regime "
                  "(expected P&L per unit of tail). Conditioning variables: the VIX minus HAR+IV gap bucket, the VIX / VIX3M curve bucket, "
                  "the WTI Dec/Dec curve regime, the FOMC calendar. Gross of financing, net of hedge costs where stated.",
        "caveats": [
            "Free data: Yahoo Finance closes and CBOE indices. Option quotes are delayed; implied vols are rebuilt from mids with Black-76.",
            "Conditional statistics are in-sample averages over overlapping daily windows; treat small buckets as indicative.",
            "The straddle backtest sells at the S&P ATM vol (VXO / sqrt(30/22) to 2021, then VIX x a ratio modelled on VIX and SKEW), "
            "with 1 bp hedge cost and 0.5 vol point bid-ask; no margin, no financing. Sold at the VIX it would look far better: "
            "VIX prices the skew, an ATM straddle does not get paid for it.",
        ],
        "warnings": warnings_,
        "links": {"repo": REPO_URL, "streamlit": "streamlit run app.py (port 8501)"},
    }
