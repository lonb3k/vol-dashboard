"""Plotly figures. One y-axis per chart, thin marks, categorical colors in fixed order."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Validated categorical palette (light surface), fixed slot order
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"], C["violet"], C["red"]]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, Helvetica, Arial, sans-serif", size=12, color=TEXT),
    margin=dict(l=48, r=24, t=84, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=MUTED)),
    xaxis=dict(showgrid=False, zeroline=False, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix="%"),
)


def _base(title: str, height: int = 380, **overrides) -> go.Figure:
    fig = go.Figure()
    layout = {**LAYOUT, **overrides}
    fig.update_layout(title=dict(text=title, x=0, y=0.98, yanchor="top", font=dict(size=14)), height=height, **layout)
    return fig


# ---------- time series ----------

def realized_vs_implied(rv: pd.DataFrame, implied: pd.Series | None, implied_name: str | None) -> go.Figure:
    fig = _base("Realized vol (annualized) vs 30d implied")
    for i, col in enumerate(rv.columns):
        fig.add_trace(go.Scatter(x=rv.index, y=rv[col], name=col, mode="lines",
                                 line=dict(width=1.5, color=SERIES[i]),
                                 hovertemplate="%{y:.1f}%"))
    if implied is not None and not implied.empty:
        fig.add_trace(go.Scatter(x=implied.index, y=implied, name=implied_name or "Implied", mode="lines",
                                 line=dict(width=2, color=C["red"]), hovertemplate="%{y:.1f}%"))
    return fig


def vrp_history(vrp: pd.DataFrame, implied_name: str) -> go.Figure:
    fig = _base("Implied today vs realized over the following 21 days")
    fig.add_trace(go.Scatter(x=vrp.index, y=vrp["implied"], name=implied_name, mode="lines",
                             line=dict(width=2, color=C["red"]), hovertemplate="%{y:.1f}%"))
    fig.add_trace(go.Scatter(x=vrp.index, y=vrp["forward_realized"], name="Forward realized 21d", mode="lines",
                             line=dict(width=1.5, color=C["blue"]), hovertemplate="%{y:.1f}%"))
    return fig


def vrp_spread(vrp: pd.DataFrame) -> go.Figure:
    fig = _base("Vol risk premium (implied - forward realized), vol points", height=300)
    pos, neg = vrp["vrp"].clip(lower=0), vrp["vrp"].clip(upper=0)
    fig.add_trace(go.Scatter(x=vrp.index, y=pos, name="Premium earned", mode="lines", fill="tozeroy",
                             line=dict(width=0.8, color=C["aqua"]), fillcolor="rgba(27,175,122,0.35)",
                             hovertemplate="%{y:.1f} pts"))
    fig.add_trace(go.Scatter(x=vrp.index, y=neg, name="Premium lost", mode="lines", fill="tozeroy",
                             line=dict(width=0.8, color=C["red"]), fillcolor="rgba(227,73,72,0.35)",
                             hovertemplate="%{y:.1f} pts"))
    fig.update_layout(yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix=""))
    return fig


def price_series(close: pd.Series, name: str) -> go.Figure:
    fig = _base(f"{name} spot", height=260, yaxis=dict(gridcolor=GRID, zeroline=False))
    fig.add_trace(go.Scatter(x=close.index, y=close, name=name, mode="lines",
                             line=dict(width=1.5, color=C["blue"]), hovertemplate="%{y:,.2f}"))
    fig.update_layout(showlegend=False)
    return fig


# ---------- vol cone ----------

def vol_cone(cone: pd.DataFrame, current: pd.Series, implied_30d: float | None = None) -> go.Figure:
    fig = _base("Vol cone: distribution of realized vol by horizon (5y lookback)", height=420,
                hovermode="x", xaxis=dict(title="Window (trading days)", showgrid=False, type="category"))
    x = [str(w) for w in cone.index]
    bands = [("q0", "q100", SEQ[0], "Min / max"), ("q10", "q90", SEQ[1], "10th / 90th pct"),
             ("q25", "q75", SEQ[2], "25th / 75th pct")]
    for lo, hi, color, name in bands:
        fig.add_trace(go.Scatter(x=x + x[::-1], y=list(cone[hi]) + list(cone[lo])[::-1], fill="toself",
                                 fillcolor=color, line=dict(width=0), name=name, hoverinfo="skip", opacity=0.9))
    fig.add_trace(go.Scatter(x=x, y=cone["q50"], name="Median", mode="lines",
                             line=dict(width=2, color=SEQ[6]), hovertemplate="%{y:.1f}%"))
    fig.add_trace(go.Scatter(x=x, y=current.values, name="Current realized", mode="lines+markers",
                             line=dict(width=2, color=C["orange"]), marker=dict(size=9),
                             hovertemplate="%{y:.1f}%"))
    if implied_30d is not None and pd.notna(implied_30d):
        fig.add_trace(go.Scatter(x=["21"], y=[implied_30d], name="Implied 30d", mode="markers",
                                 marker=dict(size=12, color=C["red"], symbol="diamond"),
                                 hovertemplate="%{y:.1f}%"))
    return fig


# ---------- term structure ----------

def term_structure(snapshot: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    fig = _base("VIX term structure today", height=340, hovermode="closest",
                xaxis=dict(title="Tenor (calendar days)", showgrid=False, type="category"))
    fig.add_trace(go.Scatter(x=snapshot["tenor_days"].astype(str), y=snapshot["vol"], mode="lines+markers+text",
                             text=[labels.get(i, i) for i in snapshot["index"]], textposition="top center",
                             textfont=dict(color=MUTED, size=11),
                             line=dict(width=2, color=C["blue"]), marker=dict(size=9), name="Implied vol",
                             hovertemplate="%{text}: %{y:.2f}%"))
    fig.update_layout(showlegend=False)
    return fig


def ratio_history(ratio: pd.Series, title: str, threshold: float = 1.0) -> go.Figure:
    fig = _base(title, height=300, yaxis=dict(gridcolor=GRID, zeroline=False))
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Ratio", mode="lines",
                             line=dict(width=1.5, color=C["blue"]), hovertemplate="%{y:.3f}"))
    fig.add_hline(y=threshold, line=dict(color=C["red"], width=1, dash="dot"),
                  annotation_text="Inverted above", annotation_position="top left",
                  annotation_font=dict(color=MUTED, size=11))
    fig.update_layout(showlegend=False)
    return fig


def single_series(s: pd.Series, title: str, suffix: str = "") -> go.Figure:
    fig = _base(title, height=280, yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix=suffix))
    fig.add_trace(go.Scatter(x=s.index, y=s, name=title, mode="lines",
                             line=dict(width=1.5, color=C["blue"]), hovertemplate="%{y:.2f}" + suffix))
    fig.update_layout(showlegend=False)
    return fig


# ---------- skew ----------

def smile_by_moneyness(chain: pd.DataFrame, info: dict) -> go.Figure:
    fig = _base(f"Implied vol smile, expiry {info['expiry'].date()} ({info['days']}d)", height=380,
                hovermode="closest", xaxis=dict(title="Strike / forward", showgrid=False, tickformat=".0%"))
    for typ, color in (("put", C["blue"]), ("call", C["orange"])):
        sub = chain[chain["type"] == typ]
        fig.add_trace(go.Scatter(x=sub["moneyness"], y=sub["iv_pct"], mode="lines+markers",
                                 name=f"OTM {typ}s", line=dict(width=2, color=color), marker=dict(size=6),
                                 customdata=sub[["strike", "delta", "mid"]].values,
                                 hovertemplate="K=%{customdata[0]:.1f}  IV=%{y:.1f}%  delta=%{customdata[1]:.2f}  mid=%{customdata[2]:.2f}"))
    fig.add_vline(x=1.0, line=dict(color=MUTED, width=1, dash="dot"))
    return fig


def smile_by_delta(chain: pd.DataFrame) -> go.Figure:
    fig = _base("Implied vol by delta (desk view)", height=380, hovermode="closest",
                xaxis=dict(title="Delta  (puts negative, calls positive)", showgrid=False, tickformat=".2f"))
    for typ, color in (("put", C["blue"]), ("call", C["orange"])):
        sub = chain[chain["type"] == typ].sort_values("delta")
        fig.add_trace(go.Scatter(x=sub["delta"], y=sub["iv_pct"], mode="lines+markers",
                                 name=f"OTM {typ}s", line=dict(width=2, color=color), marker=dict(size=6),
                                 hovertemplate="delta=%{x:.2f}  IV=%{y:.1f}%"))
    for d in (-0.25, 0.25):
        fig.add_vline(x=d, line=dict(color=MUTED, width=1, dash="dot"))
    return fig


# ---------- commodity curve ----------

def futures_curve(curve: pd.DataFrame, name: str) -> go.Figure:
    fig = _base(f"{name} futures curve", height=360, hovermode="closest",
                xaxis=dict(title="Contract month", showgrid=False),
                yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix=""))
    fig.add_trace(go.Scatter(x=curve["expiry_month"], y=curve["price"], mode="lines+markers",
                             name="Settlement", line=dict(width=2, color=C["blue"]), marker=dict(size=8),
                             customdata=curve["contract"], hovertemplate="%{customdata}: %{y:,.2f}"))
    fig.update_layout(showlegend=False)
    return fig


# ---------- VIX replication ----------

def vix_strip(strip: pd.DataFrame, title: str) -> go.Figure:
    fig = _base(title, height=340, hovermode="closest",
                xaxis=dict(title="Strike", showgrid=False),
                yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix="", title="dK/K² · Q(K)"))
    colors = {"put": C["blue"], "call": C["orange"], "atm": C["red"]}
    for side in ("put", "atm", "call"):
        sub = strip[strip["side"] == side]
        fig.add_trace(go.Bar(x=sub["strike"], y=sub["contribution"], name=f"{side}s" if side != "atm" else "K0",
                             marker=dict(color=colors[side], line=dict(width=0)),
                             hovertemplate="K=%{x}  contrib=%{y:.2e}"))
    fig.update_layout(bargap=0.15)
    return fig


# ---------- forecasting ----------

def forecast_lines(fc: pd.DataFrame, models: list[str]) -> go.Figure:
    fig = _base("Out-of-sample 21-day vol forecasts vs realized", height=400)
    fig.add_trace(go.Scatter(x=fc.index, y=fc["target"], name="Realized next 21d", mode="lines",
                             line=dict(width=2, color=TEXT), hovertemplate="%{y:.1f}%"))
    for i, mdl in enumerate(models):
        fig.add_trace(go.Scatter(x=fc.index, y=fc[mdl], name=mdl, mode="lines",
                                 line=dict(width=1.3, color=SERIES[i]), hovertemplate="%{y:.1f}%"))
    return fig


def mz_scatter(fc: pd.DataFrame, model: str) -> go.Figure:
    fig = _base(f"Mincer-Zarnowitz: realized vs {model}", height=360, hovermode="closest",
                xaxis=dict(title=f"{model} forecast", showgrid=False, ticksuffix="%"),
                yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix="%", title="Realized next 21d"))
    d = fc.dropna(subset=["target", model])
    fig.add_trace(go.Scatter(x=d[model], y=d["target"], mode="markers", name="days",
                             marker=dict(size=5, color=C["blue"], opacity=0.35),
                             hovertemplate="%{x:.1f}% -> %{y:.1f}%"))
    lim = [0, float(max(d[model].max(), d["target"].max()) * 1.05)]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="unbiased (45°)",
                             line=dict(width=1.5, color=C["red"], dash="dot"), hoverinfo="skip"))
    return fig


# ---------- crude research ----------

def samuelson_today(df: pd.DataFrame) -> go.Figure:
    fig = _base("Today: 60-day realized vol by contract vs time to expiry", height=360, hovermode="closest",
                xaxis=dict(title="Months to expiry", showgrid=False))
    fig.add_trace(go.Scatter(x=df["ttm_months"], y=df["rv"], mode="lines+markers", name="Contract",
                             line=dict(width=2, color=C["blue"]), marker=dict(size=7),
                             customdata=df["contract"], hovertemplate="%{customdata}: %{y:.1f}% at %{x:.1f}m"))
    fig.update_layout(showlegend=False)
    return fig


def samuelson_pooled(curve: pd.DataFrame) -> go.Figure:
    fig = _base("Pooled since 2018: realized vol by months-to-expiry bucket", height=360, hovermode="x",
                xaxis=dict(title="Months to expiry", showgrid=False, type="category"))
    x = list(curve.index.astype(str))
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(curve["q75"]) + list(curve["q25"])[::-1], fill="toself",
                             fillcolor=SEQ[1], line=dict(width=0), name="25th / 75th pct", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=curve["median_rv"], mode="lines+markers", name="Median",
                             line=dict(width=2, color=SEQ[5]), marker=dict(size=8),
                             customdata=curve["obs"], hovertemplate="%{y:.1f}%  (%{customdata} contract-days)"))
    return fig


def spread_vs_vol(panel: pd.DataFrame) -> go.Figure:
    fig = _base("Curve shape vs implied vol (daily, since 2018)", height=380, hovermode="closest",
                xaxis=dict(title="Dec/Dec spread, % (positive = backwardation)", showgrid=False, ticksuffix="%"),
                yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix="%", title="OVX"))
    for regime, color in (("Backwardation", C["orange"]), ("Contango", C["blue"])):
        d = panel[panel["regime"] == regime]
        fig.add_trace(go.Scatter(x=d["spread"], y=d["implied"], mode="markers", name=regime,
                                 marker=dict(size=5, color=color, opacity=0.4),
                                 customdata=d.index.strftime("%Y-%m-%d"),
                                 hovertemplate="%{customdata}: spread %{x:.1f}%, OVX %{y:.1f}%"))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dot"))
    return fig


# ---------- skew history ----------

def skew_history(hist: pd.DataFrame, metric: str = "rr25") -> go.Figure:
    fig = _base(f"{metric} history, 30-day tenor (own daily snapshots)", height=360,
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix=" pts"))
    assets = list(hist["asset"].unique())[:8]
    for i, a in enumerate(assets):
        d = hist[hist["asset"] == a]
        fig.add_trace(go.Scatter(x=pd.to_datetime(d["date"]), y=d[metric], mode="lines+markers", name=a,
                                 line=dict(width=1.5, color=SERIES[i]), marker=dict(size=6),
                                 hovertemplate="%{y:+.2f}"))
    return fig


# ---------- harvesting vol ----------

def straddle_cum(trades: pd.DataFrame) -> go.Figure:
    fig = _base("Short 1m ATM straddle, delta-hedged daily: cumulative P&L in % of spot (after costs)", height=360,
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix="%"))
    fig.add_trace(go.Scatter(x=trades.index, y=trades["cum_pnl_pct"], name="Cumulative P&L", mode="lines",
                             line=dict(width=2, color=C["blue"]), hovertemplate="%{y:.1f}%"))
    fig.add_trace(go.Scatter(x=trades.index, y=trades["drawdown_pct"], name="Drawdown", mode="lines", fill="tozeroy",
                             line=dict(width=0.8, color=C["red"]), fillcolor="rgba(227,73,72,0.3)", hovertemplate="%{y:.1f}%"))
    return fig


def straddle_vs_gamma_theta(trades: pd.DataFrame) -> go.Figure:
    fig = _base("Realized P&L vs the gamma-theta identity, per trade (% of spot)", height=360, hovermode="closest",
                xaxis=dict(title="½ Σ Γ S² (σ²impl dt − r²)", showgrid=False, ticksuffix="%"),
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix="%", title="Simulated hedged P&L"))
    fig.add_trace(go.Scatter(x=trades["gamma_theta_pct"], y=trades["pnl_pct"], mode="markers", name="trades",
                             marker=dict(size=6, color=C["blue"], opacity=0.5),
                             customdata=trades.index.strftime("%b %Y"), hovertemplate="%{customdata}: %{y:.2f}% vs %{x:.2f}%"))
    lim = [float(min(trades["gamma_theta_pct"].min(), trades["pnl_pct"].min())), float(max(trades["gamma_theta_pct"].max(), trades["pnl_pct"].max()))]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="identity", line=dict(width=1.5, color=C["red"], dash="dot"), hoverinfo="skip"))
    return fig


def pnl_vs_realized(trades: pd.DataFrame) -> go.Figure:
    fig = _base("Trade P&L (vol points) vs realized minus implied vol: the straddle is a short variance position", height=360,
                hovermode="closest", xaxis=dict(title="Realized − implied (vol pts)", showgrid=False),
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix="", title="P&L / vega (vol pts)"))
    x = trades["realized"] - trades["implied"]
    fig.add_trace(go.Scatter(x=x, y=trades["pnl_vol_pts"], mode="markers", name="straddle",
                             marker=dict(size=6, color=C["blue"], opacity=0.5), customdata=trades.index.strftime("%b %Y"),
                             hovertemplate="%{customdata}: %{y:.1f} pts at %{x:+.1f}"))
    fig.add_trace(go.Scatter(x=x, y=trades["var_swap_pts"], mode="markers", name="variance swap (same vega)",
                             marker=dict(size=5, color=C["orange"], opacity=0.5, symbol="diamond"), hovertemplate="%{y:.1f} pts"))
    return fig


def forward_curve(fc: pd.DataFrame) -> go.Figure:
    fig = _base("Forward vols from the VIX term structure", height=340)
    for i, col in enumerate(fc.columns):
        fig.add_trace(go.Scatter(x=fc.index, y=fc[col], name=col, mode="lines", line=dict(width=1.3, color=SERIES[i]),
                                 hovertemplate="%{y:.1f}%"))
    return fig


def event_implied_vs_realized(tab: pd.DataFrame) -> go.Figure:
    fig = _base("FOMC days: implied move read the day before vs realized close-to-close move", height=360, hovermode="x",
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix="%"))
    fig.add_trace(go.Bar(x=tab.index, y=tab["realized_abs_move_pct"], name="Realized |move|",
                         marker=dict(color=C["blue"], line=dict(width=0)), hovertemplate="%{y:.2f}%"))
    fig.add_trace(go.Scatter(x=tab.index, y=tab["implied_move_pct"], name="Implied move", mode="markers",
                             marker=dict(size=7, color=C["red"], symbol="diamond"), hovertemplate="%{y:.2f}%"))
    fig.update_layout(bargap=0.4)
    return fig


def strategy_paths(paths: pd.DataFrame) -> go.Figure:
    fig = _base("Sell every month vs sell only when the gap is wide: cumulative P&L (% of spot)", height=360,
                yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, ticksuffix="%"))
    for i, col in enumerate(paths.columns):
        fig.add_trace(go.Scatter(x=paths.index, y=paths[col], name=col, mode="lines", line=dict(width=1.6, color=SERIES[i]),
                                 hovertemplate="%{y:.1f}%"))
    return fig
