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
