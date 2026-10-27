"""Data access layer.

Primary source: Yahoo Finance via yfinance. Every download is mirrored to data/*.csv so the
dashboard can run offline and so the extracts can be swapped for Bloomberg exports with the
same column layout (Date, Open, High, Low, Close).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

from .universe import FUTURES_MONTH_CODES

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)

OHLC = ["Open", "High", "Low", "Close"]


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "idx_").replace("=", "_").replace(".", "_")
    return DATA_DIR / f"{safe}.csv"


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    return df


def load_ohlc(ticker: str, start: str = "2010-01-01", offline: bool = False) -> pd.DataFrame:
    """Daily OHLC for one ticker. Returns an empty frame if nothing is available."""
    path = _cache_path(ticker)
    if offline and path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)

    try:
        raw = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
    except Exception:
        raw = pd.DataFrame()
    raw = _flatten(raw)

    if raw.empty or "Close" not in raw:
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)
        return pd.DataFrame(columns=OHLC)

    df = raw[OHLC].dropna(subset=["Close"]).copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df.to_csv(path)
    return df


def load_closes(tickers: list[str], start: str = "2010-01-01", offline: bool = False) -> pd.DataFrame:
    """Close prices for several tickers, aligned on the union of dates."""
    series = {}
    for t in tickers:
        df = load_ohlc(t, start=start, offline=offline)
        if not df.empty:
            series[t] = df["Close"]
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def futures_curve(root: str, exchange: str, n_months: int = 12) -> pd.DataFrame:
    """Settlement of the next `n_months` listed futures contracts, e.g. CLZ26.NYM.

    Yahoo lists individual contracts under ROOT + month code + 2-digit year + .EXCHANGE.
    Returns columns: contract, expiry_month, price. Missing contracts are skipped.
    """
    today = dt.date.today()
    rows = []
    year, month = today.year, today.month
    for _ in range(n_months + 1):
        month += 1
        if month > 12:
            month, year = 1, year + 1
        code = f"{root}{FUTURES_MONTH_CODES[month - 1]}{str(year)[-2:]}.{exchange}"
        try:
            hist = yf.Ticker(code).history(period="5d")
        except Exception:
            continue
        if hist.empty:
            continue
        rows.append({
            "contract": code,
            "expiry_month": pd.Timestamp(year=year, month=month, day=1),
            "price": float(hist["Close"].iloc[-1]),
        })
    curve = pd.DataFrame(rows)
    if not curve.empty:
        curve.to_csv(DATA_DIR / f"curve_{root}.csv", index=False)
    return curve


def option_chain(ticker: str, target_days: int = 30) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    """Calls and puts for the listed expiry closest to `target_days`, with the spot price.

    Returns (chain, expiry, spot). `chain` columns: strike, type, last, iv_yahoo, bid, ask, volume, openInterest.
    Yahoo's own IV is unreliable (it ignores the forward), so it is kept only for reference.
    """
    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        return pd.DataFrame(), pd.NaT, float("nan")

    today = pd.Timestamp.today().normalize()
    exp_dates = [pd.Timestamp(e) for e in expiries]
    # closest expiry to target, but at least 7 days out to avoid pin/gamma noise
    candidates = [e for e in exp_dates if (e - today).days >= 7] or exp_dates
    expiry = min(candidates, key=lambda e: abs((e - today).days - target_days))

    chain = tk.option_chain(expiry.strftime("%Y-%m-%d"))
    cols = ["strike", "lastPrice", "impliedVolatility", "bid", "ask", "volume", "openInterest"]
    calls = chain.calls[cols].assign(type="call")
    puts = chain.puts[cols].assign(type="put")
    df = pd.concat([calls, puts], ignore_index=True).rename(columns={"impliedVolatility": "iv_yahoo", "lastPrice": "last"})

    hist = tk.history(period="5d")
    spot = float(hist["Close"].iloc[-1]) if not hist.empty else float("nan")
    return df, expiry, spot


def option_chain_for_expiry(ticker: str, expiry: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    """Calls and puts for one exact listed expiry (same layout as `option_chain`)."""
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry.strftime("%Y-%m-%d"))
    cols = ["strike", "lastPrice", "impliedVolatility", "bid", "ask", "volume", "openInterest"]
    df = pd.concat([chain.calls[cols].assign(type="call"), chain.puts[cols].assign(type="put")], ignore_index=True)
    df = df.rename(columns={"impliedVolatility": "iv_yahoo", "lastPrice": "last"})
    hist = tk.history(period="5d")
    spot = float(hist["Close"].iloc[-1]) if not hist.empty else float("nan")
    return df, expiry, spot


def listed_expiries(ticker: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(e) for e in yf.Ticker(ticker).options]
