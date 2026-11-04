"""Local Bloomberg mode: read extracts kept outside the repository.

Bloomberg data is licensed, not redistributable, so it never enters this repo. When a folder of extracts
exists on the machine (by default ../bloomberg_private next to the repo, or the path in VOL_BLOOMBERG_DIR),
the app offers these European benchmarks next to the free-data universe:

    file                 content
    CO1.csv ...          daily Date, Open, High, Low, Close of the generic front contract
    iv_CO1.csv ...       daily Date, IV1M, IV3M: Bloomberg constant-maturity ATM implied vol (30 days, 3 months)
    curve_CO.csv ...     generic, contract, expiry, price, open_interest: today's curve

Tickers are prefixed BBG: (prices), BBGIV: (implied vol) and BBG: (curve roots) so the loaders in data.py can
route them here. Nothing in this module runs on a host without the folder, such as Streamlit Cloud.
"""
from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import pandas as pd

from .universe import Asset

REPO = Path(__file__).resolve().parents[2]
MONTHS = "FGHJKMNQUVXZ"
MIN_OI_SHARE = 0.02       # deferred contracts below 2% of the curve's peak open interest are dropped

# name, price file, implied-vol file, curve file, unit
MARKETS = [
    ("Brent (ICE, Bloomberg)", "CO1", "iv_CO1", "curve_CO", "$/bbl"),
    ("Gasoil (ICE, Bloomberg)", "QS1", "iv_QS1", "curve_QS", "$/t"),
    ("TTF gas (ICE Endex, Bloomberg)", "TZT1", None, "curve_TZT", "€/MWh"),
    ("Copper (LME, Bloomberg)", "LMCDAS03", "iv_LP1", "curve_LP", "$/t"),
    ("Aluminium (LME, Bloomberg)", "LMAHDS03", "iv_LA1", "curve_LA", "$/t"),
    ("Milling wheat (Euronext, Bloomberg)", "CA1", "iv_CA1", "curve_CA", "€/t"),
    ("Sugar No. 11 (ICE, Bloomberg)", "SB1", None, None, "c/lb"),
]


# The free-data contract the public dashboard uses for each benchmark (for the proxy audit).
FREE_PROXY = {"Brent (ICE, Bloomberg)": ("Brent Crude", "BZ=F"), "Gasoil (ICE, Bloomberg)": ("NY heating oil", "HO=F"),
              "TTF gas (ICE Endex, Bloomberg)": ("Natural Gas", "NG=F"), "Copper (LME, Bloomberg)": ("Copper", "HG=F"),
              "Aluminium (LME, Bloomberg)": ("Aluminium", "ALI=F"), "Milling wheat (Euronext, Bloomberg)": ("Wheat", "ZW=F"),
              "Sugar No. 11 (ICE, Bloomberg)": ("Sugar", "SB=F")}


def folder() -> Path:
    return Path(os.environ.get("VOL_BLOOMBERG_DIR", REPO.parent / "bloomberg_private"))


def available() -> bool:
    return any((folder() / f"{m[1]}.csv").exists() for m in MARKETS)


def assets() -> list[Asset]:
    """The Bloomberg markets whose price file exists, as Asset rows the app can use like any other."""
    out = []
    for name, px, iv, curve, _ in MARKETS:
        if not (folder() / f"{px}.csv").exists():
            continue
        has_iv = iv is not None and (folder() / f"{iv}.csv").exists()
        has_curve = curve is not None and (folder() / f"{curve}.csv").exists()
        out.append(Asset(name, f"BBG:{px}", f"BBGIV:{iv}" if has_iv else None, None, "Commodity (Bloomberg)",
                         f"BBG:{curve}" if has_curve else None, "BBG" if has_curve else None))
    return out


def unit(name: str) -> str:
    return next((m[4] for m in MARKETS if m[0] == name), "")


def load_prices(code: str) -> pd.DataFrame:
    path = folder() / f"{code}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    return drop_extraction_day(df, path)


def drop_extraction_day(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """An extract taken during the session ends on a partial day: drop rows dated on the day the file was written."""
    written = pd.Timestamp(dt.date.fromtimestamp(path.stat().st_mtime))
    return df[df.index < written] if len(df) and df.index[-1] >= written else df


def load_implied(code: str, column: str = "IV1M") -> pd.DataFrame:
    """Implied vol shaped like a price frame (Close = the vol), so vol-index code paths take it unchanged."""
    path = folder() / f"{code}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    return pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s})


def load_implied_terms(code: str) -> pd.DataFrame:
    path = folder() / f"{code}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["IV1M", "IV3M"])
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[["IV1M", "IV3M"]].apply(pd.to_numeric, errors="coerce")


def delivery_month(contract: str) -> pd.Timestamp:
    m = re.search(r"([FGHJKMNQUVXZ])(\d{1,2})$", str(contract).strip())
    y = int(m.group(2))
    return pd.Timestamp(year=2020 + y if y < 10 else 2000 + y, month=MONTHS.index(m.group(1)) + 1, day=1)


def load_curve(code: str) -> pd.DataFrame:
    """Curve in the shape of data.futures_curve: contract, expiry_month (delivery month), price."""
    path = folder() / f"{code}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["contract", "expiry_month", "price"])
    c = pd.read_csv(path).dropna(subset=["price"])
    c["expiry_month"] = c["contract"].map(delivery_month)
    if "open_interest" in c:
        c = c[c["open_interest"] >= MIN_OI_SHARE * c["open_interest"].max()]
    return c[["contract", "expiry_month", "price"]].sort_values("expiry_month").reset_index(drop=True)
