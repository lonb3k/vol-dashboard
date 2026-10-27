"""Asset universe: underlying ticker, matching CBOE vol index, and an optionable ETF proxy.

Every ticker is a Yahoo Finance symbol so the project runs on free data.
On a Bloomberg terminal the same series are e.g. SPX Index, VIX Index, CL1 Comdty, OVX Index.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    name: str
    underlying: str          # price series
    vol_index: str | None    # CBOE implied vol index, 30-day, in vol points
    option_proxy: str | None # liquid ETF whose option chain is used for skew
    asset_class: str         # "Equity" or "Commodity"
    futures_root: str | None = None  # Yahoo futures root for curve, e.g. "CL"
    exchange: str | None = None      # Yahoo futures exchange suffix, e.g. "NYM"


UNIVERSE: list[Asset] = [
    Asset("S&P 500", "^GSPC", "^VIX", "SPY", "Equity"),
    Asset("Nasdaq 100", "^NDX", "^VXN", "QQQ", "Equity"),
    Asset("Russell 2000", "^RUT", None, "IWM", "Equity"),  # ^RVX no longer on Yahoo
    Asset("Euro Stoxx 50", "^STOXX50E", None, "FEZ", "Equity"),
    Asset("WTI Crude", "CL=F", "^OVX", "USO", "Commodity", "CL", "NYM"),
    Asset("Gold", "GC=F", "^GVZ", "GLD", "Commodity", "GC", "CMX"),
    Asset("Silver", "SI=F", None, "SLV", "Commodity", "SI", "CMX"),  # ^VXSLV has no history on Yahoo
    Asset("Natural Gas", "NG=F", None, "UNG", "Commodity", "NG", "NYM"),
]

BY_NAME = {a.name: a for a in UNIVERSE}

# CBOE VIX term structure: index -> tenor in calendar days
VIX_TERM = {"^VIX9D": 9, "^VIX": 30, "^VIX3M": 91, "^VIX6M": 182, "^VIX1Y": 365}

# Extra S&P 500 vol-of-vol / tail indicators
VIX_EXTRAS = {"^VVIX": "VVIX (vol of vol)", "^SKEW": "CBOE SKEW"}

FUTURES_MONTH_CODES = "FGHJKMNQUVXZ"
