"""What a corporate hedge costs, in percent of the futures price, from one implied vol.

Black-76 on the futures contract, flat vol (no skew: a desk would price the wings off the smile, which makes
puts dearer and lowers the call strike of a zero-cost collar), European exercise, discounted at a flat rate.

    ATM put / call          premium of the at-the-money option
    90% put                 a producer's floor 10% under today's price
    producer collar         buy the 90% put, sell the call that pays for it: the cap that comes free with the floor
    consumer collar         buy the 110% call, sell the put that pays for it: the floor that comes free with the cap
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .options import RISK_FREE, black76_price


def premium(strike_pct: float, t: float, vol: float, is_call: bool, rate: float = RISK_FREE) -> float:
    """Option premium in % of the futures price (F = 100)."""
    return float(black76_price(100.0, strike_pct, t, vol, is_call, np.exp(-rate * t)))


def zero_cost_strike(bought_strike: float, bought_call: bool, t: float, vol: float, rate: float = RISK_FREE) -> float:
    """Strike of the option sold to finance the one bought, in % of F."""
    target = premium(bought_strike, t, vol, bought_call, rate)
    sold_call = not bought_call
    f = lambda k: premium(k, t, vol, sold_call, rate) - target  # noqa: E731
    lo, hi = (100.0, 400.0) if sold_call else (1.0, 100.0)
    return float(brentq(f, lo, hi, xtol=1e-6))


def hedge_table(vol_1m: float, vol_3m: float | None = None, rate: float = RISK_FREE) -> dict:
    """Costs for 1-month and 3-month hedges. Vols in decimals."""
    v3 = vol_3m if vol_3m is not None and np.isfinite(vol_3m) else vol_1m
    out = {}
    for label, t, v in (("1M", 1 / 12, vol_1m), ("3M", 0.25, v3)):
        out[label] = {"vol": v * 100, "atm_put": premium(100, t, v, False, rate), "put_90": premium(90, t, v, False, rate),
                      "call_110": premium(110, t, v, True, rate),
                      "producer_cap": zero_cost_strike(90, False, t, v, rate),
                      "consumer_floor": zero_cost_strike(110, True, t, v, rate)}
    return out
