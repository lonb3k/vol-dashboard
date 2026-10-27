"""Black-76 inversion, implied forward and skew quotes on synthetic chains."""
import numpy as np
import pandas as pd

from vol_dashboard import options as o


def test_implied_vol_round_trip():
    F, t, disc = 100.0, 0.25, 0.99
    for K, sigma, is_call in [(90, 0.2, False), (100, 0.35, True), (115, 0.5, True), (80, 0.6, False)]:
        price = o.black76_price(F, K, t, sigma, is_call, disc)
        assert abs(o.implied_vol(price, F, K, t, is_call, disc) - sigma) < 1e-5


def _chain(spot, F, t, disc, smile):
    """Chain priced with a given IV(moneyness) function and a 2% bid-ask."""
    rows = []
    for K in np.arange(spot * 0.75, spot * 1.25, 1.0):
        iv = smile(K / F)
        for typ in ("call", "put"):
            p = o.black76_price(F, K, t, iv, typ == "call", disc)
            rows.append({"strike": K, "type": typ, "bid": p * 0.99, "ask": p * 1.01, "last": p,
                         "iv_yahoo": np.nan, "volume": 10, "openInterest": 10})
    return pd.DataFrame(rows)


def test_forward_recovered_from_parity():
    spot, t = 100.0, 30 / 365
    disc = np.exp(-o.RISK_FREE * t)
    F = 101.5  # dividends/carry push the forward away from spot
    chain = _chain(spot, F, t, disc, lambda m: 0.2)
    cleaned, info = o.clean_chain(chain, spot, pd.Timestamp.today().normalize() + pd.Timedelta(days=30))
    assert abs(info["forward"] - F) < 0.05
    # flat smile in, flat smile out
    assert cleaned["iv_pct"].std() < 0.05
    assert abs(cleaned["iv_pct"].mean() - 20) < 0.05


def test_risk_reversal_sign_follows_skew():
    spot, t = 100.0, 30 / 365
    disc = np.exp(-o.RISK_FREE * t)
    exp = pd.Timestamp.today().normalize() + pd.Timedelta(days=30)
    put_skew = lambda m: 0.2 + 0.4 * max(1 - m, 0)      # equity-like
    call_skew = lambda m: 0.3 + 0.4 * max(m - 1, 0)     # crude-like
    for smile, sign in [(put_skew, -1), (call_skew, +1)]:
        cleaned, _ = o.clean_chain(_chain(spot, spot, t, disc, smile), spot, exp)
        sk = o.skew_metrics(cleaned)
        assert np.sign(sk["rr25"]) == sign
        assert np.sign(sk["skew_90_110"]) == -sign
        assert sk["bf25"] > 0
