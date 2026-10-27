"""The CBOE strip must return the input variance on a flat-vol Black-Scholes chain."""
import numpy as np

from vol_dashboard import vix


def test_strip_recovers_flat_vol():
    for sigma in (0.12, 0.20, 0.45):
        calls, puts = vix.synthetic_chain(100.0, sigma, 30 / 365, r=0.0, step=0.5, width=0.7)
        ev = vix._strip_variance(calls, puts, 30 / 365, 0.0)
        assert abs(np.sqrt(ev.variance) - sigma) < 0.002, sigma
        assert abs(ev.forward - 100) < 0.01


def test_strip_with_rates():
    sigma, t, r = 0.2, 30 / 365, 0.05
    calls, puts = vix.synthetic_chain(100.0, sigma, t, r=r, step=0.5, width=0.7)
    ev = vix._strip_variance(calls, puts, t, r)
    assert abs(ev.forward - 100 * np.exp(r * t)) < 0.01
    assert abs(np.sqrt(ev.variance) - sigma) < 0.002


def test_put_share_rises_with_put_skew():
    """A strip priced with put skew puts more of the variance in the puts."""
    import pandas as pd
    from scipy.stats import norm
    t, F = 30 / 365, 100.0
    strikes = np.arange(50, 150, 0.5)

    def chain(smile):
        iv = np.array([smile(k / F) for k in strikes])
        d1 = (np.log(F / strikes) + 0.5 * iv**2 * t) / (iv * np.sqrt(t)); d2 = d1 - iv * np.sqrt(t)
        c = F * norm.cdf(d1) - strikes * norm.cdf(d2); p = strikes * norm.cdf(-d2) - F * norm.cdf(-d1)
        idx = pd.Index(strikes, name="strike")
        return pd.DataFrame({"bid": c, "ask": c}, index=idx), pd.DataFrame({"bid": p, "ask": p}, index=idx)

    flat = vix._strip_variance(*chain(lambda m: 0.2), t, 0.0)
    skewed = vix._strip_variance(*chain(lambda m: 0.2 + 0.3 * max(1 - m, 0)), t, 0.0)
    assert skewed.put_share > flat.put_share
    assert skewed.variance > flat.variance
