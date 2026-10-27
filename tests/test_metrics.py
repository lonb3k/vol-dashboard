"""Realized vol estimators must recover the volatility of a simulated GBM."""
import numpy as np
import pandas as pd
import pytest

from vol_dashboard import metrics as m

SIGMA = 0.25  # 25% annual


def simulate_ohlc(n_days: int = 4000, sigma: float = SIGMA, steps: int = 390, seed: int = 7) -> pd.DataFrame:
    """Daily OHLC from intraday GBM steps, no drift, no overnight gap."""
    rng = np.random.default_rng(seed)
    dt = 1 / (252 * steps)
    rows, price = [], 100.0
    for _ in range(n_days):
        path = price * np.exp(np.cumsum(rng.normal(-0.5 * sigma**2 * dt, sigma * np.sqrt(dt), steps)))
        rows.append((price, path.max(), path.min(), path[-1]))
        price = path[-1]
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


@pytest.fixture(scope="module")
def ohlc():
    return simulate_ohlc()


@pytest.mark.parametrize("name", list(m.ESTIMATORS))
def test_estimators_recover_sigma(ohlc, name):
    est = m.ESTIMATORS[name](ohlc, 21).dropna()
    assert abs(est.mean() - SIGMA * 100) < 1.5, f"{name} mean {est.mean():.2f} vs {SIGMA*100}"


def test_parkinson_is_more_efficient_than_close_to_close(ohlc):
    """Parkinson's variance of the estimate should be roughly 5x smaller (Parkinson 1980)."""
    c2c = m.rv_close_to_close(ohlc, 21).dropna()
    park = m.rv_parkinson(ohlc, 21).dropna()
    ratio = c2c.var() / park.var()
    assert 3.0 < ratio < 8.0, ratio


def test_vol_cone_is_ordered(ohlc):
    cone, current = m.vol_cone(ohlc)
    assert (cone["q25"] <= cone["q50"]).all() and (cone["q50"] <= cone["q75"]).all()
    assert set(current.index) == set(cone.index)


def test_vrp_alignment(ohlc):
    implied = pd.Series(30.0, index=ohlc.index)
    vrp = m.vol_risk_premium(implied, ohlc, 21)
    # constant implied of 30 vs ~25 realized: premium about +5
    assert abs(vrp["vrp"].mean() - 5) < 1.5
    # the last 21 days have no complete forward window
    assert vrp.index[-1] < ohlc.index[-21]
