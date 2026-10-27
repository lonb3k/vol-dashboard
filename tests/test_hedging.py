"""Delta-hedged straddle: no P&L when realized equals implied, gamma-theta identity, sign of the edge."""
import numpy as np
import pandas as pd

from vol_dashboard import hedging as h


def _batch(s_real, s_impl, n=1500):
    return pd.DataFrame([h.hedge_one_trade(h.simulate_gbm_path(s_real, seed=i), s_impl, 0, 0) for i in range(n)])


def test_zero_edge_when_realized_equals_implied():
    d = _batch(0.2, 0.2)
    assert abs(d["pnl_vol_pts"].mean()) < 0.5


def test_sign_of_edge():
    assert _batch(0.15, 0.2, 800)["pnl_vol_pts"].mean() > 3
    assert _batch(0.30, 0.2, 800)["pnl_vol_pts"].mean() < -5


def test_gamma_theta_identity_tracks_pnl():
    d = _batch(0.25, 0.2, 800)
    assert d[["pnl_pct", "gamma_theta_pct"]].corr().iloc[0, 1] > 0.9


def test_flat_path_keeps_the_premium():
    path = np.full(22, 100.0)
    res = h.hedge_one_trade(path, 0.2, 0, 0)
    assert abs(res["pnl_pct"] - res["premium_pct"]) < 1e-9
    assert abs(res["premium_pct"] - 100 * h.straddle_price(100, 100, 0.2, 21 / 252) / 100) < 1e-9


def test_costs_reduce_pnl():
    path = h.simulate_gbm_path(0.2, seed=3)
    assert h.hedge_one_trade(path, 0.2, 5, 1)["pnl_pct"] < h.hedge_one_trade(path, 0.2, 0, 0)["pnl_pct"]
