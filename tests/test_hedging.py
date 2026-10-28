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


def test_short_download_merges_into_the_mirror(tmp_path, monkeypatch):
    import pandas as pd
    from vol_dashboard import data as dm
    long = pd.DataFrame({c: 1.0 for c in dm.OHLC}, index=pd.bdate_range("2010-01-01", periods=100, name="Date"))
    path = tmp_path / "idx_TEST.csv"
    long.to_csv(path)
    monkeypatch.setattr(dm, "_cache_path", lambda t: path)
    recent = pd.DataFrame({c: 2.0 for c in dm.OHLC}, index=pd.bdate_range("2010-05-01", periods=10))
    monkeypatch.setattr(dm.yf, "download", lambda *a, **k: recent)
    out = dm.load_ohlc("TEST", start="2010-05-01")
    assert out.index[0] >= pd.Timestamp("2010-05-01") and (out.loc[recent.index, "Close"] == 2.0).all()
    assert len(pd.read_csv(path)) == 100   # the mirror keeps its full history, updated where new data came in
