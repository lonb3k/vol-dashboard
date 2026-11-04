"""Local Bloomberg mode, hedge costs and the implied-vol research, on synthetic extracts."""
import json

import numpy as np
import pandas as pd
import pytest

from vol_dashboard import bloomberg as bbg, hedge_costs, research
from vol_dashboard.data import futures_curve, load_ohlc
from vol_dashboard.options import black76_price


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setenv("VOL_BLOOMBERG_DIR", str(tmp_path))
    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.standard_normal(400) * 0.02))
    pd.DataFrame({"Date": dates, "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close}).to_csv(tmp_path / "CO1.csv", index=False)
    pd.DataFrame({"Date": dates, "IV1M": 35.0, "IV3M": 32.0}).to_csv(tmp_path / "iv_CO1.csv", index=False)
    pd.DataFrame({"generic": ["CO1 Comdty", "CO2 Comdty", "CO3 Comdty"], "contract": ["COF7", "COG7", "COH8"],
                  "expiry": ["30/11/2026", "30/12/2026", "30/01/2028"], "price": [100.0, 99.0, 80.0],
                  "open_interest": [500000, 300000, 100]}).to_csv(tmp_path / "curve_CO.csv", index=False)
    return tmp_path


def test_bloomberg_assets_appear_only_when_their_files_exist(folder):
    names = [a.name for a in bbg.assets()]
    assert names == ["Brent (ICE, Bloomberg)"]
    a = bbg.assets()[0]
    assert a.underlying == "BBG:CO1" and a.vol_index == "BBGIV:iv_CO1" and a.futures_root == "BBG:curve_CO"


def test_loaders_route_bloomberg_tickers_to_the_local_folder(folder):
    px = load_ohlc("BBG:CO1", start="2020-01-01")
    assert len(px) == 400 and list(px.columns) == ["Open", "High", "Low", "Close"]
    iv = load_ohlc("BBGIV:iv_CO1", start="2020-01-01")
    assert (iv["Close"] == 35.0).all()
    cv = futures_curve("BBG:curve_CO", "BBG")
    # delivery months from the contract codes, the illiquid March 2028 contract dropped
    assert list(cv["contract"]) == ["COF7", "COG7"] and cv["expiry_month"].iloc[0] == pd.Timestamp("2027-01-01")


def test_no_bloomberg_folder_means_no_bloomberg_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("VOL_BLOOMBERG_DIR", str(tmp_path / "missing"))
    assert not bbg.available() and bbg.assets() == []


def test_hedge_costs_are_consistent():
    t = hedge_costs.hedge_table(0.30, 0.30)["3M"]
    # put-call parity on the futures at the money: call = put
    assert hedge_costs.premium(100, 0.25, 0.30, True) == pytest.approx(t["atm_put"], rel=1e-9)
    # a zero-cost collar is zero cost, and its free strikes sit on the right side of the money
    assert hedge_costs.premium(t["producer_cap"], 0.25, 0.30, True) == pytest.approx(t["put_90"], abs=1e-6)
    assert t["producer_cap"] > 100 > t["consumer_floor"]
    assert hedge_costs.hedge_table(0.50)["3M"]["atm_put"] > t["atm_put"]


def test_variance_premium_and_calibration_recover_a_known_markup():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    close = pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(1500) * 0.015)), index=idx)
    rv = research.realized(close)
    iv = (rv * 1.2).dropna()
    cal = research.calibrate(iv, close)
    assert cal["ratio"] == pytest.approx(1.2, abs=1e-9) and cal["mae_pts"] == pytest.approx(0.0, abs=1e-9)
    flat = pd.Series(40.0, index=idx)
    vp = research.variance_premium(flat, close)
    assert vp["mean"] > 0 and 0 <= vp["hit"] <= 100


def test_term_structure_reads_inversion():
    idx = pd.bdate_range("2020-01-01", periods=300)
    iv1 = pd.Series(np.where(np.arange(300) < 150, 40.0, 25.0), index=idx)
    iv3 = pd.Series(30.0, index=idx)
    ts = research.term_structure(iv1, iv3)
    assert ts["share_inverted"] == pytest.approx(50.0) and not ts["now_inverted"]
    assert ts["next_change_inverted"] < 0


def test_the_committed_calibration_holds_only_summary_numbers():
    from vol_dashboard.hedge_note import CALIBRATION
    data = json.loads(CALIBRATION.read_text())
    for v in data["implied_from_realized"].values():
        assert set(v) <= {"ratio", "mae_pts", "p90_err_pts", "n", "start", "end", "source"}
    for v in data["proxy_audit"].values():
        assert set(v) <= {"benchmark", "free_ticker", "weekly_corr", "vol_ratio"}
