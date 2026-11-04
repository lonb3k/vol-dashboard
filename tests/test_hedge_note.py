"""The corporate-hedging note: curve reading, percentiles and the stale-contract filter."""
import numpy as np
import pandas as pd

from vol_dashboard.data import drop_stale
from vol_dashboard.hedge_note import curve_metrics, curve_sentence, ordinal, percentile, vol_band


def _curve(prices, start="2026-12-01", step=1):
    months = pd.date_range(start, periods=len(prices), freq=f"{step}MS")
    return pd.DataFrame({"contract": [f"XX{i}.EX" for i in range(len(prices))], "expiry_month": months, "price": prices})


def test_backwardation_contango_and_flat_are_read_twelve_months_out():
    back = curve_metrics(_curve(np.linspace(100, 80, 13)))
    assert back["shape"] == "backwardation" and back["far_months"] == 12 and round(back["far_pct"], 6) == -20.0
    cont = curve_metrics(_curve(np.linspace(100, 110, 13)))
    assert cont["shape"] == "contango" and round(cont["m12_pct"], 6) == 10.0
    flat = curve_metrics(_curve(np.linspace(100, 101, 13)))
    assert flat["shape"] == "flat"


def test_a_short_curve_is_read_on_its_furthest_contract_and_says_so():
    c = curve_metrics(_curve([3272.0, 3204.0, 3195.5, 3153.25, 3140.0, 3130.0, 3100.0]))   # stops at 6 months
    assert np.isnan(c["m12_pct"]) and c["far_fallback"] and c["far_months"] == 6
    assert c["shape"] == "backwardation"
    assert "furthest contract" in curve_sentence("Aluminium", c)


def test_grain_curves_with_gaps_find_the_same_season_contract():
    # December, March, May, July, September, December: the 12-month point is next December
    months = pd.to_datetime(["2026-12-01", "2027-03-01", "2027-05-01", "2027-07-01", "2027-09-01", "2027-12-01"])
    cv = pd.DataFrame({"contract": [f"ZW{i}.CBT" for i in range(6)], "expiry_month": months,
                       "price": [600, 610, 615, 621, 630, 649.2]})
    c = curve_metrics(cv)
    assert c["far_contract"] == "ZW5" and not c["far_fallback"] and round(c["far_pct"], 1) == 8.2


def test_expired_contracts_are_dropped():
    cv = pd.DataFrame({"contract": ["BZZ26", "BZF27", "BZG27"], "price": [117.6, 102.0, 99.8],
                       "last_trade": pd.to_datetime(["2026-10-30", "2026-11-04", "2026-11-04"])})
    out = drop_stale(cv)
    assert list(out["contract"]) == ["BZF27", "BZG27"]


def test_percentiles_bands_and_ordinals():
    s = pd.Series(np.arange(1, 101, dtype=float))
    assert percentile(s, 80) == 80.0
    assert vol_band(85) == "high" and vol_band(15) == "low" and vol_band(50) == "normal"
    assert [ordinal(x) for x in (1, 2, 3, 4, 11, 12, 13, 21, 83)] == ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "83rd"]
