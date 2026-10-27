"""Forward variance arithmetic and the FOMC calendar."""
import numpy as np
import pandas as pd

from vol_dashboard import events as ev


def test_forward_vol_formula():
    idx = pd.bdate_range("2024-01-01", periods=3)
    closes = pd.DataFrame({"^VIX9D": 20.0, "^VIX": 20.0, "^VIX3M": 25.0}, index=idx)
    assert np.allclose(ev.forward_vol(closes, "^VIX9D", "^VIX"), 20.0)   # flat curve: forward = spot vol
    f = ev.forward_vol(closes, "^VIX", "^VIX3M").iloc[0]
    assert f > 25.0  # upward curve: forward above the long tenor
    expected = np.sqrt((91 * 25**2 - 30 * 20**2) / 61)
    assert abs(f - expected) < 1e-9


def test_inverted_curve_gives_nan_when_forward_variance_negative():
    idx = pd.bdate_range("2024-01-01", periods=1)
    closes = pd.DataFrame({"^VIX9D": 80.0, "^VIX": 20.0}, index=idx)
    assert np.isnan(ev.forward_vol(closes, "^VIX9D", "^VIX").iloc[0])


def test_fomc_calendar_is_eight_per_year_and_sorted():
    years = pd.Series(ev.FOMC_DATES.year).value_counts()
    assert (years.drop(2020) == 8).all() and years[2020] == 8
    assert ev.FOMC_DATES.is_monotonic_increasing


def test_event_window_flag():
    idx = pd.bdate_range("2024-01-15", "2024-02-05")
    closes = pd.DataFrame({"^VIX9D": 15.0, "^VIX": 14.0}, index=idx)
    spx = pd.Series(np.linspace(100, 101, len(idx)), index=idx)
    hist = ev.event_premium_history(closes, spx)
    assert hist.loc["2024-01-25", "event_in_window"]      # FOMC on 2024-01-31, 6 days ahead
    assert not hist.loc["2024-01-16", "event_in_window"]  # 15 days ahead
    assert (hist["implied_move_pct"] >= 0).all()
