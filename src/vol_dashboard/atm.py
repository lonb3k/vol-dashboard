"""At-the-money implied vol history, so that an ATM straddle is sold at an ATM price.

The CBOE vol indices (VIX, VXN, OVX, GVZ) are model-free: they price a variance swap, which loads
the whole smile, including the expensive downside puts. An at-the-money straddle trades at the
ATM vol, which on the S&P is about 20% lower than VIX in calm markets. Selling the straddle at
the VIX level overstates its premium by the price of the skew.

History of S&P ATM vol, with free data:
- 1986 to Sep 2021: the old CBOE index VXO, built from eight near-the-money S&P 100 options.
  It annualizes with trading days, which lifts it by sqrt(30/22) (Carr and Wu, 2006, "A tale of two
  indices"), so ATM = VXO / sqrt(30/22). It is S&P 100, not S&P 500: close, not identical.
- After Sep 2021 (VXO discontinued): the ratio ATM / VIX is modelled on the overlap as
  a + b ln(VIX) + c SKEW (the ratio rises with VIX, the skew flattens in panics; it falls with the
  CBOE SKEW index) and applied to VIX. Out of sample on 27 Oct 2026 the model gives 0.76 against
  0.78 measured live (SPY 30-day ATM vol from the daily skew snapshot over the VIX close; residual
  s.d. of the fit about 0.04).
Other assets have no ATM history: their ratio is measured once, ETF 30-day ATM vol from the live
27 Oct 2026 skew snapshot (Black-76 on the parity forward, real bid-ask) over the index close, and
held constant. That is an approximation and the app says so. Crude has a call skew (25-delta risk
reversal +4), so its ratio is close to one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
VXO_FILE = DATA_DIR / "fred_VXOCLS.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VXOCLS"
TRADING_DAY_BIAS = np.sqrt(30 / 22)

# ETF 30d ATM vol (live skew snapshot, 27 Oct 2026) / vol index close the same day
MEASURED_RATIO = {"^VIX": 0.783, "^VXN": 0.853, "^OVX": 0.946, "^GVZ": 0.902}


def load_vxo(offline: bool = False) -> pd.Series:
    """VXO close from FRED (1986 to Sep 2021). The series is frozen, so the committed file is enough."""
    if not VXO_FILE.exists() and not offline:
        try:
            r = requests.get(FRED_URL, timeout=30)
            r.raise_for_status()
            VXO_FILE.write_bytes(r.content)
        except Exception:
            return pd.Series(dtype=float, name="VXO")
    if not VXO_FILE.exists():
        return pd.Series(dtype=float, name="VXO")
    df = pd.read_csv(VXO_FILE, parse_dates=[0], index_col=0)
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().rename("VXO")


def fit_ratio_model(vxo: pd.Series, vix: pd.Series, skew: pd.Series, start: str = "2004-01-01") -> dict:
    """OLS of ATM / VIX on ln(VIX) and SKEW over the VXO overlap."""
    df = pd.concat({"atm": vxo / TRADING_DAY_BIAS, "vix": vix, "skew": skew}, axis=1, sort=True).dropna()
    df = df[df.index >= pd.Timestamp(start)]
    y = (df["atm"] / df["vix"]).values
    X = np.column_stack([np.ones(len(df)), np.log(df["vix"]), df["skew"]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return {"a": float(beta[0]), "b_lnvix": float(beta[1]), "c_skew": float(beta[2]),
            "r2": float(1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()), "resid_sd": float(resid.std()),
            "obs": int(len(df)), "mean_ratio": float(y.mean()), "end": df.index[-1]}


def sp_atm_vol(vix: pd.Series, vxo: pd.Series, skew: pd.Series) -> tuple[pd.Series, pd.Series, dict]:
    """S&P 500 30-day ATM vol, daily: VXO-based where it exists, modelled ratio x VIX after.
    Returns (atm vol in points, source label per day, model fit)."""
    fit = fit_ratio_model(vxo, vix, skew)
    from_vxo = (vxo / TRADING_DAY_BIAS).reindex(vix.index)
    sk = skew.reindex(vix.index).ffill()
    ratio = (fit["a"] + fit["b_lnvix"] * np.log(vix) + fit["c_skew"] * sk).clip(0.6, 1.05)
    modelled = vix * ratio
    atm = from_vxo.where(from_vxo.notna() & (vix.index <= fit["end"]), modelled)
    source = pd.Series(np.where(from_vxo.notna() & (vix.index <= fit["end"]), "VXO", "model"), index=vix.index)
    return atm.dropna(), source, fit


def atm_vol(vol_index: str, index_close: pd.Series, vxo: pd.Series | None = None, skew: pd.Series | None = None) -> tuple[pd.Series, str]:
    """ATM vol proxy for any asset with a CBOE vol index, and a one-line description of the method."""
    if vol_index == "^VIX" and vxo is not None and not vxo.empty and skew is not None and not skew.empty:
        atm, _, fit = sp_atm_vol(index_close, vxo, skew)
        return atm, (f"VXO / sqrt(30/22) to {fit['end']:%b %Y}, then VIX x ({fit['a']:.3f} + {fit['b_lnvix']:.3f} ln VIX "
                     f"{fit['c_skew']:+.4f} SKEW), R² {fit['r2']:.2f}")
    ratio = MEASURED_RATIO.get(vol_index)
    if ratio is None:
        return index_close, "no ATM history: sold at the model-free index (overstates the premium)"
    return index_close * ratio, f"{vol_index} x {ratio:.3f}, ratio measured once on the live ETF chain (27 Oct 2026), held constant"
