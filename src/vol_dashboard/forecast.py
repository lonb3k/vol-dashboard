"""Out-of-sample volatility forecasting: HAR-RV, GARCH(1,1), implied vol, and HAR + implied.

Question answered: at a 21-trading-day horizon, does the option market (implied vol) forecast
realized vol better than models built on past realized vol alone?

Models
------
- RV22 (naive): the last 22-day realized vol, i.e. "vol stays where it is".
- HAR-RV (Corsi, 2009): regression of log forward RV on log daily, weekly and monthly RV.
  Daily RV uses the Parkinson range so a single day is not pure noise.
- GARCH(1,1): fitted by maximum likelihood on daily returns, variance forecast summed to h days.
- Implied: the CBOE 30-day index as a direct forecast (biased up by the variance premium).
- HAR + IV: HAR-RV with log implied as an extra regressor, the desk's usual "encompassing" test.

Evaluation
----------
Expanding window, refit every `refit` days, forecasts compared on the realized vol over the
next h days. Losses: RMSE in vol points, QLIKE on variance (robust to noise in the realized
proxy), and a Mincer-Zarnowitz regression RV = a + b * forecast, where (a, b) = (0, 1) means
an unbiased forecast. Targets overlap (daily series, 21-day horizon), so the MZ standard errors
are not reported; the R^2 and the bias are what matter here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .metrics import ANN, log_returns

HORIZON = 21


def _features(df: pd.DataFrame) -> pd.DataFrame:
    r = log_returns(df["Close"])
    park_day = np.sqrt((np.log(df["High"] / df["Low"]) ** 2) / (4 * np.log(2)) * ANN) * 100
    rv5 = r.rolling(5).std() * np.sqrt(ANN) * 100
    rv22 = r.rolling(22).std() * np.sqrt(ANN) * 100
    fwd = (r.rolling(HORIZON).std() * np.sqrt(ANN) * 100).shift(-HORIZON)
    out = pd.DataFrame({"rv1": park_day, "rv5": rv5, "rv22": rv22, "target": fwd, "ret": r})
    return out.replace([np.inf, -np.inf], np.nan)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def _garch_forecast(returns: pd.Series, h: int) -> float:
    from arch import arch_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(returns * 100, vol="GARCH", p=1, q=1, dist="normal", mean="Constant")
        res = am.fit(disp="off", show_warning=False)
        var = res.forecast(horizon=h).variance.iloc[-1].values  # daily variances in %^2
    return float(np.sqrt(var.mean() * ANN))


def run_forecasts(df: pd.DataFrame, implied: pd.Series | None, min_train: int = 504,
                  refit: int = 21, garch: bool = True) -> pd.DataFrame:
    """Rolling out-of-sample forecasts. Returns a frame indexed by date with one column per model
    plus `target`."""
    feats = _features(df)
    if implied is not None:
        feats["iv"] = implied.reindex(feats.index).ffill()
    feats = feats.dropna(subset=["rv1", "rv5", "rv22", "ret"])
    feats = feats[feats["rv1"] > 0]
    logf = np.log(feats[["rv1", "rv5", "rv22"]])
    has_iv = "iv" in feats and feats["iv"].notna().mean() > 0.9
    if has_iv:
        logf["iv"] = np.log(feats["iv"])
    log_target = np.log(feats["target"])

    rows = []
    beta_har = beta_hiv = None
    garch_val = np.nan
    idx = feats.index
    for i in range(min_train, len(feats)):
        if (i - min_train) % refit == 0:
            # training rows need a complete target: last usable row is i - HORIZON
            tr = slice(0, i - HORIZON)
            Xh = logf[["rv1", "rv5", "rv22"]].values[tr]
            yh = log_target.values[tr]
            ok = ~np.isnan(yh)
            beta_har = _ols(Xh[ok], yh[ok])
            if has_iv:
                Xi = logf[["rv1", "rv5", "rv22", "iv"]].values[tr]
                beta_hiv = _ols(Xi[ok], yh[ok])
            if garch:
                garch_val = _garch_forecast(feats["ret"].iloc[:i], HORIZON)
        x = logf.iloc[i]
        row = {
            "date": idx[i],
            "target": feats["target"].iloc[i],
            "RV22": feats["rv22"].iloc[i],
            "HAR-RV": float(np.exp(beta_har[0] + beta_har[1:] @ x[["rv1", "rv5", "rv22"]].values)),
        }
        if garch:
            row["GARCH(1,1)"] = garch_val
        if has_iv:
            row["Implied"] = feats["iv"].iloc[i]
            row["HAR+IV"] = float(np.exp(beta_hiv[0] + beta_hiv[1:] @ x[["rv1", "rv5", "rv22", "iv"]].values))
        rows.append(row)
    out = pd.DataFrame(rows).set_index("date")
    return out


def evaluate(fc: pd.DataFrame) -> pd.DataFrame:
    """Loss table per model on rows where the target is known."""
    data = fc.dropna(subset=["target"])
    y = data["target"].values
    rows = []
    for model in [c for c in data.columns if c != "target"]:
        f = data[model].values
        ok = np.isfinite(f) & (f > 0)
        yy, ff = y[ok], f[ok]
        err = ff - yy
        ratio = (yy**2) / (ff**2)
        qlike = np.mean(ratio - np.log(ratio) - 1)
        a, b = _ols(ff.reshape(-1, 1), yy)
        resid = yy - (a + b * ff)
        r2 = 1 - resid.var() / yy.var()
        rows.append({"Model": model, "Bias (pts)": err.mean(), "RMSE (pts)": np.sqrt((err**2).mean()),
                     "QLIKE": qlike, "MZ alpha": a, "MZ beta": b, "MZ R2": r2, "N": int(ok.sum())})
    return pd.DataFrame(rows).set_index("Model")
