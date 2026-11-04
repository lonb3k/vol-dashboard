"""Calibrate the free-data approximations on local Bloomberg extracts; write derived constants only.

Reads the Bloomberg folder (see src/vol_dashboard/bloomberg.py) and writes src/vol_dashboard/calibration.json:
summary statistics (ratios, correlations, error bands), never a price or vol series. Run on a machine that has
the extracts:

    python scripts/calibrate_from_bloomberg.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vol_dashboard import bloomberg as bbg, research  # noqa: E402
from vol_dashboard.data import load_ohlc  # noqa: E402

OUT = ROOT / "src" / "vol_dashboard" / "calibration.json"
# public market -> (Bloomberg price file, Bloomberg implied-vol file): implied estimated as ratio x realized
IMPLIED = {"Brent Crude": ("CO1", "iv_CO1"), "Copper": ("LMCDAS03", "iv_LP1"), "Aluminium": ("LMAHDS03", "iv_LA1")}


def main() -> None:
    if not bbg.available():
        sys.exit(f"No Bloomberg extracts in {bbg.folder()}")
    out = {"implied_from_realized": {}, "proxy_audit": {},
           "note": "Derived statistics from Bloomberg extracts (school terminal); no Bloomberg series is stored here."}
    for market, (px, iv) in IMPLIED.items():
        if not (bbg.folder() / f"{iv}.csv").exists():
            print(f"{market}: no {iv}.csv, skipped")
            continue
        c = research.calibrate(bbg.load_implied(iv)["Close"], bbg.load_prices(px)["Close"])
        c["source"] = f"Bloomberg {px} 30-day ATM implied vol against its 21-day realized vol"
        out["implied_from_realized"][market] = c
        print(market, c)
    for name, (label, ticker) in bbg.FREE_PROXY.items():
        code = next(m[1] for m in bbg.MARKETS if m[0] == name)
        bench = bbg.load_prices(code)["Close"]
        free = load_ohlc(ticker, start="2015-01-01")["Close"]
        if bench.empty or free.empty:
            continue
        a = research.proxy_audit(bench, free)
        out["proxy_audit"][label] = {"benchmark": name.replace(", Bloomberg", ""), "free_ticker": ticker,
                                     "weekly_corr": round(a["weekly_corr"], 2), "vol_ratio": round(a["vol_ratio"], 2)}
        print(label, out["proxy_audit"][label])
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
