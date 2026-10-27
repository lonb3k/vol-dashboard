"""Daily skew snapshot. Appends one row per asset and tenor to data/skew_history.csv.

Run after the US close (cron, launchd, or the GitHub Actions workflow in .github/workflows).
Free data has no history of the 25-delta risk reversal; this builds one.

    python scripts/snapshot_skew.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")  # yfinance triggers pandas deprecation noise

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vol_dashboard.data import DATA_DIR, option_chain  # noqa: E402
from vol_dashboard.options import clean_chain, skew_metrics  # noqa: E402
from vol_dashboard.universe import UNIVERSE  # noqa: E402

HISTORY = DATA_DIR / "skew_history.csv"
TENORS = (30, 90)
COLUMNS = ["date", "asset", "proxy", "tenor_target", "expiry", "days", "spot", "forward", "stale",
           "n_quotes", "atm_iv", "put25", "call25", "put10", "call10", "rr25", "bf25", "skew_90_110"]


def snapshot() -> pd.DataFrame:
    today = date.today().isoformat()
    rows = []
    for a in UNIVERSE:
        if not a.option_proxy:
            continue
        for tenor in TENORS:
            try:
                raw, expiry, spot = option_chain(a.option_proxy, tenor)
                cleaned, info = clean_chain(raw, spot, expiry)
            except Exception as exc:  # network or parsing failure: skip, keep going
                print(f"{a.option_proxy} {tenor}d: {exc}")
                continue
            if cleaned.empty:
                print(f"{a.option_proxy} {tenor}d: no usable quotes")
                continue
            sk = skew_metrics(cleaned)
            rows.append({"date": today, "asset": a.name, "proxy": a.option_proxy, "tenor_target": tenor,
                         "expiry": info["expiry"].date().isoformat(), "days": info["days"], "spot": info["spot"],
                         "forward": info["forward"], "stale": bool(info["stale"]), "n_quotes": len(cleaned), **sk})
            print(f"{a.option_proxy:4s} {tenor:3d}d  atm={sk['atm_iv']:.1f}  rr25={sk['rr25']:+.2f}  bf25={sk['bf25']:+.2f}")
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    new = snapshot()
    if new.empty:
        print("nothing captured")
        return
    if HISTORY.exists():
        hist = pd.read_csv(HISTORY)
        merged = pd.concat([hist, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date", "proxy", "tenor_target"], keep="last")
    else:
        merged = new
    merged.sort_values(["date", "asset", "tenor_target"]).to_csv(HISTORY, index=False)
    print(f"{len(new)} rows added, {len(merged)} total in {HISTORY}")


if __name__ == "__main__":
    main()
