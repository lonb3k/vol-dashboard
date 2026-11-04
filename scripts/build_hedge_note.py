"""Write the weekly corporate-hedging note to sheet/hedging.json. Run after the US close.

    python scripts/build_hedge_note.py [--offline]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vol_dashboard.hedge_note import build_hedge_note  # noqa: E402

OUT = ROOT / "sheet" / "hedging.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use the cached CSV extracts only")
    args = ap.parse_args()
    note = build_hedge_note(offline=args.offline)
    OUT.write_text(json.dumps(note, separators=(",", ":"), allow_nan=False))
    print(f"{OUT} ({OUT.stat().st_size / 1024:.0f} KB), as of {note['as_of']}, warnings: {note['warnings'] or 'none'}")
    print(note["headline"])
    for i in note["ideas"]:
        print(f"  {i['rank']}. {i['title']}  [{i['condition']}]")


if __name__ == "__main__":
    main()
