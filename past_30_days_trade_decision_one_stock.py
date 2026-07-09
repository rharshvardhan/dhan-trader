"""
Interactive per-day Range Breakout report for a single NSE stock.

Type a stock symbol (e.g. RELIANCE); this resolves it to a Dhan security_id via
the scrip master, pulls the last ~30 days of intraday candles through the Dhan
Historical Data API, runs the T1/T2 engine (strategy/range_breakout.py) day by
day, and prints for each trading day whether it was a T1 day, a T2 day, or
NO TRADE.

Needs the Rs.499 Data plan and .env creds (DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN).

Run:  .venv/Scripts/python.exe analyze.py
"""

import csv as csvmod
import io
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Corporate TLS fix (Windows cert store) - before any SSL/dhanhq/requests use.
import truststore
truststore.inject_into_ssl()

import requests
from dotenv import load_dotenv
from dhanhq import DhanContext, MarketFeed, dhanhq

from config import Instrument
from strategy.candle import Candle, IST
from strategy.range_breakout import RangeBreakout, State

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("analyze")

BASE_DIR = Path(__file__).resolve().parent
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
SCRIP_MASTER_FILE = BASE_DIR / "scrip_master.csv"
SCRIP_MASTER_MAX_AGE_DAYS = 7      # re-download if the cached copy is older
LOOKBACK_DAYS = 30
INTERVAL_MIN = 5

# Dhan Historical API uses STRING exchange segments and an instrument type.
SEG_INT_TO_STR = {
    MarketFeed.IDX: "IDX_I",
    MarketFeed.NSE: "NSE_EQ",
    MarketFeed.NSE_FNO: "NSE_FNO",
    MarketFeed.NSE_CURR: "NSE_CURRENCY",
    MarketFeed.BSE: "BSE_EQ",
    MarketFeed.BSE_FNO: "BSE_FNO",
    MarketFeed.BSE_CURR: "BSE_CURRENCY",
    MarketFeed.MCX: "MCX_COMM",
}


# ---------------------------------------------------------------------------
# Engine: resample raw OHLC rows -> Candles, run the T1/T2 engine day by day
# ---------------------------------------------------------------------------
def _floor(dt: datetime, interval_min: int) -> datetime:
    return dt.replace(minute=(dt.minute // interval_min) * interval_min,
                      second=0, microsecond=0)


def resample(rows, interval_min: int):
    """rows: iterable of (datetime, open, high, low, close). Any input interval is
    aggregated into `interval_min` candles (5-min input -> unchanged)."""
    rows = sorted(rows, key=lambda r: r[0])
    delta = timedelta(minutes=interval_min)
    out, bucket = [], None
    o = h = l = c = 0.0
    for dt, ro, rh, rl, rc in rows:
        b = _floor(dt, interval_min)
        if bucket is None:
            bucket, o, h, l, c = b, ro, rh, rl, rc
        elif b == bucket:
            h, l, c = max(h, rh), min(l, rl), rc
        else:
            out.append(Candle(bucket, bucket + delta, o, h, l, c))
            bucket, o, h, l, c = b, ro, rh, rl, rc
    if bucket is not None:
        out.append(Candle(bucket, bucket + delta, o, h, l, c))
    return out


def run_backtest(symbol: str, rows, interval_min: int):
    candles = resample(rows, interval_min)
    if not candles:
        log.warning("%s: no candles to backtest.", symbol)
        return

    by_day = defaultdict(list)
    for cd in candles:
        by_day[cd.start.date()].append(cd)

    results = []
    for day in sorted(by_day):
        strat = RangeBreakout(symbol, interval_min=interval_min)
        for cd in by_day[day]:
            strat.on_candle(cd)
        if strat.setup and strat.entry_price is not None:
            results.append((day, strat.setup, strat.direction, strat.entry_price))
        elif strat.setup and strat.state in (State.AWAIT_ENTRY, State.AWAIT_T2_ENTRY):
            results.append((day, strat.setup + " (signal, no entry candle in data)",
                            strat.direction, None))
        else:
            results.append((day, "NO TRADE", "", None))

    # Summary
    log.info("")
    log.info("=" * 72)
    log.info("SUMMARY  %s  (%d trading day(s))", symbol, len(results))
    log.info("=" * 72)
    counts = defaultdict(int)
    for day, setup, direction, entry in results:
        tag = setup.split()[0]
        counts[tag] += 1
        entry_s = f"entry={entry:.2f}" if entry is not None else ""
        log.info("  %s  %-8s %-12s %s", day, setup if " " not in setup else tag,
                 direction, entry_s)
    log.info("-" * 72)
    log.info("  totals: %s",
             ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none")


# ---------------------------------------------------------------------------
# Dhan Historical Data API (needs the Data plan)
# ---------------------------------------------------------------------------
def _dhan_client():
    load_dotenv(BASE_DIR / ".env")
    cid = os.getenv("DHAN_CLIENT_ID", "").strip()
    tok = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
    if not (cid and tok):
        sys.exit("Missing DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN in .env")
    return dhanhq(DhanContext(cid, tok))


def load_dhan(dhan, inst, from_date: str, to_date: str, interval: int):
    seg_str = SEG_INT_TO_STR.get(inst.segment)
    if seg_str is None:
        raise ValueError(f"{inst.name}: unsupported segment {inst.segment} for historical.")
    instrument_type = "INDEX" if inst.segment == MarketFeed.IDX else "EQUITY"
    resp = dhan.intraday_minute_data(inst.security_id, seg_str, instrument_type,
                                     from_date, to_date, interval=interval)
    if resp.get("status") != "success":
        raise RuntimeError(f"{inst.name}: historical fetch failed: {resp.get('remarks')}")
    d = resp.get("data") or {}
    ts = d.get("timestamp") or d.get("start_Time") or []
    o, h, l, c = d.get("open", []), d.get("high", []), d.get("low", []), d.get("close", [])
    rows = []
    for i in range(len(ts)):
        dt = datetime.fromtimestamp(int(ts[i]), tz=timezone.utc).astimezone(IST)
        rows.append((dt, float(o[i]), float(h[i]), float(l[i]), float(c[i])))
    return rows


# ---------------------------------------------------------------------------
# Scrip master: resolve a typed NSE symbol -> security_id
# ---------------------------------------------------------------------------
def _cache_fresh() -> bool:
    if not SCRIP_MASTER_FILE.exists():
        return False
    age = time.time() - SCRIP_MASTER_FILE.stat().st_mtime
    return age < SCRIP_MASTER_MAX_AGE_DAYS * 86400


def load_scrip_master() -> dict:
    """Return {SYMBOL (upper): security_id} for NSE cash equities."""
    if _cache_fresh():
        text = SCRIP_MASTER_FILE.read_text(encoding="utf-8")
    else:
        print("Downloading Dhan scrip master (one-time, ~a few MB)...")
        resp = requests.get(SCRIP_MASTER_URL, timeout=(30, 300))
        resp.raise_for_status()
        text = resp.text
        SCRIP_MASTER_FILE.write_text(text, encoding="utf-8", newline="")

    mapping = {}
    reader = csvmod.DictReader(io.StringIO(text))
    for row in reader:
        if row.get("SEM_EXM_EXCH_ID") != "NSE":
            continue
        if (row.get("SEM_INSTRUMENT_NAME") != "EQUITY"
                and row.get("SEM_SERIES") != "EQ"):
            continue
        sym = (row.get("SEM_TRADING_SYMBOL") or "").strip().upper()
        sid = (row.get("SEM_SMST_SECURITY_ID") or "").strip()
        if sym and sid and sym not in mapping:
            mapping[sym] = sid
    return mapping


def resolve(symbol: str, mapping: dict):
    """Return (matched_symbol, security_id) or (None, suggestions)."""
    key = symbol.strip().upper()
    if key in mapping:
        return key, mapping[key]
    hits = [s for s in mapping if key in s][:8]
    return None, hits


def main() -> int:
    try:
        mapping = load_scrip_master()
    except Exception as exc:
        print(f"Could not load scrip master: {exc}")
        return 1
    print(f"Loaded {len(mapping)} NSE equity symbols.\n")

    raw = input("Enter NSE stock symbol (e.g. RELIANCE), or blank to quit: ").strip()
    if not raw:
        return 0

    matched, result = resolve(raw, mapping)
    if matched is None:
        if result:
            print(f"'{raw}' not found. Did you mean: {', '.join(result)}")
        else:
            print(f"'{raw}' not found in the NSE equity list.")
        return 1

    security_id = result
    print(f"\n{matched} -> security_id {security_id} (NSE_EQ)")

    to_d = date.today()
    from_d = to_d - timedelta(days=LOOKBACK_DAYS)
    from_dt = f"{from_d.isoformat()} 09:15:00"
    to_dt = f"{to_d.isoformat()} 15:30:00"
    print(f"Fetching {LOOKBACK_DAYS} days of {INTERVAL_MIN}-min candles "
          f"({from_d} -> {to_d})...\n")

    inst = Instrument(matched, security_id, MarketFeed.NSE)
    dhan = _dhan_client()
    try:
        rows = load_dhan(dhan, inst, from_dt, to_dt, INTERVAL_MIN)
    except Exception as exc:
        print(f"Historical fetch failed: {exc}")
        return 1

    if not rows:
        print("No candles returned (market data plan active? symbol traded in range?).")
        return 1

    run_backtest(matched, rows, INTERVAL_MIN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
