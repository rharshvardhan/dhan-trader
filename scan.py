"""
Scan MANY NSE stocks at once for the most recent trading day's Range Breakout
setup (T1 / T2 / NO TRADE).

Type several symbols separated by commas (e.g. RELIANCE, TCS, INFY). For each,
this resolves the symbol -> Dhan security_id via the scrip master, pulls a small
window of intraday candles, takes the LATEST trading day present, runs the T1/T2
engine (strategy/range_breakout.py) on that day, and prints one row per stock:

    SETUP (T1/T2/NO), DIRECTION, ENTRY price, and the marked range MH / ML
    (range high / low) with its height in points.

Detection only - no orders, no SL/target. Needs the Data plan + .env creds.
Self-contained: depends only on config.py + strategy/ (no other script).

Run:  .venv/Scripts/python.exe scan.py
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

# The engine logs a verbose line per candle; silence it so our table stays clean.
logging.getLogger("dhan-trader.strategy").setLevel(logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
SCRIP_MASTER_FILE = BASE_DIR / "scrip_master.csv"
SCRIP_MASTER_MAX_AGE_DAYS = 7
LOOKBACK_DAYS = 7        # small window; we only use the latest trading day in it
INTERVAL_MIN = 5
REQUEST_GAP_SEC = 0.3    # gentle spacing between per-stock history calls

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
# Candle resampling (raw OHLC rows -> Candle objects)
# ---------------------------------------------------------------------------
def _floor(dt: datetime, interval_min: int) -> datetime:
    return dt.replace(minute=(dt.minute // interval_min) * interval_min,
                      second=0, microsecond=0)


def resample(rows, interval_min: int):
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


# ---------------------------------------------------------------------------
# Dhan Historical Data API
# ---------------------------------------------------------------------------
def _dhan_client():
    load_dotenv(BASE_DIR / ".env")
    cid = os.getenv("DHAN_CLIENT_ID", "").strip()
    tok = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
    if not (cid and tok):
        sys.exit("Missing DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN in .env")
    return dhanhq(DhanContext(cid, tok))


def load_dhan(dhan, inst, from_date: str, to_date: str, interval: int,
              instrument_type: str):
    seg_str = SEG_INT_TO_STR.get(inst.segment)
    if seg_str is None:
        raise ValueError(f"{inst.name}: unsupported segment {inst.segment} for historical.")
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


def load_scrip_master():
    """Load NSE scrip master into two indexes.

    Returns (eq_mapping, fut_map):
      eq_mapping: {SYMBOL (upper): security_id} for NSE cash equities.
      fut_map:    {UNDERLYING (upper): [(expiry_date, TRADING_SYMBOL_upper,
                   security_id, instrument_name), ...]} sorted by expiry ascending,
                   for NSE FUTSTK + FUTIDX contracts.
    """
    if _cache_fresh():
        text = SCRIP_MASTER_FILE.read_text(encoding="utf-8")
    else:
        print("Downloading Dhan scrip master (one-time, ~a few MB)...")
        resp = requests.get(SCRIP_MASTER_URL, timeout=(30, 300))
        resp.raise_for_status()
        text = resp.text
        SCRIP_MASTER_FILE.write_text(text, encoding="utf-8", newline="")

    eq_mapping = {}
    fut_map = defaultdict(list)
    reader = csvmod.DictReader(io.StringIO(text))
    for row in reader:
        if row.get("SEM_EXM_EXCH_ID") != "NSE":
            continue
        iname = row.get("SEM_INSTRUMENT_NAME") or ""
        series = row.get("SEM_SERIES") or ""
        tsym = (row.get("SEM_TRADING_SYMBOL") or "").strip().upper()
        sid = (row.get("SEM_SMST_SECURITY_ID") or "").strip()
        if not tsym or not sid:
            continue
        if iname == "EQUITY" or series == "EQ":
            if tsym not in eq_mapping:
                eq_mapping[tsym] = sid
        elif iname in ("FUTSTK", "FUTIDX"):
            parts = tsym.split("-")
            if len(parts) != 3 or parts[-1] != "FUT":
                continue
            underlying = parts[0]
            exp_str = (row.get("SEM_EXPIRY_DATE") or "").strip()
            try:
                exp = datetime.strptime(exp_str.split(" ")[0], "%Y-%m-%d").date()
            except ValueError:
                continue
            fut_map[underlying].append((exp, tsym, sid, iname))
    for u in fut_map:
        fut_map[u].sort()
    return eq_mapping, fut_map


def resolve(symbol: str, eq_mapping: dict, fut_map: dict):
    """Return (match, suggestions).

    match: dict with keys {symbol, security_id, segment, instrument_type} on hit,
           or None on miss.
    suggestions: list of candidate symbol strings (empty when match is not None).

    Accepted inputs:
      RELIANCE                 -> NSE cash equity
      AXISBANK-FUT             -> nearest not-yet-expired stock/index future
      NIFTY-JUL2026-FUT        -> explicit-expiry future (case-insensitive)
    """
    key = symbol.strip().upper()

    # Explicit-expiry future: UNDERLYING-MonYYYY-FUT (two dashes, ends with -FUT)
    if key.endswith("-FUT") and key.count("-") == 2:
        for _underlying, contracts in fut_map.items():
            for _exp, tsym, sid, iname in contracts:
                if tsym == key:
                    return ({"symbol": tsym, "security_id": sid,
                             "segment": MarketFeed.NSE_FNO,
                             "instrument_type": iname}, [])
        underlying = key.split("-")[0]
        hits = [tsym for contracts in fut_map.values()
                for _e, tsym, _s, _i in contracts if underlying in tsym][:8]
        return None, hits

    # Nearest-month future shortcut: UNDERLYING-FUT
    if key.endswith("-FUT"):
        underlying = key[:-len("-FUT")]
        if underlying in fut_map:
            today = datetime.now(IST).date()
            picks = [c for c in fut_map[underlying] if c[0] >= today] \
                    or fut_map[underlying]
            exp, tsym, sid, iname = picks[0]
            return ({"symbol": tsym, "security_id": sid,
                     "segment": MarketFeed.NSE_FNO,
                     "instrument_type": iname}, [])
        hits = [f"{u}-FUT" for u in fut_map if underlying in u][:8]
        return None, hits

    # Cash equity
    if key in eq_mapping:
        return ({"symbol": key, "security_id": eq_mapping[key],
                 "segment": MarketFeed.NSE,
                 "instrument_type": "EQUITY"}, [])

    hits = [s for s in eq_mapping if key in s][:5]
    hits += [f"{u}-FUT" for u in fut_map if key in u][:3]
    return None, hits[:8]


# ---------------------------------------------------------------------------
# Per-stock scan of the latest trading day
# ---------------------------------------------------------------------------
def scan_latest(symbol: str, security_id: str, segment: int,
                instrument_type: str, dhan):
    """Return a result dict for the latest trading day, or None if no data."""
    inst = Instrument(symbol, security_id, segment)
    to_d = date.today()
    from_d = to_d - timedelta(days=LOOKBACK_DAYS)
    rows = load_dhan(dhan, inst,
                     f"{from_d.isoformat()} 09:15:00",
                     f"{to_d.isoformat()} 15:30:00", INTERVAL_MIN,
                     instrument_type)
    if not rows:
        return None

    candles = resample(rows, INTERVAL_MIN)
    by_day = defaultdict(list)
    for cd in candles:
        by_day[cd.start.date()].append(cd)
    if not by_day:
        return None

    day = max(by_day)
    strat = RangeBreakout(symbol, interval_min=INTERVAL_MIN)
    for cd in by_day[day]:
        strat.on_candle(cd)

    if strat.setup and strat.entry_price is not None:
        verdict = strat.setup
    elif strat.setup and strat.state == State.AWAIT_ENTRY:
        verdict = strat.setup + "*"          # breakout seen, entry candle not formed yet
    elif strat.mh is None:
        verdict = "PENDING"                  # range not marked yet (before ~09:55)
    else:
        verdict = "NO TRADE"

    return {
        "day": day,
        "verdict": verdict,
        "direction": strat.direction or "",
        "entry": strat.entry_price,
        "mh": strat.mh,
        "ml": strat.ml,
    }


def main() -> int:
    try:
        eq_mapping, fut_map = load_scrip_master()
    except Exception as exc:
        print(f"Could not load scrip master: {exc}")
        return 1
    fut_contracts = sum(len(v) for v in fut_map.values())
    print(f"Loaded {len(eq_mapping)} NSE equity symbols "
          f"+ {fut_contracts} F&O futures across {len(fut_map)} underlyings.\n")
    print("Input formats: RELIANCE (cash), AXISBANK-FUT (nearest expiry), "
          "NIFTY-Jul2026-FUT (explicit).")

    raw = input("Enter symbols, comma-separated: ").strip()
    if not raw:
        return 0

    # Parse, upper-case, de-dupe while preserving order.
    seen, symbols = set(), []
    for part in raw.split(","):
        s = part.strip().upper()
        if s and s not in seen:
            seen.add(s)
            symbols.append(s)
    if not symbols:
        print("No symbols entered.")
        return 1

    dhan = _dhan_client()
    rows_out = []
    for i, sym in enumerate(symbols):
        match, hints = resolve(sym, eq_mapping, fut_map)
        if match is None:
            hint = f" (did you mean: {', '.join(hints)})" if hints else ""
            rows_out.append((sym, None, f"NOT FOUND{hint}"))
            continue
        matched = match["symbol"]
        try:
            res = scan_latest(matched, match["security_id"],
                              match["segment"], match["instrument_type"], dhan)
        except Exception as exc:
            rows_out.append((matched, None, f"ERROR: {exc}"))
            continue
        rows_out.append((matched, res, None) if res else (matched, None, "NO DATA"))
        if i < len(symbols) - 1:
            time.sleep(REQUEST_GAP_SEC)

    # ---- Print the table ----
    day_seen = next((r["day"] for _, r, err in rows_out if r), None)
    print()
    print("=" * 88)
    print("RANGE BREAKOUT SCAN  "
          + (f"(latest day: {day_seen.isoformat()})" if day_seen else "(no data)"))
    print("=" * 88)
    print(f"  {'SYMBOL':<22} {'SETUP':<9} {'DIRECTION':<13} {'ENTRY':>9} "
          f"{'MH':>9} {'ML':>9} {'RANGE':>8}")
    print("  " + "-" * 84)
    counts = defaultdict(int)
    for sym, res, err in rows_out:
        if err:
            print(f"  {sym:<22} {err}")
            continue
        entry = f"{res['entry']:.2f}" if res["entry"] is not None else "-"
        mh = f"{res['mh']:.2f}" if res["mh"] is not None else "-"
        ml = f"{res['ml']:.2f}" if res["ml"] is not None else "-"
        rng = (f"{res['mh'] - res['ml']:.2f}"
               if res["mh"] is not None and res["ml"] is not None else "-")
        counts[res["verdict"].rstrip("*")] += 1
        print(f"  {sym:<22} {res['verdict']:<9} {res['direction']:<13} {entry:>9} "
              f"{mh:>9} {ml:>9} {rng:>8}")
    print("  " + "-" * 84)
    print("  totals: "
          + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"))
    print("  (* = breakout confirmed but entry candle not formed yet; "
          "RANGE = MH-ML in points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
