"""
Scan MANY NSE stocks at once for the most recent trading day's Range Breakout
setup (T1 / T2 / NO TRADE).

Type several symbols separated by commas (e.g. RELIANCE, TCS, INFY). For each,
this resolves the symbol -> Dhan security_id via the scrip master, pulls a small
window of intraday candles, takes the LATEST trading day present, runs the T1/T2
engine (strategy/range_breakout.py) on that day, and prints one row per stock:

    SETUP (T1/T2/NO), DIRECTION, ENTRY price, and the marked range MH / ML
    (range high / low) with its height in points.

Detection + Target 1: the engine runs on the searched leg AND its counterpart
(future<->underlying stock). Target 1 = min(both opening ranges)/2, taken ONLY
when BOTH legs are T1 in the SAME direction; otherwise the setup is HOLD. No
orders placed, no stop-loss yet. Needs the Data plan + .env creds.
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
TARGET_STEP = 5.0        # T2/T3 ladder: each extends the target by 5 more points

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
# Target 1 (needs BOTH legs):
#   Run the Range Breakout engine on the future AND on its underlying stock.
#   A target is taken ONLY when BOTH legs are T1 and point the SAME direction.
#   Then T1 distance = min(future range, stock range) / 2, applied from the
#   future's entry by direction. Any disagreement -> HOLD (no target).
# ---------------------------------------------------------------------------
def counterpart_of(symbol: str, segment: int, eq_mapping: dict, fut_map: dict):
    """Return the opposite leg used for Target 1, or None.

    future  -> its underlying cash equity.
    equity  -> its nearest not-yet-expired future.
    Returns (symbol, security_id, segment, instrument_type) or None."""
    if segment == MarketFeed.NSE_FNO:            # future -> underlying equity
        underlying = symbol.split("-")[0]
        sid = eq_mapping.get(underlying)
        return (underlying, sid, MarketFeed.NSE, "EQUITY") if sid else None
    if segment == MarketFeed.NSE:                # equity -> nearest future
        contracts = fut_map.get(symbol.upper())
        if not contracts:
            return None
        today = datetime.now(IST).date()
        picks = [c for c in contracts if c[0] >= today] or contracts
        _exp, tsym, sid, iname = picks[0]
        return (tsym, sid, MarketFeed.NSE_FNO, iname)
    return None


def _verdict(strat) -> str:
    """Map an engine result to a verdict string.

    T1 / T2 (with '*' when the breakout is confirmed but the entry candle has not
    formed yet), PENDING (range not marked), NO TRADE, or '-' when there is no data."""
    if strat is None:
        return "-"
    if strat.setup and strat.entry_price is not None:
        return strat.setup
    if strat.setup and strat.state in (State.AWAIT_ENTRY, State.AWAIT_T2_ENTRY):
        return strat.setup + "*"
    if strat.mh is None:
        return "PENDING"
    return "NO TRADE"


def run_day_strategy(dhan, symbol, security_id, segment, instrument_type):
    """Fetch the latest trading day's candles and run the engine on it.

    Returns (day, strat) or (None, None) when there is no data."""
    inst = Instrument(symbol, security_id, segment)
    to_d = date.today()
    from_d = to_d - timedelta(days=LOOKBACK_DAYS)
    rows = load_dhan(dhan, inst,
                     f"{from_d.isoformat()} 09:15:00",
                     f"{to_d.isoformat()} 15:30:00", INTERVAL_MIN, instrument_type)
    if not rows:
        return None, None
    candles = resample(rows, INTERVAL_MIN)
    by_day = defaultdict(list)
    for cd in candles:
        by_day[cd.start.date()].append(cd)
    if not by_day:
        return None, None
    day = max(by_day)
    strat = RangeBreakout(symbol, interval_min=INTERVAL_MIN)
    for cd in by_day[day]:
        strat.on_candle(cd)
    return day, strat


def _range_of(strat):
    """Opening-range height (MH-ML) of an engine result, or None."""
    if strat is None or strat.mh is None or strat.ml is None:
        return None
    return strat.mh - strat.ml


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
                instrument_type: str, dhan, eq_mapping: dict, fut_map: dict):
    """Return a result dict for the latest trading day, or None if no data.

    Runs the engine on the searched leg AND on its counterpart (future<->stock).
    A Target 1 is produced only when BOTH legs are T1 and share the same
    direction; otherwise a T1 on the searched leg is downgraded to HOLD."""
    day, strat = run_day_strategy(dhan, symbol, security_id, segment, instrument_type)
    if strat is None:
        return None

    verdict = _verdict(strat)
    own_range = _range_of(strat)

    # ---- Run the counterpart leg (future <-> underlying stock) ----
    cp_strat = None
    cp_setup, cp_range = "-", None
    cp = counterpart_of(symbol, segment, eq_mapping, fut_map)
    if cp is not None:
        try:
            time.sleep(REQUEST_GAP_SEC)      # gentle spacing for the extra call
            _cp_day, cp_strat = run_day_strategy(dhan, cp[0], cp[1], cp[2], cp[3])
            cp_setup = _verdict(cp_strat)
            cp_range = _range_of(cp_strat)
        except Exception:
            cp_strat, cp_setup, cp_range = None, "-", None   # counterpart unavailable

    # ---- Target 1 gate: BOTH legs T1 AND same direction ----
    t1_dist = t1_price = None
    if strat.setup == "T1":
        both_t1 = cp_strat is not None and cp_strat.setup == "T1"
        same_dir = (both_t1 and strat.direction
                    and cp_strat.direction == strat.direction)
        if both_t1 and same_dir and own_range is not None and cp_range is not None:
            t1_dist = min(own_range, cp_range) / 2.0
            if strat.entry_price is not None:
                if strat.direction.startswith("BUY"):
                    t1_price = strat.entry_price + t1_dist
                elif strat.direction.startswith("SELL"):
                    t1_price = strat.entry_price - t1_dist
        else:
            verdict = "HOLD"                  # T1 not confirmed by the other leg

    # ---- T2/T3 ladder: only after a valid T1, and only while price keeps
    #      running in the trade direction. Each rung adds TARGET_STEP points.
    #      If price does not push past T1, T2 is unset -> T3 unset too. ----
    t2_price = t3_price = None
    if t1_price is not None:
        if strat.direction.startswith("SELL"):
            ext = strat.post_low             # lowest low reached after entry
            if ext is not None and ext <= t1_price:      # still moving down past T1
                t2_price = t1_price - TARGET_STEP
                if ext <= t2_price:                      # still moving down past T2
                    t3_price = t2_price - TARGET_STEP
        elif strat.direction.startswith("BUY"):
            ext = strat.post_high            # highest high reached after entry
            if ext is not None and ext >= t1_price:      # still moving up past T1
                t2_price = t1_price + TARGET_STEP
                if ext >= t2_price:                      # still moving up past T2
                    t3_price = t2_price + TARGET_STEP

    return {
        "day": day,
        "verdict": verdict,
        "direction": strat.direction or "",
        "entry": strat.entry_price,
        "mh": strat.mh,
        "ml": strat.ml,
        "own_range": own_range,
        "cp_setup": cp_setup,
        "cp_range": cp_range,
        "t1_dist": t1_dist,
        "t1_price": t1_price,
        "t2_price": t2_price,
        "t3_price": t3_price,
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
                              match["segment"], match["instrument_type"], dhan,
                              eq_mapping, fut_map)
        except Exception as exc:
            rows_out.append((matched, None, f"ERROR: {exc}"))
            continue
        rows_out.append((matched, res, None) if res else (matched, None, "NO DATA"))
        if i < len(symbols) - 1:
            time.sleep(REQUEST_GAP_SEC)

    # ---- Print the table ----
    day_seen = next((r["day"] for _, r, err in rows_out if r), None)
    print()
    print("=" * 138)
    print("RANGE BREAKOUT SCAN  "
          + (f"(latest day: {day_seen.isoformat()})" if day_seen else "(no data)"))
    print("=" * 138)
    print(f"  {'SYMBOL':<22} {'SETUP':<9} {'DIRECTION':<13} {'ENTRY':>9} "
          f"{'MH':>9} {'ML':>9} {'RANGE':>8} {'CPSET':>7} {'CPRNG':>8} "
          f"{'T1DIST':>8} {'T1':>9} {'T2':>9} {'T3':>9}")
    print("  " + "-" * 134)
    counts = defaultdict(int)
    for sym, res, err in rows_out:
        if err:
            print(f"  {sym:<22} {err}")
            continue
        entry = f"{res['entry']:.2f}" if res["entry"] is not None else "-"
        mh = f"{res['mh']:.2f}" if res["mh"] is not None else "-"
        ml = f"{res['ml']:.2f}" if res["ml"] is not None else "-"
        rng = f"{res['own_range']:.2f}" if res["own_range"] is not None else "-"
        cpset = res.get("cp_setup") or "-"
        cprng = f"{res['cp_range']:.2f}" if res["cp_range"] is not None else "-"
        t1d = f"{res['t1_dist']:.2f}" if res["t1_dist"] is not None else "-"
        t1 = f"{res['t1_price']:.2f}" if res["t1_price"] is not None else "-"
        t2 = f"{res['t2_price']:.2f}" if res.get("t2_price") is not None else "-"
        t3 = f"{res['t3_price']:.2f}" if res.get("t3_price") is not None else "-"
        counts[res["verdict"].rstrip("*")] += 1
        print(f"  {sym:<22} {res['verdict']:<9} {res['direction']:<13} {entry:>9} "
              f"{mh:>9} {ml:>9} {rng:>8} {cpset:>7} {cprng:>8} {t1d:>8} {t1:>9} "
              f"{t2:>9} {t3:>9}")
    print("  " + "-" * 134)
    print("  totals: "
          + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"))
    print("  (* = breakout confirmed but entry candle not formed yet; "
          "RANGE = own MH-ML)")
    print("  (CPSET = counterpart leg's setup; CPRNG = its range. T1 only when "
          "BOTH legs are T1 + same direction, else HOLD.)")
    print("  (T1DIST = min(RANGE,CPRNG)/2; T1 = entry +/- T1DIST. T2=T1+/-5, "
          "T3=T2+/-5, each set only if price kept running past the prior target.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
