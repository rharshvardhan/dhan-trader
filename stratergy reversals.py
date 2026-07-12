"""
Strategy: Reversals - Range High Validation with Four Sequential Checks.

WHAT IT DOES (first-breakout-wins, NOT highest-high-wins)
---------------------------------------------------------
The user types one or MANY future names (e.g. RELIANCE-FUT, AXISBANK-FUT). For
each future we run this strategy on TWO legs:

    1. the FUTURE itself, and
    2. its underlying cash STOCK (found via scan.counterpart_of).

Both legs are fed the SAME logic. A trade is only signalled when BOTH the future
AND its stock produce a breakout in this strategy (see TRADE PLAN below).

THE LOGIC (per leg, on the latest trading day)
----------------------------------------------
Step 1 - Build the initial range from 09:15 -> 10:25 (5-min candles):
             RangeHigh = highest high, RangeLow = lowest low.
         RangeHigh NEVER changes during the checks.

Step 2 - Perform exactly FOUR validation checks on the candles at:
             10:30, 10:35, 10:40, 10:45.
         For each check, read CurrentHigh = candle.high and compare it against
         the ORIGINAL RangeHigh only:
             - the FIRST check whose high is STRICTLY greater than RangeHigh is
               the valid breakout -> record BreakoutHigh, STOP checking.
             - all remaining scheduled checks are then IGNORED (even if a later
               check would print an even higher high - first break wins).
             - if none of the four exceed RangeHigh, no breakout; RangeHigh
               remains the reference.

Step 2.6 - Consolidation (accumulation) filter, monitored alongside Step 3. The
         validation box is [LowestValidationLow, HighestValidationHigh]. Starting
         after Check 4, count CONSECUTIVE candles that stay entirely inside the
         box (high <= boundary AND low >= boundary). Any boundary break stops the
         count. If the count reaches CONSOLIDATION_THRESHOLD (6) before a break,
         REJECT the setup (price never made a decisive move) - no reversal, no
         trade. (An inside-box candle is exactly one where Step 3 stays WAITING.)

Step 3 - Reversal confirmation (two-boundary state machine). From the FOUR
         validation candles compute two FIXED boundaries:
             HighestValidationHigh = max high of 10:30/35/40/45
             LowestValidationLow   = min low  of 10:30/35/40/45
         Then monitor EVERY candle after Check 4 (from 10:50 onward), in order.
         First event wins:
             - a candle's LOW < LowestValidationLow first  -> REVERSAL CONFIRMED
               (price rejected the breakout and reversed down); stop.
             - a candle's HIGH > HighestValidationHigh first -> REVERSAL
               INVALIDATED (breakout resumed up); stop, setup cancelled.
             - if a single candle breaks both, the LOW (confirmation) wins.
             - if neither happens by end of day -> still WAITING (no trade).

A tradeable signal needs BOTH legs (future AND stock) to break in the four
checks AND end in REVERSAL CONFIRMED. The confirmed move is down -> SELL/short.

This mirrors the fetch/resolve/resample plumbing used by scan.py and
past_30_days_trade_decision_one_stock.py, but the range window (09:15-10:25) and
the four-check validation are UNIQUE to this strategy and are implemented here.

Run:  .venv/Scripts/python.exe "stratergy reversals.py"

NOTE: this is the first cut of the spec file. Entry / stop-loss / exit levels in
the TRADE PLAN section are provisional and marked TODO - we refine them next.
"""

import logging
import sys
import time as time_mod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import List, Optional

# Reuse the symbol-resolution + candle-fetch plumbing from scan.py so this file
# only owns the NEW strategy logic (range window + four-check validation).
# scan.py guards its CLI behind `if __name__ == "__main__"`, so importing it here
# runs no interactive code.
from scan import (
    REQUEST_GAP_SEC,
    counterpart_of,
    load_dhan,
    load_scrip_master,
    resample,
    resolve,
    _dhan_client,
)
from strategy.candle import Candle

logging.getLogger("dhan-trader.strategy").setLevel(logging.WARNING)

INTERVAL_MIN = 5
LOOKBACK_DAYS = 7            # small window; we only use the latest trading day in it

# --- Strategy windows (5-min candles, IST) ---------------------------------
# Range is built from candles whose START is in [09:15, 10:25); i.e. the last
# range candle is 10:20-10:25. The 10:25-10:30 candle is deliberately NOT part
# of the range and NOT a check (the spec jumps straight to 10:30).
RANGE_START = time(9, 15)
RANGE_END   = time(10, 25)

# Exactly four post-range validation checks, in order. First break wins.
CHECK_TIMES: List[time] = [time(10, 30), time(10, 35), time(10, 40), time(10, 45)]

# Consolidation (accumulation) filter: reject the setup if this many consecutive
# candles after Check 4 stay entirely inside the validation box (no decisive move).
CONSOLIDATION_THRESHOLD = 6


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ReversalResult:
    """Outcome of the four-check range validation for ONE leg on ONE day."""
    symbol: str
    day: Optional[date] = None
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    breakout_found: bool = False
    breakout_high: Optional[float] = None      # == range_high when no breakout
    breakout_check: Optional[time] = None      # which check first broke, or None
    # per-check log: list of (check_time, high_or_None, broke_bool, ignored_bool)
    checks: List[tuple] = field(default_factory=list)
    note: str = ""                             # e.g. "NO DATA", "NO RANGE CANDLES"

    # --- Reversal state machine (boundaries from the four validation candles) ---
    highest_validation_high: Optional[float] = None   # fixed upper boundary
    lowest_validation_low: Optional[float] = None      # fixed lower boundary
    reversal_state: str = "WAITING"            # WAITING | CONFIRMED | INVALIDATED
    reversal_label: Optional[str] = None       # candle that first broke a boundary
    reversal_price: Optional[float] = None     # its low (confirm) or high (invalidate)

    # --- Consolidation (accumulation) filter ---
    inside_box_count: int = 0                  # consecutive candles inside the box
    rejected: bool = False                     # True -> setup rejected by a filter
    reject_reason: str = ""                    # why it was rejected

    @property
    def ok(self) -> bool:
        """True when the range was built AND a breakout was found."""
        return self.range_high is not None and self.breakout_found

    @property
    def reversal_confirmed(self) -> bool:
        return self.reversal_state == "CONFIRMED"

    @property
    def qualifies(self) -> bool:
        """Tradeable reversal on this leg: a breakout that then reversed down,
        not rejected by a filter."""
        return self.ok and not self.rejected and self.reversal_confirmed


# ---------------------------------------------------------------------------
# Core strategy logic (the part unique to "reversals")
# ---------------------------------------------------------------------------
def evaluate_reversal(symbol: str, candles: List[Candle],
                      day: Optional[date] = None) -> ReversalResult:
    """Run the range-high four-check validation on ONE day's candles.

    `candles` must be the 5-min candles for a single trading day, in order.
    Implements: build RangeHigh/RangeLow from 09:15-10:25, then check 10:30 /
    10:35 / 10:40 / 10:45 in order; the FIRST check whose high > RangeHigh wins
    and stops the scan; later checks are ignored.
    """
    res = ReversalResult(symbol=symbol, day=day)

    range_candles = [c for c in candles if RANGE_START <= c.start.time() < RANGE_END]
    if not range_candles:
        res.note = "NO RANGE CANDLES (09:15-10:25)"
        return res

    res.range_high = max(c.high for c in range_candles)
    res.range_low = min(c.low for c in range_candles)
    res.breakout_high = res.range_high            # default reference until a break

    by_time = {c.start.time(): c for c in candles}

    for ct in CHECK_TIMES:
        # Once a breakout is found, every remaining check is IGNORED (first break wins).
        if res.breakout_found:
            res.checks.append((ct, by_time.get(ct).high if ct in by_time else None,
                               False, True))
            continue

        c = by_time.get(ct)
        if c is None:                             # candle missing (e.g. no trade printed)
            res.checks.append((ct, None, False, False))
            continue

        broke = c.high > res.range_high           # STRICTLY greater = valid break
        res.checks.append((ct, c.high, broke, False))
        if broke:
            res.breakout_found = True
            res.breakout_high = c.high
            res.breakout_check = ct

    # --- Reversal state machine (boundaries fixed from the four validation
    # candles; then monitor every candle AFTER Check 4, first event wins) ----
    last_check = CHECK_TIMES[-1]
    validation_candles = [by_time[ct] for ct in CHECK_TIMES if ct in by_time]
    if validation_candles:
        res.highest_validation_high = max(vc.high for vc in validation_candles)
        res.lowest_validation_low = min(vc.low for vc in validation_candles)
        monitor = sorted((mc for mc in candles if mc.start.time() > last_check),
                         key=lambda mc: mc.start)
        for mc in monitor:
            # Decision-flow order: check the LOW break (confirmation) FIRST, so a
            # candle that breaks both boundaries counts as a confirmation.
            if mc.low < res.lowest_validation_low:
                res.reversal_state = "CONFIRMED"
                res.reversal_label = mc.label()
                res.reversal_price = mc.low
                break
            if mc.high > res.highest_validation_high:
                res.reversal_state = "INVALIDATED"
                res.reversal_label = mc.label()
                res.reversal_price = mc.high
                break

            # Neither boundary broke -> candle sits entirely inside the box.
            # Consolidation filter: too many consecutive inside-box candles means
            # price never made a decisive move -> reject (accumulation).
            res.inside_box_count += 1
            if res.inside_box_count >= CONSOLIDATION_THRESHOLD:
                res.rejected = True
                res.reject_reason = (
                    f"Validation Box Consolidation (Accumulation) - "
                    f"{res.inside_box_count} candles inside "
                    f"[{res.lowest_validation_low:.2f}, "
                    f"{res.highest_validation_high:.2f}]")
                break

    return res


# ---------------------------------------------------------------------------
# Candle fetch for the latest trading day (reuses scan.py load_dhan/resample)
# ---------------------------------------------------------------------------
def latest_day_candles(dhan, symbol: str, security_id: str, segment: int,
                       instrument_type: str):
    """Fetch a small window and return (day, candles) for the LATEST day, or (None, [])."""
    from config import Instrument                # local import: cheap, avoids cycle noise
    inst = Instrument(symbol, security_id, segment)
    to_d = date.today()
    from_d = to_d - timedelta(days=LOOKBACK_DAYS)
    rows = load_dhan(dhan, inst,
                     f"{from_d.isoformat()} 09:15:00",
                     f"{to_d.isoformat()} 15:30:00", INTERVAL_MIN, instrument_type)
    if not rows:
        return None, []
    candles = resample(rows, INTERVAL_MIN)
    by_day = defaultdict(list)
    for cd in candles:
        by_day[cd.start.date()].append(cd)
    if not by_day:
        return None, []
    day = max(by_day)
    return day, by_day[day]


def run_leg(dhan, symbol: str, security_id: str, segment: int,
            instrument_type: str) -> ReversalResult:
    """Fetch the latest day for one leg and evaluate the reversal strategy on it."""
    day, candles = latest_day_candles(dhan, symbol, security_id, segment, instrument_type)
    if not candles:
        r = ReversalResult(symbol=symbol, day=day)
        r.note = "NO DATA"
        return r
    return evaluate_reversal(symbol, candles, day)


# ---------------------------------------------------------------------------
# TRADE PLAN -- PROVISIONAL, refine next
#   A tradeable REVERSAL needs, on BOTH the future AND its stock:
#     1. a four-check breakout above RangeHigh, AND
#     2. the reversal state machine ending in CONFIRMED (low broke before high).
#   The confirmed move is a downside reversal -> the trade is a SELL/short.
# ---------------------------------------------------------------------------
@dataclass
class TradePlan:
    signal: bool = False
    direction: str = ""            # "SELL (reversal short)" on a confirmed reversal
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    exit_target: Optional[float] = None
    reason: str = ""


def _leg_fail_reason(leg: str, res: ReversalResult) -> Optional[str]:
    """Why this leg is not a qualifying reversal, or None if it qualifies."""
    if res.note:
        return f"{leg} {res.note}"
    if res.rejected:
        return f"{leg} REJECTED - {res.reject_reason}"
    if not res.breakout_found:
        return f"{leg} no-break (range held)"
    if res.reversal_state == "INVALIDATED":
        return (f"{leg} reversal invalidated ({res.reversal_label}: "
                f"high {res.reversal_price:.2f} > HVH {res.highest_validation_high:.2f})")
    if res.reversal_state != "CONFIRMED":
        return f"{leg} reversal not confirmed (still waiting by EOD)"
    return None


def build_trade_plan(fut: ReversalResult, stk: ReversalResult) -> TradePlan:
    """Combine the two legs into a reversal trade decision.

    Rule (v1): signal only when BOTH legs qualify (breakout + CONFIRMED reversal).
    Entry/SL/exit are a FIRST DRAFT - we tighten them together next.
    """
    fails = [r for r in (_leg_fail_reason("future", fut),
                         _leg_fail_reason("stock", stk)) if r]
    if fails:
        return TradePlan(signal=False, reason="; ".join(fails))

    # Both legs broke up then reversed down -> SELL bias off the FUTURE leg.
    # TODO(next): confirm entry trigger, stop placement, and target math.
    entry = fut.lowest_validation_low               # the level the reversal candle broke
    stop_loss = fut.highest_validation_high         # provisional: the invalidation boundary
    return TradePlan(
        signal=True,
        direction="SELL (reversal short)",
        entry=entry,
        stop_loss=stop_loss,
        exit_target=None,                           # TODO(next): define exit/target
        reason=(f"future: broke {fut.breakout_high:.2f}@{fut.breakout_check.strftime('%H:%M')}, "
                f"reversal CONFIRMED {fut.reversal_label} (low {fut.reversal_price:.2f} < "
                f"LVL {fut.lowest_validation_low:.2f}); "
                f"stock: broke {stk.breakout_high:.2f}@{stk.breakout_check.strftime('%H:%M')}, "
                f"reversal CONFIRMED {stk.reversal_label} (low {stk.reversal_price:.2f} < "
                f"LVL {stk.lowest_validation_low:.2f})"),
    )


# ---------------------------------------------------------------------------
# Reporting - one row per input symbol, future + stock side by side
# ---------------------------------------------------------------------------
TABLE_W = 78


def print_table(rows: list) -> None:
    """rows: list of dicts with keys symbol, fut, stk, plan, error."""
    print()
    print("=" * TABLE_W)
    print("REVERSALS - future + stock (four-check breakout + reversal confirmation)")
    print("=" * TABLE_W)
    print(f"  {'SYMBOL':<24} {'SIGNAL':>13} {'ENTRY':>9} {'SL':>9} {'EXIT':>7}")
    print("  " + "-" * (TABLE_W - 2))

    matched = 0
    for r in rows:
        sym = r["symbol"]
        if r.get("error"):
            print(f"  {sym:<24} {r['error']}")
            continue

        plan = r["plan"]
        if plan.signal:
            matched += 1
            signal_cell = "MATCH"
            entry = f"{plan.entry:.2f}" if plan.entry is not None else "TODO"
            sl = f"{plan.stop_loss:.2f}" if plan.stop_loss is not None else "TODO"
            ex = f"{plan.exit_target:.2f}" if plan.exit_target is not None else "TODO"
        else:
            signal_cell = "DOESN'T MATCH"
            entry = sl = ex = "-"

        print(f"  {sym:<24} {signal_cell:>13} {entry:>9} {sl:>9} {ex:>7}")

    print("  " + "-" * (TABLE_W - 2))
    n_ok = len([r for r in rows if not r.get("error")])
    print(f"  reversal matches (both legs broke AND CONFIRMED): {matched}/{n_ok}")
    print("  MATCH (SELL reversal) -> ENTRY/SL/EXIT shown (provisional, to be "
          "finalized next); DOESN'T MATCH -> no trade.")


def main() -> int:
    try:
        eq_mapping, fut_map = load_scrip_master()
    except Exception as exc:
        print(f"Could not load scrip master: {exc}")
        return 1

    print("Reversals strategy - Range High four-check validation (future + stock).")
    print("Input FUTURE names, comma-separated (e.g. RELIANCE-FUT, AXISBANK-FUT).")
    raw = input("Enter future names: ").strip()
    if not raw:
        return 0

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
    rows = []
    for i, sym in enumerate(symbols):
        match, hints = resolve(sym, eq_mapping, fut_map)
        if match is None:
            hint = f" (did you mean: {', '.join(hints)})" if hints else ""
            rows.append({"symbol": sym, "error": f"NOT FOUND{hint}"})
            continue

        fut_sym = match["symbol"]

        # Future leg.
        try:
            fut_res = run_leg(dhan, fut_sym, match["security_id"],
                              match["segment"], match["instrument_type"])
        except Exception as exc:
            rows.append({"symbol": fut_sym, "error": f"FUTURE ERROR: {exc}"})
            continue

        # Stock leg (the same underlying, via scan.counterpart_of).
        cp = counterpart_of(fut_sym, match["segment"], eq_mapping, fut_map)
        if cp is None:
            rows.append({"symbol": fut_sym, "error": "STOCK counterpart not found"})
            continue
        try:
            time_mod.sleep(REQUEST_GAP_SEC)      # gentle spacing for the extra call
            stk_res = run_leg(dhan, cp[0], cp[1], cp[2], cp[3])
        except Exception as exc:
            rows.append({"symbol": fut_sym, "error": f"STOCK ERROR: {exc}"})
            continue

        plan = build_trade_plan(fut_res, stk_res)
        rows.append({"symbol": fut_sym, "fut": fut_res, "stk": stk_res, "plan": plan})

        if i < len(symbols) - 1:
            time_mod.sleep(REQUEST_GAP_SEC)

    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
