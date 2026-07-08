"""
Range Breakout - unified T1 + T2 detection (logging only; no orders, no risk mgmt).

Implements specs/t1-range-breakout.md and specs/t2-range-breakout.md. T1 and T2 are
two outcomes of ONE marked range, split by WHEN the breakout happens:

  * Mark range 09:15-09:55  -> MH (max high), ML (min low).
  * 09:55-10:20 monitoring:
      - a candle CLOSES outside [ML, MH]  -> T2 (early breakout); entry = next candle.
      - stays inside                      -> keep monitoring (T1 still possible).
  * the 10:20-10:25 candle, only if still inside at 10:20 (decided at its 10:25 close):
      - CLOSES outside [ML, MH]           -> T1 (delayed breakout); entry = next candle.
      - inside                            -> no trade.

Breakouts are CLOSE-confirmed: a wick that closes back inside does not count and does
NOT invalidate. One setup per day; T1 and T2 are mutually exclusive. Every log line is
prefixed with the instrument symbol so multi-stock runs stay readable.
"""

import logging
from datetime import time
from enum import Enum, auto

from .candle import Candle

log = logging.getLogger("dhan-trader.strategy")

# 5-min candle boundaries (IST). See specs - recompute if the interval changes.
MARKING_START      = time(9, 15)
MARKING_END        = time(9, 55)    # marking candles: start in [09:15, 09:55)
MONITORING_END     = time(10, 20)   # T2 window: candles with start in [09:55, 10:20)
T1_BREAKOUT_CANDLE = time(10, 20)   # the 10:20-10:25 candle (decided at its 10:25 close)


class State(Enum):
    WAITING     = auto()
    MARKING     = auto()
    MONITORING  = auto()
    AWAIT_ENTRY = auto()
    DONE        = auto()            # entered / no-trade (idle until next session)


class RangeBreakout:
    def __init__(self, symbol: str, interval_min: int = 5):
        self.symbol = symbol
        self.interval_min = interval_min
        self._reset(None)

    def _say(self, msg: str, *args) -> None:
        """Log one line, prefixed with the instrument symbol."""
        log.info("%-14s | " + msg, self.symbol, *args)

    def _reset(self, day) -> None:
        self.day = day
        self.state = State.WAITING
        self.mh = None
        self.ml = None
        self.setup = None            # "T1" or "T2"
        self.direction = None        # "BUY (long)" / "SELL (short)"
        self.entry_price = None      # T1/T2 entry candle open (for summaries)
        self.entry_label = None      # T1/T2 entry candle label
        self._mark_high = None
        self._mark_low = None

    def on_candle(self, c: Candle) -> None:
        """Feed one COMPLETED candle, in chronological order."""
        day = c.start.date()
        if day != self.day:                       # new session -> re-arm
            self._reset(day)
            self._say("-------- %s | strategy armed --------", day)

        # A breakout was confirmed last candle -> THIS candle (any time) is the entry.
        if self.state == State.AWAIT_ENTRY:
            self.entry_price = c.open
            self.entry_label = c.label()
            self._say("[%s ENTRY OK]    %s  %s @ entry open=%.2f  "
                      "(detection only - no order, no SL/target yet)",
                      self.setup, c.label(), self.direction, c.open)
            self.state = State.DONE
            return

        if self.state == State.DONE:
            return

        t = c.start.time()
        if t < MARKING_START:
            return                                # pre-open noise

        # Phase 1 - mark the opening range (09:15-09:55)
        if t < MARKING_END:
            self.state = State.MARKING
            self._mark_high = c.high if self._mark_high is None else max(self._mark_high, c.high)
            self._mark_low  = c.low  if self._mark_low  is None else min(self._mark_low,  c.low)
            self._say("[MARKING]       %s  H=%.2f L=%.2f  | running MH=%.2f ML=%.2f",
                      c.label(), c.high, c.low, self._mark_high, self._mark_low)
            return

        # First candle at/after 09:55 -> finalize the marked range
        if self.mh is None:
            self._finalize_range()
            if self.state == State.DONE:
                return

        # Phase 2 - T2 window: early-breakout monitoring (09:55-10:20)
        if t < MONITORING_END:
            self._check_monitoring(c)
            return

        # Phase 3 - T1 breakout check on the 10:20-10:25 candle (at its 10:25 close)
        if t == T1_BREAKOUT_CANDLE:
            self._check_t1_breakout(c)
            return

        # Later candles with no setup armed: nothing to do today.

    # -- helpers --------------------------------------------------------------
    def _finalize_range(self) -> None:
        self.mh, self.ml = self._mark_high, self._mark_low
        if self.mh is None:
            self._say("[RANGE]         no marking candles received - cannot arm setup")
            self.state = State.DONE
            return
        self.state = State.MONITORING
        self._say("[RANGE MARKED]  09:15-09:55  MH=%.2f  ML=%.2f  (height=%.2f)",
                  self.mh, self.ml, self.mh - self.ml)

    def _arm_entry(self, setup: str, direction: str) -> None:
        self.setup, self.direction, self.state = setup, direction, State.AWAIT_ENTRY

    def _check_monitoring(self, c: Candle) -> None:
        """09:55-10:20: an early CLOSE outside the range is a T2 breakout."""
        if c.close > self.mh:
            self._arm_entry("T2", "BUY (long)")
            self._say("[T2 BREAKOUT OK] %s  close=%.2f > MH=%.2f  -> BULLISH early breakout "
                      "(close-confirmed); next candle = T2 entry", c.label(), c.close, self.mh)
        elif c.close < self.ml:
            self._arm_entry("T2", "SELL (short)")
            self._say("[T2 BREAKOUT OK] %s  close=%.2f < ML=%.2f  -> BEARISH early breakout "
                      "(close-confirmed); next candle = T2 entry", c.label(), c.close, self.ml)
        else:
            wick = ""
            if c.high > self.mh or c.low < self.ml:
                wick = " (wick pierced, close inside - still in range)"
            self._say("[CONSOLIDATE OK] %s  close=%.2f inside [%.2f, %.2f]%s",
                      c.label(), c.close, self.ml, self.mh, wick)

    def _check_t1_breakout(self, c: Candle) -> None:
        """10:20-10:25 candle (only reached if price held inside through 10:20)."""
        if c.close > self.mh:
            self._arm_entry("T1", "BUY (long)")
            self._say("[T1 BREAKOUT OK] %s  close=%.2f > MH=%.2f  -> BULLISH (close-confirmed); "
                      "next candle = T1 entry", c.label(), c.close, self.mh)
        elif c.close < self.ml:
            self._arm_entry("T1", "SELL (short)")
            self._say("[T1 BREAKOUT OK] %s  close=%.2f < ML=%.2f  -> BEARISH (close-confirmed); "
                      "next candle = T1 entry", c.label(), c.close, self.ml)
        else:
            wick = ""
            if c.high > self.mh or c.low < self.ml:
                wick = " (wick pierced but close back inside - does NOT count)"
            self._say("[NO BREAKOUT]   %s  close=%.2f within [%.2f, %.2f]%s -> no trade today",
                      c.label(), c.close, self.ml, self.mh, wick)
            self.state = State.DONE
