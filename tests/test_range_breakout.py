"""Tests for the Range Breakout engine (strategy/range_breakout.py).

Covers: opening-range marking, T2 (early) vs T1 (delayed) detection in both
directions, close-confirmation (wicks don't count), the no-gate rule (T1 can
fire at any later candle), entry timing (T1 = next candle; T2 = deferred to
11:10 using the 11:05 close), post-entry excursion tracking
(post_low / post_high used for the T2/T3 ladder), and the PENDING / NO-TRADE
end states.
"""

from datetime import date, datetime, timedelta

import pytest

from strategy.candle import Candle, IST
from strategy.range_breakout import RangeBreakout, State

DAY = date(2026, 7, 7)


def c(hh, mm, o, h, l, cl, day=DAY):
    start = datetime(day.year, day.month, day.day, hh, mm, tzinfo=IST)
    return Candle(start, start + timedelta(minutes=5), o, h, l, cl)


def feed(strat, candles):
    for cd in candles:
        strat.on_candle(cd)


# Two marking candles -> MH=110, ML=90. Reused by most scenarios.
MARKING = [
    c(9, 15, 100, 110, 95, 105),   # high 110
    c(9, 20, 105, 108, 90, 100),   # low 90
]


def new():
    return RangeBreakout("TEST")


# --------------------------------------------------------------------------- #
# Marking / range finalization
# --------------------------------------------------------------------------- #
def test_marking_sets_mh_ml_on_first_candle_after_0955():
    s = new()
    feed(s, MARKING)
    assert s.mh is None                       # not finalized until >= 09:55
    feed(s, [c(9, 55, 100, 105, 96, 100)])    # inside -> finalize + consolidate
    assert s.mh == 110
    assert s.ml == 90
    assert s.state == State.MONITORING


def test_pending_when_no_candle_after_0955():
    s = new()
    feed(s, MARKING)
    assert s.mh is None                       # range never finalized -> PENDING
    assert s.state == State.MARKING


def test_no_marking_candles_cannot_arm():
    s = new()
    feed(s, [c(9, 55, 100, 105, 96, 100)])    # first candle is already >= 09:55
    assert s.mh is None
    assert s.state == State.DONE               # no range -> done, no setup
    assert s.setup is None


# --------------------------------------------------------------------------- #
# T2 - early breakout in the 09:55-10:25 window
# --------------------------------------------------------------------------- #
def test_t2_bullish_early_breakout():
    s = new()
    feed(s, MARKING)
    feed(s, [c(9, 55, 100, 105, 96, 100)])            # consolidate
    feed(s, [c(10, 0, 108, 116, 107, 115)])           # close 115 > MH 110 -> T2 BUY
    assert s.setup == "T2"
    assert s.direction == "BUY (long)"
    assert s.state == State.AWAIT_T2_ENTRY            # entry deferred to 11:10
    feed(s, [c(10, 5, 116, 118, 114, 117)])           # before 11:05 -> keep holding
    assert s.state == State.AWAIT_T2_ENTRY
    feed(s, [c(11, 5, 120, 125, 119, 123)])           # entry = 11:05 close (price at 11:10)
    assert s.entry_price == 123
    assert s.state == State.DONE


def test_t2_bearish_early_breakout():
    s = new()
    feed(s, MARKING)
    feed(s, [c(9, 55, 100, 105, 96, 100)])
    feed(s, [c(10, 0, 92, 93, 84, 85)])               # close 85 < ML 90 -> T2 SELL
    assert s.setup == "T2"
    assert s.direction == "SELL (short)"
    assert s.state == State.AWAIT_T2_ENTRY
    feed(s, [c(11, 5, 80, 82, 76, 78)])               # entry = 11:05 close (price at 11:10)
    assert s.entry_price == 78
    assert s.state == State.DONE


# --------------------------------------------------------------------------- #
# T1 - delayed breakout at/after 10:25
# --------------------------------------------------------------------------- #
def _consolidate_through_1015():
    return [
        c(9, 55, 100, 105, 96, 100),
        c(10, 0, 100, 104, 97, 101),
        c(10, 5, 101, 103, 98, 100),
        c(10, 10, 100, 105, 96, 102),
        c(10, 15, 102, 106, 95, 100),
    ]


def test_t1_bearish_at_1025():
    s = new()
    feed(s, MARKING)
    feed(s, _consolidate_through_1015())
    feed(s, [c(10, 20, 100, 108, 95, 101)])           # still inside in the T2 window
    feed(s, [c(10, 25, 92, 93, 84, 85)])              # close 85 < ML 90 -> T1 SELL
    assert s.setup == "T1"
    assert s.direction == "SELL (short)"
    assert s.state == State.AWAIT_ENTRY
    feed(s, [c(10, 30, 84, 86, 80, 82)])
    assert s.entry_price == 84


def test_t1_bullish_at_1025():
    s = new()
    feed(s, MARKING)
    feed(s, _consolidate_through_1015())
    feed(s, [c(10, 20, 100, 108, 95, 101)])           # still inside in the T2 window
    feed(s, [c(10, 25, 108, 116, 107, 115)])          # close 115 > MH 110 -> T1 BUY
    assert s.setup == "T1"
    assert s.direction == "BUY (long)"


def test_t1_fires_at_a_much_later_candle_no_gate():
    """No single-candle gate: a break at 11:00 must still fire T1."""
    s = new()
    feed(s, MARKING)
    feed(s, _consolidate_through_1015())
    feed(s, [c(10, 25, 100, 108, 95, 101)])           # still inside at 10:25
    feed(s, [c(10, 30, 101, 107, 96, 100)])           # still inside
    feed(s, [c(11, 0, 108, 117, 107, 116)])           # break at 11:00 -> T1 BUY
    assert s.setup == "T1"
    assert s.direction == "BUY (long)"


def test_no_trade_when_price_stays_inside_all_day():
    s = new()
    feed(s, MARKING)
    feed(s, _consolidate_through_1015())
    feed(s, [c(10, 20, 100, 108, 95, 101), c(10, 25, 101, 107, 96, 100)])
    assert s.setup is None
    assert s.mh == 110                                 # range marked, just no break
    assert s.state == State.MONITORING


def test_t1_still_fires_just_before_1400():
    """A break on the 13:55 candle (closes 14:00) is still inside the T1 window."""
    s = new()
    feed(s, MARKING)
    feed(s, _consolidate_through_1015())
    feed(s, [c(13, 55, 108, 117, 107, 116)])           # break at 13:55 -> T1 BUY
    assert s.setup == "T1"
    assert s.direction == "BUY (long)"


def test_break_after_1400_is_no_trade():
    """The T1 window closes at 14:00: a break at 14:00 or later does NOT fire T1."""
    s = new()
    feed(s, MARKING)
    feed(s, _consolidate_through_1015())
    feed(s, [c(13, 55, 100, 108, 95, 101)])            # still inside just before 14:00
    feed(s, [c(14, 0, 108, 117, 107, 116)])            # break at 14:00 -> too late
    assert s.setup is None
    assert s.state == State.DONE                       # window closed -> no trade


# --------------------------------------------------------------------------- #
# Close-confirmation: a wick that closes back inside must NOT count
# --------------------------------------------------------------------------- #
def test_wick_pierce_does_not_trigger_then_real_close_does():
    s = new()
    feed(s, MARKING)
    feed(s, _consolidate_through_1015())
    # wick to 115 but closes 105 (inside) -> no breakout
    feed(s, [c(10, 20, 100, 115, 95, 105)])
    assert s.setup is None
    assert s.state == State.MONITORING
    # next candle truly closes above MH -> T1 BUY
    feed(s, [c(10, 25, 106, 118, 105, 116)])
    assert s.setup == "T1"
    assert s.direction == "BUY (long)"


def test_wick_below_during_monitoring_stays_in_range():
    s = new()
    feed(s, MARKING)
    feed(s, [c(9, 55, 100, 105, 96, 100)])
    feed(s, [c(10, 0, 95, 100, 85, 95)])              # low 85 pierces ML but close 95 inside
    assert s.setup is None
    assert s.state == State.MONITORING


# --------------------------------------------------------------------------- #
# Post-entry excursion tracking (feeds the T2/T3 ladder in scan.py)
# --------------------------------------------------------------------------- #
def test_post_low_tracks_after_short_entry():
    s = new()
    feed(s, MARKING)
    feed(s, _consolidate_through_1015())
    feed(s, [c(10, 25, 92, 93, 84, 85)])              # T1 SELL
    feed(s, [c(10, 30, 84, 86, 80, 82)])              # entry
    feed(s, [c(10, 35, 82, 83, 75, 78), c(10, 40, 78, 79, 70, 72)])
    assert s.post_low == 70                            # lowest low after entry
    assert s.entry_price == 84


def test_post_high_tracks_after_long_entry():
    s = new()
    feed(s, MARKING)
    feed(s, _consolidate_through_1015())
    feed(s, [c(10, 25, 108, 116, 107, 115)])          # T1 BUY
    feed(s, [c(10, 30, 116, 118, 114, 117)])          # entry
    feed(s, [c(10, 35, 117, 125, 116, 124), c(10, 40, 124, 130, 123, 129)])
    assert s.post_high == 130
    assert s.entry_price == 116


def test_new_session_resets_state():
    s = new()
    feed(s, MARKING)
    feed(s, [c(9, 55, 100, 105, 96, 100)])
    assert s.mh == 110
    next_day = DAY + timedelta(days=1)
    feed(s, [c(9, 15, 200, 210, 195, 205, day=next_day)])
    assert s.day == next_day
    assert s.mh is None                                # re-armed for the new day
    assert s.setup is None
