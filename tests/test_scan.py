"""Tests for scan.py.

Covers the pure helpers (resample, counterpart_of, _verdict, _range_of) and the
cross-instrument decision in scan_latest: the Target-1 gate (BOTH legs must be
T1 in the SAME direction, else HOLD) and the T2/T3 ladder (each rung set only
if price kept running past the prior target).

The two data-fetch functions (run_day_strategy, counterpart_of) are monkey-
patched so no network / Dhan API is needed.
"""

import types
from datetime import date, datetime, timedelta

import pytest

import scan
from strategy.candle import IST
from strategy.range_breakout import State

NSE = scan.MarketFeed.NSE
FNO = scan.MarketFeed.NSE_FNO
IDX = scan.MarketFeed.IDX


# --------------------------------------------------------------------------- #
# resample
# --------------------------------------------------------------------------- #
def _row(hh, mm, o, h, l, cl):
    return (datetime(2026, 7, 7, hh, mm, tzinfo=IST), o, h, l, cl)


def test_resample_merges_rows_into_interval_candles():
    rows = [
        _row(9, 15, 100, 105, 99, 102),
        _row(9, 17, 102, 110, 98, 108),   # same 09:15-09:20 bucket
        _row(9, 20, 108, 112, 107, 111),  # next bucket
    ]
    out = scan.resample(rows, 5)
    assert len(out) == 2
    first = out[0]
    assert first.open == 100 and first.high == 110 and first.low == 98 and first.close == 108
    assert out[1].open == 108 and out[1].close == 111


def test_resample_sorts_unordered_rows():
    rows = [_row(9, 20, 108, 112, 107, 111), _row(9, 15, 100, 105, 99, 102)]
    out = scan.resample(rows, 5)
    assert out[0].start.minute == 15
    assert out[1].start.minute == 20


# --------------------------------------------------------------------------- #
# counterpart_of
# --------------------------------------------------------------------------- #
def test_counterpart_future_to_equity():
    cp = scan.counterpart_of("KOTAKBANK-JUL2026-FUT", FNO, {"KOTAKBANK": "123"}, {})
    assert cp == ("KOTAKBANK", "123", NSE, "EQUITY")


def test_counterpart_future_missing_equity_returns_none():
    assert scan.counterpart_of("KOTAKBANK-JUL2026-FUT", FNO, {}, {}) is None


def test_counterpart_equity_picks_nearest_unexpired_future():
    fut_map = {"KOTAKBANK": [
        (date(2020, 1, 30), "KOTAKBANK-JAN2020-FUT", "9", "FUTSTK"),   # expired
        (date(2099, 7, 30), "KOTAKBANK-JUL2099-FUT", "10", "FUTSTK"),  # future
    ]}
    cp = scan.counterpart_of("KOTAKBANK", NSE, {}, fut_map)
    assert cp == ("KOTAKBANK-JUL2099-FUT", "10", FNO, "FUTSTK")


def test_counterpart_equity_not_in_futmap_returns_none():
    assert scan.counterpart_of("SOMESTOCK", NSE, {}, {}) is None


def test_counterpart_index_segment_returns_none():
    assert scan.counterpart_of("NIFTY", IDX, {}, {}) is None


# --------------------------------------------------------------------------- #
# _verdict / _range_of
# --------------------------------------------------------------------------- #
def fstrat(setup=None, direction=None, entry=None, mh=None, ml=None,
           post_low=None, post_high=None, state=None):
    s = types.SimpleNamespace(setup=setup, direction=direction, entry_price=entry,
                              mh=mh, ml=ml, post_low=post_low, post_high=post_high)
    if state is None:
        state = State.AWAIT_ENTRY if (setup and entry is None) else State.DONE
    s.state = state
    return s


def test_verdict_none_is_dash():
    assert scan._verdict(None) == "-"


def test_verdict_confirmed_setup():
    assert scan._verdict(fstrat("T1", "SELL (short)", entry=90, mh=110, ml=90)) == "T1"


def test_verdict_breakout_without_entry_gets_star():
    s = fstrat("T1", "SELL (short)", entry=None, mh=110, ml=90, state=State.AWAIT_ENTRY)
    assert scan._verdict(s) == "T1*"


def test_verdict_pending_when_range_not_marked():
    assert scan._verdict(fstrat(mh=None)) == "PENDING"


def test_verdict_no_trade_when_range_marked_but_no_setup():
    assert scan._verdict(fstrat(mh=110, ml=90)) == "NO TRADE"


def test_range_of():
    assert scan._range_of(None) is None
    assert scan._range_of(fstrat(mh=110, ml=90)) == 20
    assert scan._range_of(fstrat(mh=None, ml=90)) is None


# --------------------------------------------------------------------------- #
# scan_latest - patched legs
# --------------------------------------------------------------------------- #
def patch_legs(monkeypatch, main, cp, cp_tuple=("STK", "1", NSE, "EQUITY")):
    seq = iter([(date(2026, 7, 8), main), (date(2026, 7, 8), cp)])
    monkeypatch.setattr(scan, "run_day_strategy", lambda *a, **k: next(seq))
    monkeypatch.setattr(scan, "counterpart_of", lambda *a, **k: cp_tuple)


def run(monkeypatch, main, cp, cp_tuple=("STK", "1", NSE, "EQUITY")):
    patch_legs(monkeypatch, main, cp, cp_tuple)
    return scan.scan_latest("FUT-X", "1", FNO, "FUTSTK", dhan=None,
                            eq_mapping={}, fut_map={})


# Reusable legs: future range = 20, stock range = 14 -> min 14 -> T1DIST = 7.
def main_sell(post_low=None):
    return fstrat("T1", "SELL (short)", entry=90, mh=110, ml=90, post_low=post_low)


def main_buy(post_high=None):
    return fstrat("T1", "BUY (long)", entry=90, mh=110, ml=90, post_high=post_high)


def cp_sell():
    return fstrat("T1", "SELL (short)", entry=50, mh=100, ml=86)


def cp_buy():
    return fstrat("T1", "BUY (long)", entry=50, mh=100, ml=86)


def test_both_t1_sell_computes_target(monkeypatch):
    r = run(monkeypatch, main_sell(), cp_sell())
    assert r["verdict"] == "T1"
    assert r["cp_setup"] == "T1"
    assert r["t1_dist"] == pytest.approx(7.0)
    assert r["t1_price"] == pytest.approx(83.0)   # 90 - 7 (SELL subtracts)


def test_both_t1_buy_computes_target(monkeypatch):
    r = run(monkeypatch, main_buy(), cp_buy())
    assert r["verdict"] == "T1"
    assert r["t1_price"] == pytest.approx(97.0)   # 90 + 7 (BUY adds)


def test_uses_smaller_range(monkeypatch):
    # stock range (14) < future range (20) -> min is 14 -> dist 7
    r = run(monkeypatch, main_sell(), cp_sell())
    assert r["t1_dist"] == pytest.approx(7.0)


def test_future_t1_stock_t2_is_hold(monkeypatch):
    cp = fstrat("T2", "SELL (short)", entry=50, mh=100, ml=86)
    r = run(monkeypatch, main_sell(), cp)
    assert r["verdict"] == "HOLD"
    assert r["cp_setup"] == "T2"
    assert r["t1_price"] is None


def test_both_t1_opposite_direction_is_hold(monkeypatch):
    r = run(monkeypatch, main_sell(), cp_buy())
    assert r["verdict"] == "HOLD"
    assert r["t1_price"] is None


def test_future_t1_stock_no_trade_is_hold(monkeypatch):
    cp = fstrat(setup=None, mh=100, ml=86)        # NO TRADE
    r = run(monkeypatch, main_sell(), cp)
    assert r["verdict"] == "HOLD"
    assert r["cp_setup"] == "NO TRADE"


def test_no_counterpart_is_hold(monkeypatch):
    # e.g. an index future with no cash equity leg -> cp_tuple None
    r = run(monkeypatch, main_sell(), cp_sell(), cp_tuple=None)
    assert r["verdict"] == "HOLD"
    assert r["cp_setup"] == "-"
    assert r["t1_price"] is None


def test_future_t2_is_not_hold_and_no_target(monkeypatch):
    m = fstrat("T2", "SELL (short)", entry=90, mh=110, ml=90)
    r = run(monkeypatch, m, cp_sell())
    assert r["verdict"] == "T2"
    assert r["t1_price"] is None


def test_future_no_trade(monkeypatch):
    m = fstrat(setup=None, mh=110, ml=90)
    r = run(monkeypatch, m, cp_sell())
    assert r["verdict"] == "NO TRADE"


def test_main_no_data_returns_none(monkeypatch):
    monkeypatch.setattr(scan, "run_day_strategy", lambda *a, **k: (None, None))
    monkeypatch.setattr(scan, "counterpart_of", lambda *a, **k: None)
    assert scan.scan_latest("FUT-X", "1", FNO, "FUTSTK", None, {}, {}) is None


def test_t1_confirmed_but_entry_not_formed(monkeypatch):
    # both legs T1 same dir, but the future's entry candle has not formed yet.
    m = fstrat("T1", "SELL (short)", entry=None, mh=110, ml=90, state=State.AWAIT_ENTRY)
    r = run(monkeypatch, m, cp_sell())
    assert r["verdict"] == "T1*"
    assert r["t1_dist"] == pytest.approx(7.0)      # distance known
    assert r["t1_price"] is None                   # but no entry -> no price
    assert r["t2_price"] is None and r["t3_price"] is None


# --------------------------------------------------------------------------- #
# T2 / T3 ladder (T1 SELL: T1=83, T2=78, T3=73)
# --------------------------------------------------------------------------- #
def test_ladder_sell_full_run_sets_t2_and_t3(monkeypatch):
    r = run(monkeypatch, main_sell(post_low=70), cp_sell())   # ran below T3
    assert r["t1_price"] == pytest.approx(83.0)
    assert r["t2_price"] == pytest.approx(78.0)
    assert r["t3_price"] == pytest.approx(73.0)


def test_ladder_sell_partial_sets_t2_only(monkeypatch):
    r = run(monkeypatch, main_sell(post_low=80), cp_sell())   # past T1, not T2
    assert r["t2_price"] == pytest.approx(78.0)
    assert r["t3_price"] is None


def test_ladder_sell_no_follow_through_sets_neither(monkeypatch):
    r = run(monkeypatch, main_sell(post_low=85), cp_sell())   # never reached T1
    assert r["t2_price"] is None
    assert r["t3_price"] is None


# T1 BUY: T1=97, T2=102, T3=107
def test_ladder_buy_full_run_sets_t2_and_t3(monkeypatch):
    r = run(monkeypatch, main_buy(post_high=110), cp_buy())
    assert r["t1_price"] == pytest.approx(97.0)
    assert r["t2_price"] == pytest.approx(102.0)
    assert r["t3_price"] == pytest.approx(107.0)


def test_ladder_buy_partial_sets_t2_only(monkeypatch):
    r = run(monkeypatch, main_buy(post_high=100), cp_buy())
    assert r["t2_price"] == pytest.approx(102.0)
    assert r["t3_price"] is None


def test_ladder_buy_no_follow_through_sets_neither(monkeypatch):
    r = run(monkeypatch, main_buy(post_high=95), cp_buy())
    assert r["t2_price"] is None
    assert r["t3_price"] is None
