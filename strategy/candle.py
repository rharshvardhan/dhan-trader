"""
5-minute OHLC candle + a tick→candle aggregator.

The live Ticker feed gives LTP only, so we bucket LTP into fixed intervals to build
O/H/L/C. A candle is emitted when the FIRST tick of the next interval arrives (i.e.
the previous interval is then complete). Intervals with no ticks are simply skipped.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

# India Standard Time (no external dependency needed).
IST = timezone(timedelta(hours=5, minutes=30))


@dataclass(frozen=True)
class Candle:
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float

    def label(self) -> str:
        """Human label like '09:50-09:55'."""
        return f"{self.start:%H:%M}-{self.end:%H:%M}"


class CandleAggregator:
    """Aggregates (timestamp, price) ticks into fixed-interval OHLC candles."""

    def __init__(self, interval_min: int = 5):
        self.interval_min = interval_min
        self._delta = timedelta(minutes=interval_min)
        self._bucket: Optional[datetime] = None
        self._o = self._h = self._l = self._c = 0.0

    def _floor(self, ts: datetime) -> datetime:
        """Floor a timestamp to the start of its interval bucket."""
        minute = (ts.minute // self.interval_min) * self.interval_min
        return ts.replace(minute=minute, second=0, microsecond=0)

    def on_tick(self, ts: datetime, price: float) -> Optional[Candle]:
        """Add a tick. Returns the just-completed Candle when a bucket rolls over."""
        bucket = self._floor(ts)

        if self._bucket is None:                 # first ever tick
            self._open(bucket, price)
            return None

        if bucket == self._bucket:               # same interval → update OHLC
            self._h = max(self._h, price)
            self._l = min(self._l, price)
            self._c = price
            return None

        completed = self._finalize()             # new interval → close the old one
        self._open(bucket, price)
        return completed

    def _open(self, bucket: datetime, price: float) -> None:
        self._bucket = bucket
        self._o = self._h = self._l = self._c = price

    def _finalize(self) -> Candle:
        return Candle(self._bucket, self._bucket + self._delta,
                      self._o, self._h, self._l, self._c)
