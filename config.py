"""
Load instrument + feed configuration from a YAML file (instruments.yaml).

Dhan subscribes by numeric security_id + exchange segment (NOT symbol). The YAML
lets you list one or many instruments; they all stream over a SINGLE websocket
connection (Dhan allows up to 5000 instruments/connection, 5 connections/user),
so adding instruments here does not add API calls or risk the quote rate limit.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml
from dhanhq import MarketFeed


class InstrumentConfigError(Exception):
    """Raised for any problem loading/parsing instruments.yaml."""


# Friendly exchange-segment names -> Dhan MarketFeed constants. Both the SDK's
# string names (NSE_EQ) and short aliases (NSE) are accepted.
SEGMENT_MAP = {
    "IDX": MarketFeed.IDX, "IDX_I": MarketFeed.IDX, "INDEX": MarketFeed.IDX,
    "NSE": MarketFeed.NSE, "NSE_EQ": MarketFeed.NSE,
    "NSE_FNO": MarketFeed.NSE_FNO, "FNO": MarketFeed.NSE_FNO,
    "NSE_CURR": MarketFeed.NSE_CURR, "NSE_CURRENCY": MarketFeed.NSE_CURR,
    "BSE": MarketFeed.BSE, "BSE_EQ": MarketFeed.BSE,
    "MCX": MarketFeed.MCX, "MCX_COMM": MarketFeed.MCX,
    "BSE_CURR": MarketFeed.BSE_CURR, "BSE_CURRENCY": MarketFeed.BSE_CURR,
    "BSE_FNO": MarketFeed.BSE_FNO,
}

FEED_TYPE_MAP = {
    "Ticker": MarketFeed.Ticker,   # LTP only (enough for the strategy)
    "Quote": MarketFeed.Quote,     # + OHLC/volume
    "Full": MarketFeed.Full,       # + market depth/OI
}


@dataclass(frozen=True)
class Instrument:
    name: str          # display label only
    security_id: str   # Dhan security_id (from the scrip master)
    segment: int       # a MarketFeed.* constant


@dataclass(frozen=True)
class AppConfig:
    instruments: List[Instrument]
    feed_version: str = "v2"
    feed_type: int = MarketFeed.Ticker
    interval_min: int = 5


def load_config(path) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise InstrumentConfigError(
            f"Config file not found: {p}. Create it (see instruments.yaml format)."
        )
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise InstrumentConfigError(f"Invalid YAML in {p}: {exc}") from exc

    raw = data.get("instruments") or []
    if not raw:
        raise InstrumentConfigError(f"No 'instruments' listed in {p}.")

    instruments: List[Instrument] = []
    seen = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InstrumentConfigError(f"instruments[{idx}] must be a mapping.")
        try:
            name = str(item["name"]).strip()
            sid = str(item["security_id"]).strip()
            seg_name = str(item["exchange_segment"]).strip().upper()
        except KeyError as exc:
            raise InstrumentConfigError(
                f"instruments[{idx}] needs name, security_id, exchange_segment "
                f"(missing {exc})."
            ) from exc
        if seg_name not in SEGMENT_MAP:
            raise InstrumentConfigError(
                f"instruments[{idx}] '{name}': unknown exchange_segment '{seg_name}'. "
                f"Valid: {sorted(SEGMENT_MAP)}"
            )
        key = (SEGMENT_MAP[seg_name], sid)
        if key in seen:                       # silently de-dupe exact repeats
            continue
        seen.add(key)
        instruments.append(Instrument(name, sid, SEGMENT_MAP[seg_name]))

    feed = data.get("feed") or {}
    strat = data.get("strategy") or {}

    ft_name = str(feed.get("type", "Ticker"))
    if ft_name not in FEED_TYPE_MAP:
        raise InstrumentConfigError(
            f"feed.type '{ft_name}' invalid. Valid: {sorted(FEED_TYPE_MAP)}"
        )

    try:
        interval = int(strat.get("interval_min", 5))
    except (TypeError, ValueError) as exc:
        raise InstrumentConfigError("strategy.interval_min must be an integer.") from exc
    if interval <= 0:
        raise InstrumentConfigError("strategy.interval_min must be > 0.")

    return AppConfig(
        instruments=instruments,
        feed_version=str(feed.get("version", "v2")),
        feed_type=FEED_TYPE_MAP[ft_name],
        interval_min=interval,
    )
