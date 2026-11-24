"""timmytime: finds the date ranges a question talks about.

    import timmytime
    timmytime.parse("what did I decide last week", anchor=date(2026, 11, 20))
    # [DateRange(start=date(2026, 11, 9), end=date(2026, 11, 15))]
"""

from datetime import date, datetime
from functools import lru_cache

from timmytime.resolve import DateRange
from timmytime.runtime import Parser

__all__ = ["DateRange", "Parser", "parse"]


@lru_cache(maxsize=1)
def _default() -> Parser:
    return Parser()


def parse(text: str, anchor: datetime | date | None = None, tz: str = "UTC") -> list[DateRange]:
    """Date ranges in `text`, read relative to `anchor` (default: today in `tz`). [] = no date."""
    return _default().parse(text, anchor, tz)
