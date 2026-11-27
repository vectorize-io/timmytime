"""Drop-in for hindsight's temporal extraction.

    from text2date.hindsight import extract_temporal_constraint
    extract_temporal_constraint("what did I decide last week", reference_date)  # -> (start, end) | None

Same shape as hindsight_api.engine.search.temporal_extraction.extract_temporal_constraint: naive
datetimes, start at 00:00, end at 23:59:59.999999. hindsight takes one window today, so this returns
the first range the question names; `extract_all` returns every range for when it takes more.
Never raises: any failure is "no temporal constraint", as recall requires.
"""

import logging
from datetime import UTC, datetime, time
from functools import lru_cache

from text2date.resolve import DateRange
from text2date.runtime import Parser

logger = logging.getLogger(__name__)

# Open ends ("before 2020", "after May") become the widest dates datetime can hold.
_OPEN_START = datetime(1, 1, 1)
_OPEN_END = datetime(9999, 12, 31, 23, 59, 59, 999999)


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser()


def _window(r: DateRange) -> tuple[datetime, datetime]:
    start = datetime.combine(r.start, time.min) if r.start else _OPEN_START
    end = datetime.combine(r.end, time.max) if r.end else _OPEN_END
    return start, end


def extract_all(query: str, reference_date: datetime | None = None, tz: str = "UTC") -> list[tuple[datetime, datetime]]:
    """Every date window in the question, in order. [] = none."""
    try:
        anchor = reference_date or datetime.now(UTC).replace(tzinfo=None)
        return [_window(r) for r in _parser().parse(query, anchor, tz)]
    except Exception as e:  # recall must never fail because of date parsing
        logger.warning("text2date temporal extraction failed (%s): %s", type(e).__name__, e)
        return []


def extract_temporal_constraint(query: str, reference_date: datetime | None = None,
                                tz: str = "UTC") -> tuple[datetime, datetime] | None:
    """The first date window in the question, or None."""
    windows = extract_all(query, reference_date, tz)
    return windows[0] if windows else None
