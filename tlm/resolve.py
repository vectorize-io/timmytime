"""Meaning code -> date ranges. The calendar half of tlm; it knows no language.

The model writes one line in this code; this module turns it into dates.
Whole days only. Weeks start on Monday. Tokens are split on spaces.

  line   := none | range (; range)*
  range  := period | [period] .. [period] | before period | after period
          | last N unit | next N unit | fuzzy
  period := [early|mid|late|h1|h2|around] base
  base   := now | unit N | YYYY | YYYY-MM | YYYY-MM-DD | *-MM-DD | YYYYs
          | sel [N] | (year N | YYYY) (month|season|quarter) | week N (weekday|weekend)
  unit   := day | week | month | quarter | year            N = signed int, 0 = current
  sel    := jan..dec | mon..sun | spring summer autumn winter | q1..q4 | weekend | holiday
  holiday := easter (Good Friday to Easter Monday) | thanksgiving (US) | cny (7 days) | midautumn
          | dragonboat | ramadan | eidfitr (3 days) | eidadha (4 days)
             Fixed-date holidays need no word: Christmas = *-12-25, Christmas 2022 = 2022-12-25.
  fuzzy  := recent | otherday | while | longago | (couple|few) unit

A bare `sel` is the latest one already started (a weekday: the latest before
today); `sel N` steps N cycles from there. `*-MM-DD` is the latest such date not
after today. Seasons follow the hemisphere of `tz`. Anything unreadable -> [].
"""

import calendar
import json
import re
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from importlib import resources
from typing import NamedTuple
from zoneinfo import ZoneInfo


class DateRange(NamedTuple):
    start: date | None  # None = open ("before 2020")
    end: date | None  # None = open ("after May")


UNITS = ("day", "week", "month", "quarter", "year")
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
QUARTERS = ("q1", "q2", "q3", "q4")
# First month of each season, northern hemisphere; the south is six months off.
SEASONS = {"spring": 3, "summer": 6, "autumn": 9, "winter": 12}
# Lunar holidays per year, from scripts/make_holidays.py (1950-2076).
LUNAR = json.loads(resources.files("tlm").joinpath("holidays.json").read_text())
HOLIDAYS = ("easter", "thanksgiving", *LUNAR)
MODS = ("early", "mid", "late", "h1", "h2", "around")
MAX_N = 10_000

# Fixed windows, in days back from today: (start, end); None = open.
# The couple/few day, week and month windows are hindsight's old rules, unchanged.
# ponytail: recent/otherday/while/longago/years are our own picks, not agreed yet.
FUZZY = {
    "couple day": (3, 1),
    "few day": (5, 2),
    "couple week": (21, 7),
    "few week": (35, 14),
    "couple month": (90, 30),
    "few month": (150, 60),
    "couple year": (3 * 365, 365),
    "few year": (5 * 365, 2 * 365),
    "recent": (30, 0),
    "otherday": (7, 1),
    "while": (180, 30),
    "longago": (None, 365),
}

_N = re.compile(r"[+-]?\d{1,5}$")
_ISO = re.compile(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


class Unreadable(ValueError):
    pass


def resolve(code: str, anchor: datetime | None = None, tz: str = "UTC") -> list[DateRange]:
    """Date ranges for one meaning-code line. [] = no date, or a line we can't read.

    `anchor` is when the question was asked (naive = UTC, default now); `tz` picks
    which calendar day that is, and the hemisphere for seasons.
    """
    zone = ZoneInfo(tz)  # an unknown tz is the caller's bug: raise, don't hide it
    anchor = anchor or datetime.now(UTC)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=UTC)
    today = anchor.astimezone(zone).date()
    try:
        return _Line(today, _is_southern(tz)).read(code)
    except (Unreadable, ValueError, OverflowError, IndexError):
        return []


class _Line:
    def __init__(self, today: date, south: bool):
        self.today = today
        self.south = south

    def read(self, code: str) -> list[DateRange]:
        tokens = code.replace(";", " ; ").split()
        if tokens == ["none"]:
            return []
        ranges, part = [], []
        for t in [*tokens, ";"]:
            if t != ";":
                part.append(t)
                continue
            if not part:
                raise Unreadable("empty range")
            ranges.append(self.range(part))
            part = []
        return ranges

    def range(self, t: list[str]) -> DateRange:
        if ".." in t:
            i = t.index("..")
            start = self.period(t[:i])[0] if t[:i] else None
            end = self.period(t[i + 1 :])[1] if t[i + 1 :] else None
            if start and end and start > end and len(t[:i]) == 1 and _is_sel(t[0]):
                # "between March and July" asked in April: last year's March, not no date
                start = self.cycle(t[0], -1)[0]
            if start is None and end is None:
                raise Unreadable("both ends open")
            if start and end and start > end:
                raise Unreadable("start after end")
            return DateRange(start, end)
        if t[0] == "before":
            return DateRange(None, self.period(t[1:])[0] - timedelta(days=1))
        if t[0] == "after":
            return DateRange(self.period(t[1:])[1] + timedelta(days=1), None)
        if t[0] in ("last", "next") and len(t) == 3:
            n = _n(t[1], signed=False)
            moved = _shift(self.today, t[2], n if t[0] == "next" else -n)
            return DateRange(*sorted((moved, self.today)))
        key = " ".join(t)
        if key in FUZZY:
            back_start, back_end = FUZZY[key]
            start = None if back_start is None else self.today - timedelta(days=back_start)
            return DateRange(start, self.today - timedelta(days=back_end))
        return DateRange(*self.period(t))

    def period(self, t: list[str]) -> tuple[date, date]:
        if t and t[0] in MODS:
            return _modify(t[0], self.base(t[1:]))
        return self.base(t)

    def base(self, t: list[str]) -> tuple[date, date]:
        if t == ["now"]:
            return self.today, self.today
        if len(t) == 2 and t[0] in UNITS and _N.match(t[1]):
            return _unit_span(self.today, t[0], _n(t[1]))
        if len(t) == 1:
            return self.absolute(t[0])
        if len(t) == 2 and _is_sel(t[0]) and _N.match(t[1]):
            return self.cycle(t[0], _n(t[1]))
        if len(t) == 3 and t[0] in ("year", "week"):
            return self.within(_unit_span(self.today, t[0], _n(t[1])), t[0], t[2])
        if len(t) == 2 and re.fullmatch(r"\d{4}", t[0]):
            return self.within(self.absolute(t[0]), "year", t[1])
        raise Unreadable(" ".join(t))

    def absolute(self, tok: str) -> tuple[date, date]:
        if _is_sel(tok):
            return self.cycle(tok, 0)
        if m := re.fullmatch(r"(\d{3})0s", tok):
            y = int(m[1]) * 10
            return date(y, 1, 1), date(y + 9, 12, 31)
        if m := re.fullmatch(r"\*-(\d{2})-(\d{2})", tok):
            month, day = int(m[1]), int(m[2])
            for y in range(self.today.year, self.today.year - 9, -1):  # 9 years covers Feb 29
                try:
                    d = date(y, month, day)
                except ValueError:
                    continue
                if d <= self.today:
                    return d, d
            raise Unreadable(tok)
        m = _ISO.match(tok)
        if not m:
            raise Unreadable(tok)
        y, month, day = int(m[1]), m[2] and int(m[2]), m[3] and int(m[3])
        if day:
            d = date(y, month, day)
            return d, d
        if month:
            return _month_span(y, month)
        return date(y, 1, 1), date(y, 12, 31)

    def cycle(self, sel: str, n: int) -> tuple[date, date]:
        """The latest `sel` already started, then n cycles on."""
        if sel in WEEKDAYS:
            back = (self.today.weekday() - WEEKDAYS.index(sel)) % 7 or 7
            d = self.today - timedelta(days=back) + timedelta(weeks=n)
            return d, d
        if sel == "weekend":
            monday = self.today - timedelta(days=self.today.weekday())
            sat = monday + timedelta(days=5)
            if sat > self.today:
                sat -= timedelta(weeks=1)
            sat += timedelta(weeks=n)
            return sat, sat + timedelta(days=1)
        year = self.today.year
        if self._in_year(sel, year)[0] > self.today:
            year -= 1
        return self._in_year(sel, year + n)

    def within(self, span: tuple[date, date], unit: str, sel: str) -> tuple[date, date]:
        if unit == "year" and (sel in MONTHS or sel in SEASONS or sel in QUARTERS or sel in HOLIDAYS):
            return self._in_year(sel, span[0].year)
        if unit == "week" and sel in WEEKDAYS:
            d = span[0] + timedelta(days=WEEKDAYS.index(sel))
            return d, d
        if unit == "week" and sel == "weekend":
            return span[0] + timedelta(days=5), span[1]
        raise Unreadable(f"{unit} {sel}")

    def _in_year(self, sel: str, year: int) -> tuple[date, date]:
        if sel in MONTHS:
            return _month_span(year, MONTHS.index(sel) + 1)
        if sel in QUARTERS:
            first = 3 * QUARTERS.index(sel) + 1
            return date(year, first, 1), _month_span(year, first + 2)[1]
        if sel == "easter":
            sunday = _easter(year)
            return sunday - timedelta(days=2), sunday + timedelta(days=1)
        if sel == "thanksgiving":
            nov1 = date(year, 11, 1)
            d = nov1 + timedelta(days=(3 - nov1.weekday()) % 7 + 21)  # 4th Thursday
            return d, d
        if sel in LUNAR:
            # ponytail: one entry per year (the first); a Ramadan twice in one year (2030) keeps January's
            if str(year) not in LUNAR[sel]:
                raise Unreadable(f"{sel} {year}")
            start, end = LUNAR[sel][str(year)]
            return date.fromisoformat(start), date.fromisoformat(end)
        first = SEASONS[sel] if not self.south else (SEASONS[sel] + 6 - 1) % 12 + 1
        end = _add_months(date(year, first, 1), 3) - timedelta(days=1)
        return date(year, first, 1), end  # winter 2024 = Dec 2024 to Feb 2025


def _is_sel(tok: str) -> bool:
    return tok in MONTHS or tok in WEEKDAYS or tok in SEASONS or tok in QUARTERS or tok in HOLIDAYS or tok == "weekend"


def _easter(year: int) -> date:
    """Western Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    n = h + l - 7 * ((a + 11 * h + 22 * l) // 451) + 114
    return date(year, n // 31, n % 31 + 1)


def _n(tok: str, signed: bool = True) -> int:
    if not _N.match(tok) or (not signed and tok[0] in "+-"):
        raise Unreadable(tok)
    n = int(tok)
    if abs(n) > MAX_N:
        raise Unreadable(tok)
    return n


def _month_span(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _shift(d: date, unit: str, n: int) -> date:
    if unit == "day":
        return d + timedelta(days=n)
    if unit == "week":
        return d + timedelta(weeks=n)
    if unit in ("month", "quarter", "year"):
        return _add_months(d, n * {"month": 1, "quarter": 3, "year": 12}[unit])
    raise Unreadable(unit)


def _unit_span(today: date, unit: str, n: int) -> tuple[date, date]:
    if unit == "day":
        d = today + timedelta(days=n)
        return d, d
    if unit == "week":
        monday = today - timedelta(days=today.weekday()) + timedelta(weeks=n)
        return monday, monday + timedelta(days=6)
    months = {"month": 1, "quarter": 3, "year": 12}[unit]
    first_month = (today.month - 1) // months * months + 1
    start = _add_months(date(today.year, first_month, 1), n * months)
    return start, _add_months(start, months) - timedelta(days=1)


def _modify(mod: str, span: tuple[date, date]) -> tuple[date, date]:
    start, end = span
    if mod == "around":
        pad = timedelta(days=max(3, ((end - start).days + 1) // 2))
        return start - pad, end + pad
    parts, pick = {"early": (3, 0), "mid": (3, 1), "late": (3, 2), "h1": (2, 0), "h2": (2, 1)}[mod]
    months = _whole_months(start, end)
    if months and months % parts == 0:  # a year splits into Jan-Apr, May-Aug, Sep-Dec
        step = months // parts
        first = _add_months(start, step * pick)
        return first, _add_months(first, step) - timedelta(days=1)
    days = (end - start).days + 1
    size = -(-days // parts)  # ceil
    first = start + timedelta(days=size * pick)
    return first, min(end, first + timedelta(days=size - 1))


def _whole_months(start: date, end: date) -> int:
    if start.day != 1 or (end + timedelta(days=1)).day != 1:
        return 0
    return (end.year - start.year) * 12 + end.month - start.month + 1


@lru_cache
def _is_southern(tz: str) -> bool:
    """Whether a timezone's main city is south of the equator. Unknown (UTC, links) -> north."""
    # ponytail: only canonical names in zone tables; alias links like Australia/ACT read as north.
    for table in ("zone1970.tab", "zone.tab"):
        text = resources.files("tzdata").joinpath("zoneinfo", table).read_text()
        for line in text.splitlines():
            cols = line.split("\t")
            if len(cols) >= 3 and not line.startswith("#") and cols[2] == tz:
                return cols[1].startswith("-")
    return False
