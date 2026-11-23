from datetime import UTC, date, datetime

import pytest

from tlm.baseline.temporal_extraction import extract_temporal_constraint
from tlm.resolve import DateRange, resolve

TUE = datetime(2026, 11, 17, 12, 0)  # the spec's examples are asked on this Tuesday


def d(s: str) -> date:
    return date.fromisoformat(s)


def one(code, anchor=TUE, tz="UTC"):
    (r,) = resolve(code, anchor, tz)
    return (r.start and r.start.isoformat(), r.end and r.end.isoformat())


@pytest.mark.parametrize(
    "code, start, end",
    [
        # the spec's table
        ("day -1", "2026-11-16", "2026-11-16"),
        ("week -1", "2026-11-09", "2026-11-15"),
        ("day -3", "2026-11-14", "2026-11-14"),
        ("last 6 month", "2026-05-17", "2026-11-17"),
        ("week -1 mon", "2026-11-09", "2026-11-09"),
        ("2023-03", "2023-03-01", "2023-03-31"),
        ("may .. jul", "2026-05-01", "2026-07-31"),
        ("oct .. dec", "2025-10-01", "2025-12-31"),  # Dec hasn't come yet, so both last year
        ("sep .. feb", "2025-09-01", "2026-02-28"),
        # the rest of the grammar
        ("now", "2026-11-17", "2026-11-17"),
        ("mon", "2026-11-16", "2026-11-16"),
        ("tue", "2026-11-10", "2026-11-10"),  # a bare weekday never means today
        ("fri 1", "2026-11-20", "2026-11-20"),
        ("week 0 weekend", "2026-11-21", "2026-11-22"),
        ("weekend", "2026-11-14", "2026-11-15"),
        ("dec", "2025-12-01", "2025-12-31"),  # latest December already started
        ("nov", "2026-11-01", "2026-11-30"),
        ("may -1", "2025-05-01", "2025-05-31"),
        ("year -1 may", "2025-05-01", "2025-05-31"),
        ("2024 summer", "2024-06-01", "2024-08-31"),
        ("summer -1", "2025-06-01", "2025-08-31"),
        ("winter", "2025-12-01", "2026-02-28"),
        ("q3", "2026-07-01", "2026-09-30"),
        ("quarter -1", "2026-07-01", "2026-09-30"),
        ("1990s", "1990-01-01", "1999-12-31"),
        ("2024-03-12", "2024-03-12", "2024-03-12"),
        ("*-03-12", "2026-03-12", "2026-03-12"),
        ("*-12-25", "2025-12-25", "2025-12-25"),
        ("*-02-29", "2024-02-29", "2024-02-29"),
        ("jan .. now", "2026-01-01", "2026-11-17"),
        ("2025 ..", "2025-01-01", None),
        (".. 2019", None, "2019-12-31"),
        ("before 2020", None, "2019-12-31"),
        ("after 2026-03", "2026-04-01", None),
        ("next 2 week", "2026-11-17", "2026-12-01"),
        ("early year -1", "2025-01-01", "2025-04-30"),
        ("mid year -1", "2025-05-01", "2025-08-31"),
        ("late 2025-03", "2025-03-23", "2025-03-31"),
        ("h2 2024", "2024-07-01", "2024-12-31"),
        ("around 2026-05-10", "2026-05-07", "2026-05-13"),
        ("recent", "2026-10-18", "2026-11-17"),
        ("longago", None, "2025-11-17"),
    ],
)
def test_codes(code, start, end):
    assert one(code) == (start, end)


def test_several_ranges_in_order():
    assert resolve("2025-05 ; 2026-06", TUE) == [
        DateRange(d("2025-05-01"), d("2025-05-31")),
        DateRange(d("2026-06-01"), d("2026-06-30")),
    ]


def test_southern_seasons_follow_timezone():
    assert one("2024 summer", tz="Australia/Sydney") == ("2024-12-01", "2025-02-28")
    assert one("2024 summer", tz="America/Argentina/Buenos_Aires") == ("2024-12-01", "2025-02-28")
    assert one("2024 summer", tz="Europe/Rome") == ("2024-06-01", "2024-08-31")


def test_timezone_picks_the_day():
    late = datetime(2026, 11, 17, 23, 30, tzinfo=UTC)
    assert one("now", late) == ("2026-11-17", "2026-11-17")
    assert one("now", late, tz="Asia/Tokyo") == ("2026-11-18", "2026-11-18")
    assert one("now", late.replace(tzinfo=None)) == ("2026-11-17", "2026-11-17")  # naive = UTC


def test_unknown_timezone_raises():
    with pytest.raises(Exception):
        resolve("now", TUE, tz="Mars/Olympus")


@pytest.mark.parametrize(
    "code",
    ["", "none", ";", "day", "day x", "week 1.5", "yesterday", "2026-13", "2026-02-30", "0000",
     "2026 .. 2025", "..", "before", "last -3 day", "last 3 weekend", "day 99999", "year 9000",
     "month 3 weekend", "2026-05 ;", "early", "around around now", "x" * 10_000, "\x00", "日 -1"],
)
def test_unreadable_lines_give_no_date(code):
    assert resolve(code, TUE) == []


# The old hand-written rules, as codes. The calendar must agree with them exactly.
OLD_RULES = {
    "what happened yesterday": "day -1",
    "позавчера": "day -2",
    "notes from today": "now",
    "a couple of days ago": "couple day",
    "a few days ago": "few day",
    "a couple of weeks ago": "couple week",
    "a few weeks ago": "few week",
    "a couple of months ago": "couple month",
    "a few months ago": "few month",
    "last week": "week -1",
    "last month": "month -1",
    "last year": "year -1",
    "last weekend": "week -1 weekend",
    "march 2023": "2023-03",
    "in 2019": "2019",
}
REFERENCE_DATES = [
    datetime(2026, 8, 12, 14, 30), datetime(2026, 1, 1), datetime(2026, 12, 31, 23, 59),
    datetime(2024, 2, 29, 12), datetime(2026, 8, 3, 9), datetime(2026, 8, 8), datetime(2026, 8, 9, 18),
]


@pytest.mark.parametrize("query, code", OLD_RULES.items())
@pytest.mark.parametrize("ref", REFERENCE_DATES, ids=lambda r: r.date().isoformat())
def test_same_answers_as_old_rules(query, code, ref):
    old = extract_temporal_constraint(query, reference_date=ref)
    assert resolve(code, ref) == [DateRange(old[0].date(), old[1].date())]


@pytest.mark.parametrize("code,want", [
    ("easter", ("2026-04-03", "2026-04-06")),            # Good Friday to Easter Monday
    ("2024 easter", ("2024-03-29", "2024-04-01")),
    ("easter 1", ("2027-03-26", "2027-03-29")),
    ("thanksgiving", ("2025-11-27", "2025-11-27")),      # 2026's is Nov 26, after Nov 20
    ("year 0 thanksgiving", ("2026-11-26", "2026-11-26")),
    ("cny", ("2026-02-17", "2026-02-23")),
    ("ramadan", ("2026-02-18", "2026-03-19")),
    ("2025 eidadha", ("2025-06-06", "2025-06-09")),
    ("*-12-25", ("2025-12-25", "2025-12-25")),           # Christmas needs no word
    ("around 2022-12-25", ("2022-12-22", "2022-12-28")),
    ("late mar", ("2026-03-23", "2026-03-31")),
])
def test_holidays_and_parts(code, want):
    got = resolve(code, datetime(2026, 11, 20))
    assert [(s.isoformat(), e.isoformat()) for s, e in got] == [want]


def test_lunar_table_ends_cleanly():
    assert resolve("2090 cny", datetime(2026, 11, 20)) == []
