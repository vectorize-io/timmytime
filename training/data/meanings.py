"""Random meanings to write questions about: (kind, code, plain-English description).

The description is what the LLM sees. It must say exactly what the code means,
so a question written from it carries that code as its label.
English conventions fixed here: "last Monday" = the latest Monday before today;
"Monday last week" = the Monday of the previous calendar week; "last summer" =
the latest summer already started.
"""

import calendar
import random

from timmytime.resolve import MONTHS, WEEKDAYS

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEASON_NAMES = ["spring", "summer", "autumn", "winter"]
UNIT_WORDS = {"day": "days", "week": "weeks", "month": "months", "quarter": "quarters", "year": "years"}
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


FAR_SHARE = 0.08  # share of years far from today (gen --far-share)
MONTH_END_SHARE = 0.1  # share of days forced to the 29th-31st (gen --month-end-share)



def _day(r, y, m):
    """A real day of that month (y=None: any year, so no Feb 29). The last days get extra weight:
    they were missing for a long time (every sampler stopped at 28)."""
    last = calendar.monthrange(y or 2001, m)[1]
    return r.randint(29, last) if last > 28 and r.random() < MONTH_END_SHARE else r.randint(1, last)

def _year(r: random.Random) -> int:
    k = r.random()
    if k < FAR_SHARE / 2:  # far from today: history, old records, long-term plans
        return r.randint(1800, 1979)
    if k < FAR_SHARE:
        return r.randint(2028, 2199)
    return r.choice([r.randint(2015, 2027)] * 4 + [r.randint(1980, 2014)])


def _rel(unit: str, n: int) -> str:
    """Plain words for `unit n` (a whole calendar period)."""
    if unit == "day":
        return {0: "today", -1: "yesterday", -2: "the day before yesterday", 1: "tomorrow",
                2: "the day after tomorrow"}.get(n, f"exactly {-n} days ago (that single day)" if n < 0
                                                    else f"exactly {n} days from now (that single day)")
    the = {"week": "calendar week (Monday to Sunday)", "month": "calendar month",
           "quarter": "calendar quarter", "year": "calendar year"}[unit]
    if n == 0:
        return f"this {unit} (the current {the})"
    if n == -1:
        return f"last {unit} (the previous {the})"
    if n == 1:
        return f"next {unit} (the following {the})"
    if n == -2:
        return f"the {unit} before last (the {the} two {UNIT_WORDS[unit]} back)"
    return f"the {the} {abs(n)} {UNIT_WORDS[unit]} {'ago' if n < 0 else 'from now'} (that whole {unit})"


def _date_text(r: random.Random, y: int, m: int, d: int) -> str:
    month = MONTH_NAMES[m - 1]
    styles = [f"{month} {d}, {y}", f"{d} {month} {y}", f"{y}-{m:02d}-{d:02d}", f"{month[:3]} {d} {y}",
              f"{m}/{d}/{y} (US style month/day/year)", f"the {d}{_th(d)} of {month} {y}"]
    return r.choice(styles)


def _th(d: int) -> str:
    return "th" if 10 < d % 100 < 14 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")


# Each family returns (code, description). Weights roughly follow how often people ask.
def f_day(r):
    n = r.choice([-1] * 8 + [0] * 3 + [-2] * 2 + list(range(-10, -2)) + [1, 1, 2, 3, 4, -4, -15, -20, -30, -45, -60, -99])
    return f"day {n}", _rel("day", n)


def f_week(r):
    n = r.choice([-1] * 6 + [0] * 3 + [-2] * 2 + [-3, -4, -5, -6, -8, 1, 1, 2, 2, 3])
    return f"week {n}", _rel("week", n)


def f_month(r):
    n = r.choice([-1] * 6 + [0] * 3 + [-2] * 2 + [-3, -4, -5, -6, -8, -12, -18, -22, 1, 2, 2, 3])
    return f"month {n}", _rel("month", n)


def f_year(r):
    n = r.choice([-1] * 6 + [0] * 3 + [-2] * 2 + [-3, -3, -4, -5, -10, 1, 2, 2, 3])
    return f"year {n}", _rel("year", n)


def f_quarter(r):
    k = r.random()
    if k < 0.5:
        n = r.choice([-1, -1, 0, 1, -2])
        return f"quarter {n}", _rel("quarter", n)
    q = r.randint(1, 4)
    if k < 0.8:
        y = _year(r)
        return f"{y} q{q}", f"Q{q} of {y} (the {['first', 'second', 'third', 'fourth'][q - 1]} quarter of {y})"
    return f"q{q}", f"Q{q}, no year given (the most recent Q{q} that has started)"


def f_weekday(r):
    i = r.randrange(7)
    k = r.random()
    if k < 0.5:
        return WEEKDAYS[i], f"last {DAY_NAMES[i]}, or just 'on {DAY_NAMES[i]}' about the past (the most recent {DAY_NAMES[i]} before today)"
    if k < 0.65:
        return f"{WEEKDAYS[i]} 1", f"this coming {DAY_NAMES[i]} (the next {DAY_NAMES[i]} from now)"
    n = r.choice([-1, -1, 0, 0, 1])
    which = {-1: "last week", 0: "this week", 1: "next week"}[n]
    return f"week {n} {WEEKDAYS[i]}", f"{DAY_NAMES[i]} {which} (the {DAY_NAMES[i]} of {which}'s calendar week)"


def f_weekend(r):
    n = r.choice([-1, -1, -1, 0, 0, 1, -2])
    words = {-1: "last weekend", 0: "this weekend", 1: "next weekend", -2: "the weekend before last"}[n]
    return f"week {n} weekend", f"{words} (Saturday and Sunday)"


def f_month_name(r):
    m = r.randrange(12)
    k = r.random()
    if k < 0.4:
        return MONTHS[m], f"in {MONTH_NAMES[m]}, no year given (the most recent {MONTH_NAMES[m]} that has started)"
    if k < 0.7:
        n = r.choice([-1, -1, 0, 0, 1, -2, 2])
        which = {-1: "last year", 0: "this year", 1: "next year", -2: "two years ago", 2: "the year after next"}[n]
        return f"year {n} {MONTHS[m]}", f"{MONTH_NAMES[m]} of {which}"
    y = _year(r)
    return f"{y}-{m + 1:02d}", f"{MONTH_NAMES[m]} {y} (the whole month)"


def f_date(r):
    y, m = _year(r), r.randint(1, 12)
    d = _day(r, y, m)
    if r.random() < 0.3:
        d = _day(r, None, m)
        return f"*-{m:02d}-{d:02d}", f"{r.choice([f'{MONTH_NAMES[m - 1]} {d}', f'{d} {MONTH_NAMES[m - 1]}', f'the {d}{_th(d)} of {MONTH_NAMES[m - 1]}'])}, no year given (that single day)"
    return f"{y}-{m:02d}-{d:02d}", f"the single day written as: {_date_text(r, y, m, d)}"


def f_year_abs(r):
    y = _year(r)
    return str(y), f"the year {y} (the whole year)"


def f_season(r):
    s = r.randrange(4)
    name = SEASON_NAMES[s]
    k = r.random()
    if k < 0.4:
        return name, f"'last {name}' or 'in the {name}' (the most recent {name} that has started)"
    if k < 0.7:
        y = _year(r)
        return f"{y} {name}", f"{name} {y}"
    return f"year -1 {name}", f"the {name} of last year"


def f_decade(r):
    y = r.choice([1960, 1970, 1980, 1990, 2000, 2010])
    return f"{y}s", f"the {y}s (the decade {y} to {y + 9})"


def f_last_n(r):
    unit = r.choice(["day"] * 3 + ["week"] * 3 + ["month"] * 3 + ["year"])
    if r.random() < 0.2:  # one unit counted back from today, or "half a month" (15 days)
        n, unit = r.choice([(1, "day"), (1, "week"), (1, "month"), (1, "year"), (15, "day"), (14, "day"), (30, "day")])
        way = r.choice(["last", "last", "next"])
        when = "ago up to today" if way == "last" else "from now, starting today"
        desc = {(1, "day"): f"the {'past' if way == 'last' else 'next'} 24 hours (one day {when})",
                (15, "day"): f"the {way} half month (15 days {when})"}.get(
            (n, unit), f"the {way} {n} {unit if n == 1 else UNIT_WORDS[unit]} counted from today ({n} {unit if n == 1 else UNIT_WORDS[unit]} {when}, not the calendar {unit})")
        return f"{way} {n} {unit}", desc
    n = r.randint(2, 12) if r.random() < 0.85 else r.randint(13, 30)
    word = r.choice([str(n), NUM_WORDS[n]]) if n <= 12 else str(n)
    if r.random() < 0.85:
        return f"last {n} {unit}", f"the last {word} {UNIT_WORDS[unit]} (from {n} {UNIT_WORDS[unit]} ago up to today)"
    return f"next {n} {unit}", f"the next {word} {UNIT_WORDS[unit]} (from today up to {n} {UNIT_WORDS[unit]} from now)"


FUZZY_TEXT = {
    "recent": "recently or lately (a vague recent past)",
    "otherday": "'the other day' (a few days ago, vague)",
    "while": "'a while ago' (some weeks or months ago, vague)",
    "longago": "'a long time ago' or 'ages ago' (more than a year ago, vague)",
    "couple day": "a couple of days ago", "few day": "a few days ago",
    "couple week": "a couple of weeks ago", "few week": "a few weeks ago",
    "couple month": "a couple of months ago", "few month": "a few months ago",
    "couple year": "a couple of years ago", "few year": "a few years ago",
}


def f_fuzzy(r):
    code = r.choice(list(FUZZY_TEXT))
    return code, FUZZY_TEXT[code]


def f_modified(r):
    mod = r.choice(["early"] * 3 + ["mid"] * 2 + ["late"] * 3 + ["h1", "h2", "around", "around"])
    if mod == "around":
        if r.random() < 0.5:
            y, m = _year(r), r.randint(1, 12)
            d = _day(r, y, m)
            return f"around {y}-{m:02d}-{d:02d}", f"around {_date_text(r, y, m, d)} (roughly that date, give or take a few days)"
        y, m = _year(r), r.randint(1, 12)
        return f"around {y}-{m:02d}", f"around {MONTH_NAMES[m - 1]} {y} (roughly, give or take)"
    word = {"early": "early", "mid": "the middle of", "late": "late", "h1": "the first half of", "h2": "the second half of"}[mod]
    k = r.random()
    if k < 0.35:
        n = r.choice([-1, 0])
        return f"{mod} year {n}", f"{word} {'last' if n else 'this'} year"
    if k < 0.6:
        y = _year(r)
        return f"{mod} {y}", f"{word} {y}"
    if k < 0.8:
        n = r.choice([-1, 0])
        return f"{mod} month {n}", f"{word} {'last' if n else 'this'} month"
    if k < 0.9:
        y, m = _year(r), r.randint(1, 12)
        return f"{mod} {y}-{m:02d}", f"{word} {MONTH_NAMES[m - 1]} {y}"
    m = r.randrange(12)
    if r.random() < 0.7:
        return f"{mod} {MONTHS[m]}", f"{word} {MONTH_NAMES[m]}, no year given (in the most recent {MONTH_NAMES[m]} that has started)"
    return f"{mod} year -1 {MONTHS[m]}", f"{word} {MONTH_NAMES[m]} last year"


def f_future(r):
    """'in two weeks', '3 days from now': one whole calendar period that far ahead."""
    unit = r.choice(["day"] * 3 + ["week"] * 3 + ["month"] * 2 + ["year"])
    n = r.choice([1, 1, 2, 2, 3, 4, 5, 6]) if unit in ("week", "month") else r.randint(2, 6) if unit == "day" else r.choice([1, 2])
    word = r.choice([str(n), NUM_WORDS[n]]) if n > 1 else "a"
    span = "that single day" if unit == "day" else f"that whole calendar {unit}"
    return f"{unit} {n}", f"in {word} {unit if n == 1 else UNIT_WORDS[unit]}, or {word} {unit if n == 1 else UNIT_WORDS[unit]} from now ({span})"


# (code date or word, name). Fixed-date holidays are plain dates; moving ones have their own word.
FIXED_HOLIDAYS = [("12-25", "Christmas (December 25; people just say 'Christmas' or 'Xmas')"),
                  ("12-25", "Christmas Day"), ("12-24", "Christmas Eve"), ("01-01", "New Year's Day"),
                  ("12-31", "New Year's Eve"), ("02-14", "Valentine's Day"), ("10-31", "Halloween"),
                  ("07-04", "the US Independence Day (July 4th)"), ("01-06", "Epiphany / Three Kings' Day (January 6)"),
                  ("05-01", "May Day / Labour Day (May 1)"), ("10-03", "German Unity Day (October 3)"),
                  ("10-01", "China's National Day (October 1)"), ("12-26", "Boxing Day / St Stephen's Day (December 26)"),
                  ("03-17", "St Patrick's Day"), ("11-01", "All Saints' Day (November 1)"),
                  ("08-15", "Independence Day in India / Liberation Day in Korea (August 15)"),
                  ("01-26", "India's Republic Day (January 26)"), ("08-17", "Indonesia's Independence Day (August 17)"),
                  ("09-02", "Vietnam's National Day (September 2)"), ("05-05", "Children's Day in Japan and Korea (May 5)")]
# Multi-day festivals with fixed dates: "around" their middle day (give or take 3 days).
FIXED_FESTIVALS = [("04-14", "Songkran, the Thai New Year water festival (April 13-15)"),
                   ("05-02", "Golden Week in Japan (April 29 to May 5)"), ("08-14", "Obon in Japan (mid-August)")]
MOVING_HOLIDAYS = [("easter", "Easter (the Easter weekend, Good Friday to Easter Monday)"),
                   ("thanksgiving", "US Thanksgiving (the fourth Thursday of November)"),
                   ("cny", "Chinese New Year / Spring Festival (the week starting on the lunar new year)"),
                   ("cny", "Lunar New Year: Seollal in Korea, Tet in Vietnam (the lunar new year holiday)"),
                   ("midautumn", "the Mid-Autumn Festival (moon festival)"),
                   ("midautumn", "Chuseok, the Korean harvest festival (lunar August 15)"),
                   ("eidfitr", "Lebaran / Idul Fitri in Indonesia (Eid al-Fitr)"),
                   ("dragonboat", "the Dragon Boat Festival"), ("ramadan", "Ramadan (the whole month)"),
                   ("eidfitr", "Eid al-Fitr (the end of Ramadan)"), ("eidadha", "Eid al-Adha")]


def _short(name: str) -> str:
    return name.split(" (")[0].split(" / ")[0].removeprefix("the ")


def f_holiday(r):
    k = r.random()
    if r.random() < 0.1:
        md, name = r.choice(FIXED_FESTIVALS)
        if k < 0.6:
            return f"around *-{md}", f"{name}, no year given, or 'last {_short(name)}' (the most recent one)"
        y = _year(r)
        return f"around {y}-{md}", f"{name} {y}"
    if r.random() < 0.5:
        md, name = r.choice(FIXED_HOLIDAYS)
        if k < 0.45:
            return f"*-{md}", f"{name}, no year given, or 'last {_short(name)}' (the most recent one, that single day)"
        if k < 0.8:
            y = _year(r)
            return f"{y}-{md}", f"{name} {y} (that single day)"
        if k < 0.9:
            return f"around *-{md}", f"around {name}, no year given (roughly that date, give or take a few days)"
        y = _year(r)
        return f"around {y}-{md}", f"around {name} {y} (give or take a few days)"
    word, name = r.choice(MOVING_HOLIDAYS)
    if k < 0.45:
        return word, f"{name}, no year given, or 'last {_short(name)}' (the most recent one that has started)"
    if k < 0.75:
        y = r.randint(2010, 2027)
        return f"{y} {word}", f"{name} in {y}"
    if k < 0.9:
        return f"year -1 {word}", f"{name} last year"
    return f"{word} 1", f"the coming {name} (the next one, not yet started)"


def f_range(r):
    k = r.random()
    if k < 0.3:
        a, b = sorted(r.sample(range(12), 2))
        return f"{MONTHS[a]} .. {MONTHS[b]}", f"between {MONTH_NAMES[a]} and {MONTH_NAMES[b]}, no year given (from the start of {MONTH_NAMES[a]} to the end of {MONTH_NAMES[b]})"
    if k < 0.5:
        a = _year(r)
        b = a + r.randint(1, 4)
        return f"{a} .. {b}", f"from {a} to {b} (start of {a} to end of {b})"
    if k < 0.65:
        a, b = sorted(r.sample(range(7), 2))
        n = r.choice([-1, 0])
        which = "last week" if n else "this week"
        return f"week {n} {WEEKDAYS[a]} .. week {n} {WEEKDAYS[b]}", f"from {DAY_NAMES[a]} to {DAY_NAMES[b]} {which}"
    if k < 0.75:
        y, m = _year(r), r.randint(1, 11)
        m2 = r.randint(m + 1, 12)
        return f"{y}-{m:02d} .. {y}-{m2:02d}", f"from {MONTH_NAMES[m - 1]} {y} to {MONTH_NAMES[m2 - 1]} {y}"
    if k < 0.85:  # day to day: "June 5 to June 8, 2024", "June 5-8"
        y, m = _year(r), r.randint(1, 12)
        d2 = _day(r, None, m)
        d = r.randint(1, max(1, d2 - 1)) if d2 > 1 else 1
        d2 = max(d2, d + 1)
        if r.random() < 0.3:
            return f"*-{m:02d}-{d:02d} .. *-{m:02d}-{d2:02d}", f"from {MONTH_NAMES[m - 1]} {d} to {MONTH_NAMES[m - 1]} {d2}, no year given (those days)"
        return f"{y}-{m:02d}-{d:02d} .. {y}-{m:02d}-{d2:02d}", f"from {MONTH_NAMES[m - 1]} {d} to {d2}, {y} (those days, both ends included)"
    code, desc = r.choice([
        ("day -1 .. now", "yesterday and today (both days)"), ("day 1 .. day 2", "tomorrow and the day after tomorrow (both days)"),
        ("day 0 .. day 1", "today and tomorrow (both days)"), ("year -1 .. year 0", "from last year to this year (both whole years)"),
        ("week -1 fri .. week 0 mon", "from last Friday to this Monday (crossing the weekend)"),
        ("week 0 fri .. week 1 mon", "from this Friday to next Monday (crossing the weekend)"),
        ("week 1 fri .. week 2 mon", "from Friday next week to the Monday after it"),
        ("week -1 fri .. week -1 sun", "from Friday to Sunday last week"), ("month -1 .. month 0", "from last month to this month"),
    ])
    return code, desc


def f_open(r):
    k = r.random()
    if k < 0.5:
        code, desc = r.choice([
            lambda: (MONTHS[(m := r.randrange(12))], f"{MONTH_NAMES[m]} (no year given)"),
            lambda: (str(y := _year(r)), f"{y}"),
            lambda: ("week -1", "last week"), lambda: ("month -1", "last month"), lambda: ("year -1", "last year"),
            lambda: (f"{(y := _year(r))}-{(m := r.randint(1, 12)):02d}", f"{MONTH_NAMES[m - 1]} {y}"),
            lambda: ("year 0", "the start of this year"), lambda: ("quarter 0", "the start of this quarter"),
            lambda: ("week 0", "the start of this week (Monday)"), lambda: ("week -1 weekend", "last weekend"),
            lambda: ("year -1 now", "this same day last year"), lambda: ("day -1", "yesterday / last night"),
            lambda: (f"day -{(n := r.randint(2, 10))}", f"{n} days ago"),
            lambda: (f"{(y := _year(r))}-{(m := r.randint(1, 12)):02d}-{(d := _day(r, y, m)):02d}", _date_text(r, y, m, d)),
        ])()
        return f"{code} .. now", f"since {desc} (from its start up to today)"
    if k < 0.6:  # from a point on, with no end: "starting tomorrow", "from next week on"
        code, desc = r.choice([("day 1 ..", "from tomorrow on (no end)"), ("week 1 ..", "from next week on (no end)"),
                               ("month 1 ..", "from next month on (no end)"), (f"day {(n := r.randint(2, 7))} ..", f"from {n} days from now on (no end)")])
        return code, desc
    if k < 0.65:
        code, desc = r.choice([("before year 0", "before this year (any time up to the end of last year)"),
                               ("before week -1", "before last week (any time up to the end of the week before last)"),
                               ("after month 0", "after this month (any time from the start of next month)"),
                               (f"before {(y := _year(r))}-{(m := r.randint(1, 12)):02d}", f"before {MONTH_NAMES[m - 1]} {y} (up to the end of the month before)")])
        return code, desc
    y = _year(r)
    if k < 0.75:
        return f"before {y}", f"before {y} (any time up to the end of {y - 1})"
    return f"after {y}", f"after {y} (any time from the start of {y + 1})"


def f_sameday(r):
    """A day pinned inside a relative year or month: 'this day last year', 'June 5 this year', 'the 3rd of last month'."""
    k = r.random()
    if k < 0.35:
        n = r.choice([-1, -1, -1, 1, -2])
        which = {-1: "last year", 1: "next year", -2: "two years ago"}[n]
        return f"year {n} now", f"this same day {which} (today's date, {which}; that single day)"
    if k < 0.7:
        n, m = r.choice([0, 0, -1, 1]), r.randint(1, 12)
        d = _day(r, None, m)
        which = {0: "this year", -1: "last year", 1: "next year"}[n]
        return f"year {n} *-{m:02d}-{d:02d}", f"{MONTH_NAMES[m - 1]} {d} {which} (that single day, even if it is still ahead)"
    n, d = r.choice([-1, -1, 0, 1]), _day(r, None, r.randint(1, 12))
    which = {-1: "last month", 0: "this month", 1: "next month"}[n]
    return f"month {n} day {d}", f"the {d}{_th(d)} of {which} (that single day)"


def f_half(r):
    """'half a month ago', 'in a year and a half': as days or calendar months."""
    code, desc = r.choice([("day -15", "half a month ago (that single day, 15 days ago)"),
                           ("day 15", "in half a month (that single day, 15 days from now)"),
                           ("month -6", "half a year ago (that calendar month, six months back)"),
                           ("month 6", "in half a year (that calendar month, six months ahead)"),
                           ("month -18", "a year and a half ago (that calendar month, 18 months back)"),
                           ("month 18", "in a year and a half (that calendar month, 18 months ahead)"),
                           ("month -30", "two and a half years ago (that calendar month, 30 months back)"),
                           ("month 30", "in two and a half years (that calendar month, 30 months ahead)")])
    return code, desc


SINGLE = [(f_day, 14), (f_week, 10), (f_month, 8), (f_year, 6), (f_quarter, 3), (f_weekday, 8), (f_weekend, 4),
          (f_month_name, 9), (f_date, 7), (f_year_abs, 5), (f_season, 3), (f_decade, 1), (f_last_n, 7),
          (f_fuzzy, 6), (f_modified, 4), (f_range, 4), (f_open, 4), (f_future, 3), (f_holiday, 4), (f_sameday, 2), (f_half, 1)]


def f_multi(r):
    """Two separate periods in one question, e.g. 'May 2025 and June 2026'."""
    (c1, d1), (c2, d2) = (r.choices([f for f, _ in SINGLE[:12]], [w for _, w in SINGLE[:12]])[0](r) for _ in range(2))
    return f"{c1} ; {c2}", f"two separate periods, in this order: first {d1}; second {d2}"


FAMILIES = SINGLE + [(f_multi, 3)]


# People also say "a month ago" for the previous period; described only as "last month",
# generated questions never used that phrasing.
ONE_AGO = {"week -1": "a week ago (the previous calendar week (Monday to Sunday))",
           "month -1": "a month ago (the previous calendar month)",
           "year -1": "a year ago (the previous calendar year)"}


def sample(r: random.Random) -> tuple[str, str, str]:
    """One (kind, code, description)."""
    family = r.choices([f for f, _ in FAMILIES], [w for _, w in FAMILIES])[0]
    code, desc = family(r)
    if code in ONE_AGO and r.random() < 0.3:
        desc = ONE_AGO[code]
    return family.__name__[2:], code, desc


if __name__ == "__main__":
    r = random.Random(0)
    for _ in range(40):
        print(*sample(r), sep=" | ")
