"""The numpy runner on real weights: odd input never breaks it, and dates are days in the time zone."""

from datetime import date

import tlm
from tlm.runtime import Parser


def test_odd_input_never_crashes():
    parser = Parser()
    for text in ["", " ", "\x00\x00", "🧠" * 500, "a" * 10_000, "\ud800 broken surrogate", "yesterday" * 100,
                 "SELECT * FROM t WHERE d > '2024-01-01'", "\n\t\r", "٣ أيام", "1" * 300]:
        ranges = parser.parse(text, None)
        assert isinstance(ranges, list)


def test_date_anchor_is_a_day_in_tz():
    for tz in ("UTC", "America/New_York", "Asia/Tokyo"):
        assert tlm.parse("what did I do yesterday", date(2026, 11, 20), tz) == [(date(2026, 11, 19), date(2026, 11, 19))]


def test_runtime_never_imports_training():
    """The package ships tlm/ only: nothing in it may need the training code."""
    import re
    from pathlib import Path

    for path in (Path(tlm.__file__).parent).glob("*.py"):
        assert not re.search(r"^\s*(from|import) training\b", path.read_text(), re.M), path
