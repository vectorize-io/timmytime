"""The numpy runner on real weights: odd input never breaks it, and dates are days in the time zone."""

from datetime import date

import text2date
from text2date.runtime import Parser


def test_odd_input_never_crashes():
    parser = Parser()
    for text in ["", " ", "\x00\x00", "🧠" * 500, "a" * 10_000, "\ud800 broken surrogate", "yesterday" * 100,
                 "SELECT * FROM t WHERE d > '2024-01-01'", "\n\t\r", "٣ أيام", "1" * 300]:
        ranges = parser.parse(text, None)
        assert isinstance(ranges, list)


def test_date_anchor_is_a_day_in_tz():
    for tz in ("UTC", "America/New_York", "Asia/Tokyo"):
        assert text2date.parse("what did I do yesterday", date(2026, 11, 20), tz) == [(date(2026, 11, 19), date(2026, 11, 19))]


def test_runtime_never_imports_training():
    """The package ships text2date/ only: nothing in it may need the training code."""
    import re
    from pathlib import Path

    for path in (Path(text2date.__file__).parent).glob("*.py"):
        assert not re.search(r"^\s*(from|import) training\b", path.read_text(), re.M), path


def test_long_text_is_read_in_pieces():
    from text2date.resolve import DateRange
    from text2date.runtime import MAX_BYTES, _chunks

    filler = "They prefer functional patterns and asked that the reranker stay on the local cross-encoder. " * 6
    for text in (filler, "x" * 900, "上周" * 300, " ".join(["word"] * 400)):
        pieces = _chunks(text)
        assert all(0 < len(p.encode()) <= MAX_BYTES for p in pieces)
    assert text2date.parse(filler + "The migration ran on 2026-06-10.", date(2026, 8, 12)) == [
        DateRange(date(2026, 6, 10), date(2026, 6, 10))]
