"""The calendar code gives the same windows as hindsight's old hand-written rules, where they overlap."""

from datetime import datetime

import pytest

from text2date.resolve import DateRange, resolve
from training.eval.baseline.temporal_extraction import extract_temporal_constraint

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
