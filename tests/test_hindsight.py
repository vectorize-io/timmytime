from datetime import datetime

import pytest

from text2date.hindsight import extract_all, extract_temporal_constraint
from text2date.runtime import WEIGHTS

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="no trained weights")
REF = datetime(2026, 11, 17, 14, 30)


def test_same_shape_as_hindsight():
    start, end = extract_temporal_constraint("what did I decide yesterday", REF)
    assert (start, end) == (datetime(2026, 11, 16), datetime(2026, 11, 16, 23, 59, 59, 999999))


def test_no_date_and_bad_input_give_none():
    assert extract_temporal_constraint("how does the reranker work", REF) is None
    assert extract_temporal_constraint("", REF) is None
    assert extract_temporal_constraint("x", REF, tz="Not/AZone") is None  # never raises


def test_first_range_and_all_ranges():
    windows = extract_all("may 2025 and june 2026", REF)
    assert extract_temporal_constraint("may 2025 and june 2026", REF) == windows[0]
    assert windows[0][0] == datetime(2025, 5, 1)
