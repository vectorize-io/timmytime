"""The previous week, month or year must also be written as "a <unit> ago" (issue #2).

Described only as "last month", generated questions never said "a month ago" or "one year ago",
and the model read those as other periods.
"""

import random
from collections import defaultdict

from training.data import en_templates
from training.data.meanings import sample


def test_previous_period_is_also_described_as_one_unit_ago():
    r = random.Random(0)
    descs = defaultdict(set)
    for _ in range(30_000):
        _, code, desc = sample(r)
        descs[code].add(desc)
    for unit in ("week", "month", "year"):
        assert any(f"a {unit} ago" in d for d in descs[f"{unit} -1"]), unit


def test_templates_write_one_unit_ago():
    r = random.Random(0)
    rows = {en_templates.row(r) for _ in range(20_000)}
    for unit in ("week", "month", "year"):
        assert any(f"a {unit} ago" in q and code == f"{unit} -1" for q, code in rows), unit
