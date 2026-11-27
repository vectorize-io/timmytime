"""Every row of the README's "What it understands" table, run against the shipped model."""

from datetime import date
from pathlib import Path

import pytest

import text2date

README = (Path(__file__).parent.parent / "README.md").read_text()
TABLE = README.split("<!-- examples:start -->")[1].split("<!-- examples:end -->")[0]
ROWS = [[c.strip() for c in line.strip("|").split("|")] for line in TABLE.strip().splitlines()[2:]]
TODAY = date(2026, 11, 18)


def _fmt(ranges) -> str:
    if not ranges:
        return "no date"
    one = lambda r: (r.start.isoformat() if r.start else "…") + (
        "" if r.start == r.end else " → " + (r.end.isoformat() if r.end else "…"))
    return " ; ".join(one(r) for r in ranges)


def test_table_is_there():
    assert len(ROWS) > 90 and all(len(r) == 4 for r in ROWS)


@pytest.mark.parametrize("kind,question,tz,want", ROWS, ids=[r[1] for r in ROWS])
def test_readme_row(kind, question, tz, want):
    assert _fmt(text2date.parse(question, TODAY, tz or "UTC")) == want
