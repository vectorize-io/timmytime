"""The ported baseline must answer exactly like hindsight main did.

Golden file copied from hindsight-api-slim/tests/data/query_analyzer_golden.json.
"""

import json
from datetime import datetime
from pathlib import Path

from tests.baseline.query_analyzer_corpus import build_corpus
from tlm.baseline.query_analyzer import DateparserQueryAnalyzer
from tlm.baseline.temporal_extraction import extract_temporal_constraint

GOLDEN_PATH = Path(__file__).parent / "data" / "query_analyzer_golden.json"
REFERENCE_DATES = [
    datetime(2026, 8, 12, 14, 30, 45, 123456),
    datetime(2026, 1, 1, 0, 0, 0),
    datetime(2026, 12, 31, 23, 59, 59),
    datetime(2024, 2, 29, 12, 0, 0),
    datetime(2026, 8, 3, 9, 0, 0),
    datetime(2026, 8, 9, 18, 0, 0),
]


def test_golden_corpus_unchanged() -> None:
    golden = json.loads(GOLDEN_PATH.read_text())
    analyzer = DateparserQueryAnalyzer()
    diffs = []
    for query, _ in build_corpus():
        for ref in REFERENCE_DATES:
            key = f"{ref.isoformat()}\x1f{query}"
            r = extract_temporal_constraint(query, reference_date=ref, analyzer=analyzer)
            actual = None if r is None else [r[0].isoformat(), r[1].isoformat()]
            if key not in golden:
                diffs.append(f"no golden value: {key!r}")
            elif golden[key] != actual:
                diffs.append(f"{key!r}: golden={golden[key]} actual={actual}")
    assert not diffs, f"{len(diffs)} differences:\n" + "\n".join(diffs[:25])
