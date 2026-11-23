import pytest

from training.eval.baseline.query_analyzer import DateparserQueryAnalyzer


@pytest.fixture(scope="session")
def query_analyzer():
    return DateparserQueryAnalyzer()
