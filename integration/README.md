# Plugging tlm into hindsight

`hindsight-tlm-analyzer.patch` applies to hindsight `origin/main` (checked against f6038a742 era main,
2026-11-19) with `patch -p1` from the repo root. It adds `TlmQueryAnalyzer` next to
`DateparserQueryAnalyzer` and an opt-in switch:

    HINDSIGHT_API_QUERY_ANALYZER=tlm     # default stays dateparser

The patch also adds `tests/test_tlm_query_analyzer.py`.

Same contract as today: one window per query (tlm's first range), whole days, naive datetimes.
tlm has to be importable (`pip install -e ~/dev/tlm` until it is packaged). It needs numpy only,
so it also runs on hindsight-api-slim. Tested on 2026-11-20 on a local branch `tlm/query-analyzer` (a git
worktree from hindsight origin/main, in this session's scratchpad): the new tests plus
`tests/test_query_analyzer.py` and `tests/test_temporal_extraction.py` (golden file unchanged) gave
614 passed, 1 skipped. The latency/wall-clock tests were left out. Nothing committed or pushed.
