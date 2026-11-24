"""Score a date finder on the English test sets.

    python -m training.eval.score baseline        # hindsight's current code
    python -m training.eval.score gemma           # the LLM writing codes itself (reference only)
    python -m training.eval.score timmytime             # our model
    python -m training.eval.score timmytime --lang es   # one language's test set (data/test_es.jsonl)
    python -m training.eval.score timmytime --lang hindsight   # hindsight's own tests (data/test_hindsight.jsonl)

Sets: data/test_en*.jsonl (LLM-written, checked; _3 = extra rows of the small kinds), data/hand_en.tsv (hand-written
traps, asked on 2026-11-17), and the no-date questions of hindsight's golden corpus.
Exact = every range right, start and end. False date = a range on a no-date question.
"""

import json
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from timmytime.resolve import resolve

DATA = Path(__file__).resolve().parents[2] / "data"
HAND_ANCHOR = date(2026, 11, 17)


def _ranges(code: str, anchor: date) -> list:
    return [[s and s.isoformat(), e and e.isoformat()] for s, e in resolve(code, datetime.combine(anchor, datetime.min.time()))]


def load_lang_set(lang: str) -> dict[str, list[dict]]:
    """gen-xx: the first set (not for English, see load_sets); big-xx: ~3k rows covering every kind."""
    out = {}
    for name, path in ((f"gen-{lang}", DATA / f"test_{lang}.jsonl"), (f"big-{lang}", DATA / f"test_{lang}_big.jsonl")):
        if path.exists() and not (lang == "en" and name == "gen-en"):
            out[name] = [json.loads(line) for line in path.open()]
    return out


def load_sets() -> dict[str, list[dict]]:
    def rows(name):
        path = DATA / name
        return [json.loads(line) for line in path.open()] if path.exists() else []

    # gen: the natural mix (first set + its 2026-11-19 extension); gen-kinds: extra rows of the small kinds.
    sets = {"gen": rows("test_en.jsonl") + rows("test_en_2.jsonl"), "gen-kinds": rows("test_en_3.jsonl")}
    hand = []
    for line in (DATA / "hand_en.tsv").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            q, code = line.split("\t")
            hand.append({"q": q, "code": code, "kind": "none" if code == "none" else "hand",
                         "anchor": HAND_ANCHOR.isoformat(), "ranges": _ranges(code, HAND_ANCHOR)})
    sets["hand"] = hand
    sys.path.insert(0, str(DATA.parent))
    from training.tests.baseline.query_analyzer_corpus import FALSE_POSITIVE_TRAPS, NON_TEMPORAL_EN, TZ_ABBREV_TRAPS

    sets |= load_lang_set("en")
    sets["golden-none"] = [{"q": q, "code": "none", "kind": "none", "anchor": HAND_ANCHOR.isoformat(), "ranges": []}
                           for q in dict.fromkeys(NON_TEMPORAL_EN + TZ_ABBREV_TRAPS + FALSE_POSITIVE_TRAPS)]
    return sets


def predictor(name: str):
    """A function (question, anchor date) -> ranges as [[start, end], ...]."""
    if name == "baseline":
        from training.eval.baseline.temporal_extraction import extract_temporal_constraint

        def run(q, anchor):
            r = extract_temporal_constraint(q, reference_date=datetime.combine(anchor, datetime.min.time()))
            return [] if r is None else [[r[0].date().isoformat(), r[1].date().isoformat()]]
        return run
    if name == "gemma":
        from training.data.gen import read_code

        return lambda q, anchor: _ranges(read_code(q), anchor)
    if name == "timmytime":
        from timmytime.runtime import Parser

        parser = Parser()
        return lambda q, anchor: [[s and s.isoformat(), e and e.isoformat()] for s, e in parser.parse(q, anchor)]
    raise SystemExit(f"unknown predictor {name}")


def score(run, rows: list[dict], workers: int = 1) -> dict:
    if workers > 1:
        from training.data.llm import run_all

        preds = run_all(lambda r: run(r["q"], date.fromisoformat(r["anchor"])), rows, workers)
        times = []
    else:
        preds, times = [], []
        for r in rows:
            t = time.perf_counter()
            preds.append(run(r["q"], date.fromisoformat(r["anchor"])))
            times.append(time.perf_counter() - t)
        times.sort()
    by_kind = defaultdict(lambda: [0, 0])
    exact = first = dated = false = nodate = 0
    clean_rows = clean_exact = clean_nodate = clean_false = 0  # leaving out rows training.data.audit marked disputed
    misses = []
    for row, pred in zip(rows, preds):
        pred = pred or []
        ok = pred == row["ranges"]
        by_kind[row["kind"]][0] += ok
        by_kind[row["kind"]][1] += 1
        exact += ok
        if row["ranges"]:
            dated += 1
            first += bool(pred) and pred[0] == row["ranges"][0]
        else:
            nodate += 1
            false += bool(pred)
        if not row.get("disputed"):
            clean_rows += 1
            clean_exact += ok
            if not row["ranges"]:
                clean_nodate += 1
                clean_false += bool(pred)
        if not ok:
            misses.append((row["q"], row["code"], row["ranges"], pred))
    return {"clean_n": clean_rows, "clean_exact": clean_exact / max(clean_rows, 1),
            "clean_false": clean_false / max(clean_nodate, 1),
            "n": len(rows), "exact": exact / len(rows), "first_range": first / max(dated, 1),
            "false_date": false / max(nodate, 1), "kinds": {k: v[0] / v[1] for k, v in sorted(by_kind.items())},
            "kind_n": {k: v[1] for k, v in sorted(by_kind.items())}, "misses": misses,
            "p50_ms": times[len(times) // 2] * 1e3 if times else 0, "p95_ms": times[int(len(times) * 0.95)] * 1e3 if times else 0}


def main() -> None:
    name = sys.argv[1]
    run = predictor(name)
    show = "-v" in sys.argv
    lang = sys.argv[sys.argv.index("--lang") + 1] if "--lang" in sys.argv else "en"
    if lang == "hindsight":  # hindsight's own analyzer tests and golden corpus, as timmytime codes
        all_sets = {"hindsight": [json.loads(line) for line in (DATA / "test_hindsight.jsonl").open()]}
    elif lang == "en":
        all_sets = load_sets()
        all_sets["all-gen (per kind)"] = all_sets["gen"] + all_sets["gen-kinds"]
    else:
        all_sets = load_lang_set(lang)
    for set_name, rows in all_sets.items():
        if not rows:
            continue
        t = time.perf_counter()
        s = score(run, rows, workers=32 if name == "gemma" else 1)
        print(f"\n== {name} on {set_name}: n={s['n']} exact={s['exact']:.1%} first-range={s['first_range']:.1%} "
              f"false-date={s['false_date']:.1%}  | without disputed rows (n={s['clean_n']}): exact={s['clean_exact']:.1%} "
              f"false-date={s['clean_false']:.1%}  p50={s['p50_ms']:.2f}ms p95={s['p95_ms']:.2f}ms  ({time.perf_counter() - t:.1f}s)")
        print("   " + "  ".join(f"{k}={v:.0%}/{s['kind_n'][k]}" for k, v in s["kinds"].items()))
        for q, code, want, got in s["misses"][: 40 if show else 0]:
            print(f"   MISS {code:24} want={want} got={got} | {q[:80]}")


if __name__ == "__main__":
    main()
