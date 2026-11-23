"""Pick the confidence cut-off on the tuning set (data/tune_*.jsonl), never on the test sets.

    python -m training.model.tune

Score = exact - 3 x false-date rate: a wrong window hurts recall, a missing one changes nothing.
Prints the table per language and the chosen value for MIN_CONFIDENCE in tlm/runtime.py.
"""

import json
from datetime import date, datetime
from pathlib import Path

from tlm.resolve import resolve
from tlm.runtime import Parser

DATA = Path(__file__).resolve().parents[2] / "data"
THRESHOLDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
FALSE_WEIGHT = 3.0


def main() -> None:
    parser = Parser()
    preds = {}  # lang -> [(confidence, predicted ranges, wanted ranges)]
    for path in sorted(DATA.glob("tune_*.jsonl")):
        lang = path.stem.split("_")[1]
        out = preds.setdefault(lang, [])
        for row in map(json.loads, path.open()):
            code, conf = parser.code(row["q"])
            anchor = datetime.combine(date.fromisoformat(row["anchor"]), datetime.min.time())
            got = [[s and s.isoformat(), e and e.isoformat()] for s, e in resolve(code, anchor)]
            out.append((conf if code != "none" else 1.0, got, row["ranges"]))

    def stats(rows, th):
        exact = sum((got if conf >= th else []) == want for conf, got, want in rows) / len(rows)
        nodate = [r for r in rows if not r[2]]
        false = sum(bool(got) and conf >= th for conf, got, _ in nodate) / max(len(nodate), 1)
        return exact, false

    every = [r for rows in preds.values() for r in rows]
    print("threshold  " + "  ".join(f"{lang:>14}" for lang in preds) + "           all   score")
    best = None
    for th in THRESHOLDS:
        cells = [stats(rows, th) for rows in preds.values()]
        ex, fd = stats(every, th)
        score = ex - FALSE_WEIGHT * fd
        best = max(best or (score, th), (score, th))
        print(f"   {th:.1f}     " + "  ".join(f"{e:6.1%} /{f:5.1%}" for e, f in cells) + f"   {ex:6.1%} /{fd:5.1%}   {score:.4f}")
    print(f"\nchosen MIN_CONFIDENCE = {best[1]}  ({len(every)} tuning rows; cells are exact / false dates)")


if __name__ == "__main__":
    main()
