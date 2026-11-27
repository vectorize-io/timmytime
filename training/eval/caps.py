"""Exact match on the big test sets as written and in ALL CAPS (issue #3).

    TEXT2DATE_WEIGHTS=... python -m training.eval.caps en es de
"""

import json
import sys

from training.eval.score import DATA, predictor, score


def main():
    run = predictor("text2date")
    for lang in sys.argv[1:]:
        rows = [json.loads(line) for line in (DATA / f"test_{lang}_big.jsonl").open()][:800]
        a, b = score(run, rows)["exact"], score(run, [dict(r, q=r["q"].upper()) for r in rows])["exact"]
        print(f"{lang}: as written {a:.1%}, ALL CAPS {b:.1%}")


if __name__ == "__main__":
    main()
