"""Hard-example mining: keep the pool rows the current model gets wrong.

    python -m training.eval.mine data/pool_*.jsonl > data/train_hard.jsonl

The pool is written with the training prompts (never the test ones) and already passed the
LLM check, so its labels are as good as the rest of the training data. Training on the rows the
model fails targets its weak spots instead of generating traps blind.
"""

import json
import sys
from collections import Counter
from datetime import date, datetime

from text2date.resolve import resolve
from text2date.runtime import Parser


def main() -> None:
    parser = Parser()
    kept, seen = Counter(), Counter()
    for path in sys.argv[1:]:
        for line in open(path):
            row = json.loads(line)
            anchor = datetime.combine(date.fromisoformat(row["anchor"]), datetime.min.time())
            want = [[s and s.isoformat(), e and e.isoformat()] for s, e in resolve(row["code"], anchor)]
            got = [[s and s.isoformat(), e and e.isoformat()] for s, e in parser.parse(row["q"], anchor)]
            seen[row["lang"], row["kind"]] += 1
            if got != want:
                kept[row["lang"], row["kind"]] += 1
                print(json.dumps(row, ensure_ascii=False))
    for key in sorted(seen):
        print(f"{key[0]} {key[1]:12} {kept[key]:5} wrong of {seen[key]:6}", file=sys.stderr)


if __name__ == "__main__":
    main()
