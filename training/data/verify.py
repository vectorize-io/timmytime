"""Second opinion on rows labelled 'none' that the model dates.

    python -m training.data.verify data/hard_en_none.jsonl data/hard_en_none.verified.jsonl

The yes/no check that labels no-date rows lets some dated text through ("feeling meh today").
For each row the LLM reads the message cold (training.data.gen.read_code):
- the LLM sees no date: a real trap, kept as 'none';
- the LLM and the model read the same dates: the label was wrong, relabelled with that code;
- they disagree: dropped.
"""

import json
import sys
from datetime import date, datetime

from training.data import llm
from training.data.gen import read_code, same_meaning
from timmytime.resolve import resolve
from timmytime.runtime import Parser


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    rows = [json.loads(line) for line in open(src)]
    parser = Parser()
    reads = llm.run_all(lambda r: read_code(r["q"]), rows)
    out, counts = [], {"trap kept": 0, "relabelled": 0, "dropped": 0}
    for row, gemma in zip(rows, reads):
        mine, _ = parser.code(row["q"])
        if gemma is None:
            counts["dropped"] += 1
        elif gemma == "none" or same_meaning(gemma, "none"):
            out.append(row)
            counts["trap kept"] += 1
        elif same_meaning(gemma, mine):
            anchor = datetime.combine(date.fromisoformat(row["anchor"]), datetime.min.time())
            ranges = [[s and s.isoformat(), e and e.isoformat()] for s, e in resolve(mine, anchor)]
            if ranges:
                out.append(row | {"code": mine, "kind": "relabelled", "ranges": ranges})
                counts["relabelled"] += 1
            else:
                counts["dropped"] += 1
        else:
            counts["dropped"] += 1
    with open(dst, "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    print(counts)


if __name__ == "__main__":
    main()
