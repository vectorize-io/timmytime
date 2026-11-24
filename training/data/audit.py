"""Mark no-date test rows the LLM itself reads a date in ('disputed'), without looking at timmytime.

    python -m training.data.audit data/test_en_2.jsonl ...

Rows labelled 'none' by the lenient yes/no check (all but data/test_en.jsonl, whose no-date rows
passed the stricter round trip) get "disputed": true when the LLM, reading the message cold,
writes a date code. The scorer reports false dates with and without them. Model-independent, so
it can't tilt the test toward timmytime; it does also drop traps the LLM misreads.
"""

import json
import sys

from training.data import llm
from training.data.gen import read_code


def main() -> None:
    for path in sys.argv[1:]:
        rows = [json.loads(line) for line in open(path)]
        idx = [i for i, r in enumerate(rows) if r["code"] == "none"]
        reads = llm.run_all(lambda i: read_code(rows[i]["q"]), idx)
        n = 0
        for i, code in zip(idx, reads):
            rows[i]["disputed"] = bool(code) and code.strip() != "none"
            n += rows[i]["disputed"]
        with open(path, "w") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"{path}: {n} of {len(idx)} no-date rows disputed")


if __name__ == "__main__":
    main()
