"""Short English queries from templates: an id number next to a real date ("port 9077 and also last
Tuesday"), year 0000 (no date), 100+ days ago, far ISO dates (1890-03-05, 2150-07-01). Training data only.

    python -m training.data.en_templates --n 6000 --seed 1 --out data/train_en_tpl.jsonl
"""

import argparse
import calendar
import json
import random
from datetime import date, datetime, timedelta

from timmytime.resolve import resolve

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
NOISE = ["port {}", "listening on {}", "listening in {}", "PR {}", "job {}", "buffer size {}", "room {}", "ticket {}", "chunk size {}",
         "build {}", "order {}", "issue #{}", "error code {}", "page {}"]
DATES = [("last tuesday", "tue"), ("last friday", "fri"), ("yesterday", "day -1"), ("last week", "week -1"), ("last month", "month -1"),
         ("two days ago", "day -2"), ("this morning", "day 0"), ("tonight", "day 0"), ("last night", "day -1"), ("tomorrow morning", "day 1")]
JOIN = [" and also ", " and ", ", ", " - what happened "]


def row(r):
    k = r.random()
    if k < 0.3:
        n = r.choice([r.randint(1000, 9999), r.randint(10, 999)])
        noise = r.choice(NOISE).format(n)
        w, c = r.choice(DATES)
        return (noise + r.choice(JOIN) + w) if r.random() < 0.6 else (w + r.choice(JOIN) + noise), c
    if k < 0.4:
        m = r.randrange(12)
        return r.choice(["what happened in {} 0000", "notes from {} 0000", "{} 0000 report", "cosa accadde a {} 0000"]).format(MONTHS[m]), "none"
    if k < 0.5:
        n = r.choice([100, 100, 120, 150, 180, 200, 250, 300, 365, r.randint(100, 365)])
        if r.random() < 0.5:
            return r.choice(["{} days ago", "what happened {} days ago", "notes from {} days ago", "{} days back"]).format(n), f"day -{n}"
        return r.choice(["in {} days", "{} days from now", "the plan for {} days from now"]).format(n), f"day {n}"
    if k < 0.8:
        y = r.choice([r.randint(1800, 1979), r.randint(2030, 2199), r.choice(range(2030, 2200, 10))])
        m = r.randint(1, 12)
        d = r.randint(1, calendar.monthrange(y, m)[1])
        p = r.choice(["notes from {}", "the roadmap milestone on {}", "what happened on {}", "the record dated {}", "{} entry", "events of {}",
                      "{}", "deadline {}", "the launch on {}", "plan for {}", "due {}", "meeting on {}"])
        return p.format(f"{y}-{m:02d}-{d:02d}"), f"{y}-{m:02d}-{d:02d}"
    y = r.choice([r.randint(1800, 1979), r.randint(2030, 2199)])
    m = r.randint(1, 12)
    p = r.choice(["notes from {} {}", "what happened in {} {}", "the plan for {} {}", "letters written in {} {}"])
    return p.format(MONTHS[m - 1].capitalize(), y), f"{y}-{m:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = random.Random(a.seed)
    seen, rows = set(), []
    while len(rows) < a.n:
        q, code = row(r)
        if q in seen:
            continue
        anchor = date(2015, 1, 1) + timedelta(days=r.randrange(365 * 13))
        ranges = resolve(code, datetime(anchor.year, anchor.month, anchor.day, 12))
        seen.add(q)
        rows.append({"q": q, "code": code, "kind": "en_tpl", "lang": "en", "anchor": anchor.isoformat(),
                     "ranges": [[x.isoformat() if x else None for x in rg] for rg in ranges]})
    with open(a.out, "w") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
