"""Clean training files in place: drop questions that are in a test set, apply English label rules.

    python -m training.data.clean data/train_*.jsonl

Rules (the conventions in timmytime/meanings.py; the LLM reads these phrases both ways):
Drops no-date rows with a weekday + part of day ("Friday Night Lights"): real users mean a day.
"last Monday" (no "week" in the message) = mon, not week -1 mon; "last summer" = summer, not year -1 summer.
"""

import json
import re
import sys
import zlib

from training.data.gen import has_gloss
from training.eval.score import load_lang_set, load_sets

WD = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
SEASON = {"summer": "summer", "winter": "winter", "spring": "spring", "autumn": "autumn", "fall": "autumn"}


def _month_name_trap_dropped(r: dict) -> bool:
    """Keep half of the no-date rows that mention an English month name (people called June,
    'the may report'): with all of them the model said 'no date' to plain 'june weather forecast'.
    Which half is fixed by a hash of the question, so cleaning twice drops the same rows."""
    if r["code"] != "none" or not re.search(r"\b(" + "|".join(MONTHS_EN) + r")\b", r["q"], re.I):
        return False
    return zlib.crc32(r["q"].encode()) % 2 == 0


def _show_name_trap(r: dict) -> bool:
    """No-date rows like 'Friday Night Lights vibe': they taught the model that 'friday night' is never a date."""
    return r["code"] == "none" and re.search(rf"\b({WD})\s+(morning|afternoon|evening|night)", r["q"], re.I) is not None


MONTHS_EN = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
             "october", "november", "december"]
SHORT = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def relabel(q: str, code: str) -> str:
    if not re.search(r"\bweek", q, re.I):
        for m in re.finditer(rf"\blast ({WD})\b", q, re.I):
            code = re.sub(rf"\bweek -1 ({m[1][:3].lower()})\b", r"\1", code)
    # "this June" (no year) = June of this year, also when it is still ahead ("remind me ... this June").
    for m in re.finditer(r"\bthis (" + "|".join(MONTHS_EN) + r")\b", q, re.I):
        mon = SHORT[MONTHS_EN.index(m[1].lower())]
        code = " ; ".join(f"year 0 {mon}" if part.strip() == mon else part.strip() for part in code.split(";"))
    for m in re.finditer(r"\blast (summer|winter|spring|autumn|fall)\b", q, re.I):
        code = re.sub(rf"\byear -1 {SEASON[m[1].lower()]}\b", SEASON[m[1].lower()], code)
    return code


DROP_MONTH_TRAPS = "--keep-month-traps" not in sys.argv


def main() -> None:
    from training.data.gen import LANGS
    sets = load_sets() | {k: v for lang in LANGS if lang != "en" for k, v in load_lang_set(lang).items()}
    sets["hindsight"] = [json.loads(line) for line in open("data/test_hindsight.jsonl")]
    test = {r["q"].lower().strip() for rows in sets.values() for r in rows}
    for path in [a for a in sys.argv[1:] if not a.startswith("--")]:
        rows = [json.loads(line) for line in open(path)]
        keep = [r for r in rows if r["q"].lower().strip() not in test and not _show_name_trap(r)
                and not (r["lang"] != "en" and has_gloss(r["q"]))
                and not (DROP_MONTH_TRAPS and _month_name_trap_dropped(r))]
        changed = 0
        for r in keep:
            new = relabel(r["q"], r["code"])
            changed += new != r["code"]
            r["code"] = new
        with open(path, "w") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)
        print(f"{path}: {len(rows)} rows, {len(rows) - len(keep)} test overlaps dropped, {changed} relabelled")


if __name__ == "__main__":
    main()
