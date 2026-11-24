"""Generate labelled questions with an LLM: meaning first, words second.

    python -m training.data.gen --split test  --lang en --n 1000 --seed 1 --out data/test_en.jsonl
    python -m training.data.gen --split train --lang en --n 30000 --seed 2 --out data/train_en.jsonl

Code picks a meaning; the LLM writes messages that say exactly that; a second
call checks each one and drops what doesn't match. Test and train use different
prompts, personas and seeds so the test set isn't a copy of the training style.
"""

import argparse
import json
import re
import random
import zlib
from datetime import date, datetime, timedelta
from pathlib import Path

from training.data import llm
from training.data.meanings import sample
from timmytime.resolve import resolve

LANGS = {"en": "English", "es": "Spanish", "de": "German", "zh": "Chinese (simplified)", "ar": "Arabic",
         "ja": "Japanese", "ko": "Korean", "hi": "Hindi (Devanagari, or sometimes Hinglish in Latin letters)",
         "id": "Indonesian", "vi": "Vietnamese", "th": "Thai", "it": "Italian", "fr": "French", "ru": "Russian",
         "pt": "Portuguese (Brazilian or European)", "zh-hant": "Chinese (traditional characters, as written in Taiwan or Hong Kong)"}

TOPICS = ["work meetings", "a project at work", "health and doctor visits", "workouts", "travel", "family",
          "friends", "money and bills", "shopping", "cooking", "a side project", "code and deployments",
          "emails", "notes I took", "decisions I made", "ideas", "books and films", "studying", "the kids",
          "home repairs", "a car", "the dog", "job interviews", "customers", "bugs and incidents", "a conference",
          "my mood and journal", "phone calls", "a trip abroad", "a birthday party", "rent and the landlord"]
STYLES = ["a normal question", "a short keyword search with no full sentence", "a command like 'show me' or 'find'",
          "casual chat, all lowercase", "with a typo or two", "long and rambling, 20 to 30 words",
          "very short, 2 to 4 words", "formal and polite", "texting style with abbreviations",
          "starting with the time phrase", "ending with the time phrase", "the time phrase in the middle"]
PERSONAS = ["a nurse working shifts", "a startup founder", "a retired teacher", "a university student",
            "a software engineer on call", "a parent of three", "a freelance designer", "a sales manager",
            "a researcher", "a small shop owner", "a marathon runner", "someone learning a language"]
# English words that look like dates but aren't one here; used for no-date messages.
TRAPS_EN = ["the modal verb 'may'", "the verb 'march' (to walk)", "a person named April, June or August",
            "a version number like 2.1 or v2024.3", "a price like $2019 or 1,500 euros", "a port or ticket number like 8080 or #2023",
            "a duration like 'a 3-day workshop' or 'it took 2 weeks'", "a frequency like 'every Monday' or 'weekly report'",
            "'day job' or 'day one' or 'a big day'", "'second' meaning number two", "an age like 'my 30 year old brother'",
            "a time of day with no day, like 'the 3pm meeting'", "a quantity like '12 months of data' or 'a year's supply'",
            "the word 'date' as in a romantic date or a fruit", "the word 'now' or 'current' with no time period",
            "a product name with a year in it, like 'Office 2019' or 'Model 3'", "'last' meaning final, like 'the last slide'",
            "'next' meaning following, like 'the next step'",
            "a person named May, June, April or August as the subject of the sentence", "a newspaper or show like 'the Sunday Times' or 'Friday Night Lights'",
            "a pet or person named Friday, Sunday or Summer", "a season word used for a thing, like 'winter tires', 'summer intern', 'spring rolls'",
            "a product with a year in its name, like 'Office 2021' or 'Windows 2000'", "a code snippet, file path or URL with numbers in it",
            "a single short word that could be a date abbreviation, like 'mar', 'sun', 'sat' or 'wed'", "an amount like '2024 steps' or '1990 calories'",
            "a price or amount with a currency word, like '2019 usd', '1500 dollars' or '450 eur'",
            "a room, bed, patient, order or invoice number, like 'pt in 205', 'bed 12', 'order 2023'",
            "a model or product number, like 'Forerunner 955', 'iPhone 15', 'RTX 4090'",
            "a person named May, June, April or August doing something in the sentence", "'spring' as in a metal spring or a water spring",
            "'fall' meaning to fall down", "a street or room number like '1999 Main St' or 'room 2024'",
            "a config value, size or limit in a technical note, like 'chunk size of 2000 tokens', 'timeout 1990 ms' or 'max 2048 rows'",
            "an idiom with a time word that points to no date, like 'yesterday's news' or 'not born yesterday'",
            "a brand, app, game, song or show named with a time word, like 'Today show', 'Tomorrowland' or 'Monday.com'",
            "no trap, just a plain question"]

# Trap types for other languages: the English list is full of English-only words.
TRAPS_ANY = ["a word in {lang} that also names a month, weekday or season but is used in another sense here",
             "a person's name that is also a month or weekday name in {lang}", "a price or amount of money with a 4-digit number",
             "a version, port, room, ticket or order number", "a duration, like 'it took 3 days' or 'a 2-week course'",
             "a frequency, like 'every Monday' or 'weekly'", "a time of day with no day", "an age, like '30 years old'",
             "a product name with a year or number in it", "'last' meaning final, or 'next' meaning following",
             "a quantity of time units that is not a period, like '12 months of data'", "a code snippet or URL with numbers",
             "a config value, size or limit in a technical note, like a chunk size of 2000 tokens or a 1990 ms timeout",
             "an idiom or set phrase in {lang} that contains a time word but points to no date (like 'yesterday's news' meaning old news)",
             "a brand, app, game, song, show or person's name in {lang} that contains a time word",
             "no trap, just a plain question"]
# Chinese words that contain date characters but are no date.
TRAPS_ZH = ["a famous person's name starting with 周, like 周杰伦 or 周星驰, right after 上/下/这 (上周杰伦的歌)",
            "food or things with 月 or 年 in the name, like 月饼, 月亮, 年糕, 年货, 月票, 年终奖",
            "a game, app or show named with 明日/今日/后天, like 明日方舟, 明日之后, 今日头条, 后天 (a film)",
            "a word with 后天/先天 meaning acquired/innate, like 后天免疫 or 先天不足",
            "a frequency like 每年, 每周, 每天, 每个月, 每年上半年 (every year, every week)",
            "a duration like 三天时间, 两周的课程, 一年的数据", "a price, amount or count with a 4-digit number like 2024元 or 1990步"]

# Two prompt families: train and test never share wording.
WRITE = {
    "train": (
        "You write realistic messages that people type to their personal AI memory assistant, "
        "which remembers their notes, chats, emails, meetings and life events.",
        "Write {k} different messages in {lang} about {topic}. Every message must refer to this time period:\n"
        "  {desc}\n"
        "The words in brackets only explain the meaning; don't copy them. Say it the way a real person would, "
        "so a reader gets exactly that period, no broader and no narrower. No other dates or times in the message. "
        "Style: {style}. Vary the wording between messages. If the period has two parts, every message mentions both.\n"
        "Reply with only a JSON array of {k} strings.",
    ),
    "test": (
        "You are {persona}. You use an assistant app that remembers everything you tell it.",
        "Type {k} things you might ask or tell the app in {lang}. Each one has to be about this exact time: "
        "{desc}. (Explanations in parentheses are for you, don't repeat them.) Mention no other time or date. "
        "Write the way you really type: {style}. Make the {k} quite different from each other. "
        "If the time has two parts, each message names both.\n"
        "Output: a JSON list of {k} strings, nothing else.",
    ),
}
WRITE_NONE = {
    "train": (
        WRITE["train"][0],
        "Write {k} different messages in {lang} about {topic}, with NO reference to any date, day or time period. "
        "Include this on purpose: {trap}. Style: {style}.\nReply with only a JSON array of {k} strings.",
    ),
    "test": (
        WRITE["test"][0],
        "Type {k} things you might ask or tell the app in {lang}. None of them may point to a date, day or period of time. "
        "Work this into them: {trap}. Write the way you really type: {style}.\nOutput: a JSON list of {k} strings, nothing else.",
    ),
}
# The check: the LLM reads the message cold and writes the code; the row stays only
# if that code gives the same dates as the label. Rules match timmytime/meanings.py.
READ = """Read a message someone sent to their memory assistant and write the time period it refers to, as one line of a tiny code.

Codes (N is a signed whole number; 0 = the current one, -1 = the previous one, 1 = the next one):
  none                      no date or time period at all. Durations ("took 2 weeks"), frequencies ("every Monday",
                            "weekly"), times of day alone ("3pm"), versions, prices, ages, "last" meaning final,
                            "next" meaning following, words like may/march/fall/spring/second in other senses: none
                            names, brands, games, songs and idioms that contain a time word (周杰伦, 月饼, 明日方舟,
                            "yesterday's news"), config values ("chunk size of 2000"), frequencies (每年 = every year): none
  day N                     today = day 0, yesterday = day -1, day before yesterday = day -2, 5 days ago = day -5, tomorrow = day 1,
                            "in 3 days" = day 3
  week N / month N / quarter N / year N    calendar periods: last week = week -1, this month = month 0,
                            3 months ago = month -3, next year = year 1, last quarter = quarter -1,
                            "in two weeks" = week 2, "in 3 months" = month 3
  mon tue wed thu fri sat sun   "last Friday" or "on Friday" about the past = fri; "this coming Friday" = fri 1
  week N mon                a weekday of a given week: "Monday last week" = week -1 mon, "this Friday" = week 0 fri
  week N weekend            last weekend = week -1 weekend, this weekend = week 0 weekend
  jan feb mar apr may jun jul aug sep oct nov dec   a month with no year ("in March") = mar
  year N mar                "March last year" = year -1 mar, "March this year" = year 0 mar
  YYYY  YYYY-MM  YYYY-MM-DD  absolute: 2019, 2024-03 (March 2024), 2024-03-12. US numeric dates are month/day/year
  *-MM-DD                   a day with no year: "June 4th" = *-06-04
  spring summer autumn winter   with no year ("last summer") = summer; "summer 2024" = 2024 summer; "summer last year" = year -1 summer
  q1 q2 q3 q4               a quarter with no year = q3; "Q3 2024" = 2024 q3
  1990s                     a decade
  last N unit / next N unit     "in the last 6 months" = last 6 month, "past two weeks" = last 2 week, "next 3 days" = next 3 day
  recent  otherday  while  longago   recently/lately; the other day; a while ago; a long time ago
  couple day / few day / couple week / few week / couple month / few month / couple year / few year   "a few weeks ago" = few week
  early X / mid X / late X / h1 X / h2 X   part of a period: "early last year" = early year -1, "late March 2024" = late 2024-03,
                            "mid March" (no year) = mid mar, "end of last month" = late month -1,
                            "first half of 2023" = h1 2023
  around X                  roughly a date: "around March 12 2024" = around 2024-03-12
  holidays                  fixed-date ones are dates: Christmas = *-12-25, Christmas 2022 = 2022-12-25,
                            New Year's Eve = *-12-31, "around Christmas" = around *-12-25.
                            Moving ones: easter thanksgiving cny (Chinese New Year) midautumn dragonboat ramadan eidfitr eidadha;
                            "last Easter" = easter, "Easter 2022" = 2022 easter, "Ramadan last year" = year -1 ramadan,
                            "the coming Eid al-Adha" = eidadha 1. Lunar New Year / Seollal / Tet = cny, Chuseok = midautumn,
                            Lebaran = eidfitr. Multi-day fixed festivals: Golden Week = around *-05-02,
                            Songkran = around *-04-14, Obon = around *-08-14. Birthdays and personal events: none
  X .. Y                    a range: "between May and July" = may .. jul, "from 2019 to 2021" = 2019 .. 2021
  X .. now                  since: "since March" = mar .. now, "since last week" = week -1 .. now, "since 3 days ago" = day -3 .. now,
                            "since this day last year" = year -1 now .. now; "from tomorrow on" = day 1 .. (open end)
  year N now / year N *-MM-DD   "this day last year" = year -1 now, "June 5 this year" = year 0 *-06-05 (even if still ahead)
  month N day D             "the 3rd of last month" = month -1 day 3, "the 10th of next month" = month 1 day 10
  more examples             "the week after next" = week 2, "the day after the day after tomorrow" = day 3,
                            "the past 24 hours" = last 1 day, "half a month ago" = day -15, "a year and a half ago" = month -18,
                            "half a year ago" = month -6, "June 5 to 8, 2024" = 2024-06-05 .. 2024-06-08,
                            "yesterday and today" = day -1 .. now, "from last Friday to this Monday" = week -1 fri .. week 0 mon
  before X / after X        "before 2020" = before 2020, "before last week" = before week -1, "after this month" = after month 0
  X ; Y                     two separate periods: "May 2025 and June 2026" = 2025-05 ; 2026-06

Message: {q}

Reply with only the code line."""


# No-date rows: a yes/no check. The round trip would drop every trap the LLM itself misreads
# ("the may report" read as May), and those traps are exactly what the model must learn.
CHECK_NONE = (
    "A user sent this message to a memory assistant:\n\"{q}\"\n\n"
    "Does it ask about or point to a specific date, day, or period of time when something happened or will happen "
    "(like yesterday, last week, in May, in 2023, recently)? Names of people or things that happen to be month, day "
    "or season words, durations, frequencies, versions, prices, product names and times of day alone don't count. "
    "Answer only yes or no."
)


def same_meaning(a: str, b: str) -> bool:
    """Two codes agree if they give the same dates on a few different anchors."""
    anchors = [date(2026, 8, 12), date(2026, 1, 1), date(2025, 3, 9), date(2024, 2, 29), date(2026, 8, 3), date(2026, 11, 17)]
    return all(ranges_json(a, x) == ranges_json(b, x) for x in anchors) and (a == "none") == (b == "none")


def random_anchor(r: random.Random) -> date:
    if r.random() < 0.2:
        return r.choice([date(2024, 2, 29), date(2026, 1, 1), date(2025, 12, 31), date(2026, 3, 1),
                         date(2026, 8, 3), date(2026, 8, 9), date(2026, 8, 8), date(2027, 6, 30)])
    return date(2023, 1, 1) + timedelta(days=r.randrange(5 * 365))


def ranges_json(code: str, anchor: date) -> list:
    return [[s and s.isoformat(), e and e.isoformat()] for s, e in resolve(code, datetime.combine(anchor, datetime.min.time()))]


def make_jobs(split: str, lang: str, n: int, seed: int, k: int, none_share: float = 1 / 3,
              kinds: set[str] | None = None, trap_filter: str = "") -> list[dict]:
    r = random.Random(seed)
    base = TRAPS_EN if lang == "en" else [t.format(lang=LANGS[lang]) for t in TRAPS_ANY]
    if lang.startswith("zh"):
        base = base + TRAPS_ZH * 2
    traps = [t for t in base if re.search(trap_filter, t, re.I)] if trap_filter else base
    jobs = []
    for _ in range(-(-n // k)):
        job = {"lang": lang, "k": k, "topic": r.choice(TOPICS), "style": r.choice(STYLES), "persona": r.choice(PERSONAS),
               "split": split, "seed": r.randrange(2**31)}
        if r.random() < none_share:
            job |= {"kind": "none", "code": "none", "desc": "", "trap": r.choice(traps)}
        else:
            job["kind"], job["code"], job["desc"] = sample(r)
            while kinds and job["kind"] not in kinds:
                job["kind"], job["code"], job["desc"] = sample(r)
            if job["kind"] in ("day", "weekday") and r.random() < 0.3:  # answers are whole days, so the label holds
                job["desc"] += " (also mention a part of that day, like morning, afternoon, evening or night)"
            elif r.random() < 0.15:  # numbers that are not dates must not move the label
                job["desc"] += (" (also mention an unrelated number that is NOT a date, like a price, amount, count, "
                                "order or room number, e.g. '2019 usd', '1500 steps', 'room 2024')")
        jobs.append(job)
    return jobs


# An English note in brackets inside a non-English message: "明日… (I'm asking the app where …)"
_GLOSS = re.compile(r"\((?:[A-Za-z']+[\s,.?!:;/-]+){3,}")
_NON_LATIN = re.compile(r"[\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")


def has_gloss(t: str) -> bool:
    """Only for messages in a non-Latin script; Indonesian or Hinglish text in brackets is fine."""
    return bool(_NON_LATIN.search(t) and _GLOSS.search(t))


def write(job: dict) -> list[dict]:
    system, prompt = (WRITE_NONE if job["kind"] == "none" else WRITE)[job["split"]]
    fields = job | {"lang": LANGS[job["lang"]]}
    texts = llm.json_list(llm.chat(prompt.format(**fields), system=system.format(**fields)))
    ok = [t.strip() for t in texts if isinstance(t, str) and 0 < len(t.strip()) < 400]
    if job["lang"] != "en":  # Gemma sometimes adds an English gloss: "明年这时候… (A year from now)"
        ok = [t for t in ok if not has_gloss(t)]
    return [job | {"q": t} for t in ok]


def read_code(q: str) -> str:
    """The LLM's own reading of a message, as one code line."""
    reply = llm.chat(READ.format(q=q), temperature=0, max_tokens=40)
    return reply.strip().strip("`").strip().splitlines()[0].strip() if reply.strip() else ""


def check(row: dict) -> bool:
    if row["kind"] == "none":
        return llm.chat(CHECK_NONE.format(q=row["q"]), temperature=0, max_tokens=5).strip().lower().startswith("no")
    return same_meaning(read_code(row["q"]), row["code"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--k", type=int, default=0, help="messages per LLM call (default: 3 test, 5 train)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--none-share", type=float, default=1 / 3)
    ap.add_argument("--style", default="", help="force this writing style for every call")
    ap.add_argument("--trap-filter", default="", help="regex: only trap ideas matching it (no-date rows)")
    ap.add_argument("--kinds", default="", help="comma list: only these meaning kinds (see timmytime/meanings.py)")
    ap.add_argument("--far-share", type=float, default=None, help="share of years far from today (default 0.08)")
    ap.add_argument("--month-end-share", type=float, default=None, help="share of days on the 29th-31st (default 0.1)")
    a = ap.parse_args()
    if a.far_share is not None:
        import training.data.meanings as meanings
        meanings.FAR_SHARE = a.far_share
    if a.month_end_share is not None:
        import training.data.meanings as meanings
        meanings.MONTH_END_SHARE = a.month_end_share
    k = a.k or (3 if a.split == "test" else 5)

    jobs = make_jobs(a.split, a.lang, a.n, a.seed, k, a.none_share, set(filter(None, a.kinds.split(","))), a.trap_filter)
    if a.style:
        jobs = [j | {"style": a.style} for j in jobs]
    # Saved chunk by chunk, so a crash loses at most one chunk; rerun the same command to resume.
    progress = a.out.with_suffix(".progress")
    done = int(progress.read_text()) if progress.exists() and a.out.exists() else 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if not done:
        a.out.write_text("")
    seen = {json.loads(line)["q"].lower() for line in a.out.open()}
    chunk = 400
    for start in range(done, len(jobs), chunk):
        part = jobs[start : start + chunk]
        rows = [row for batch in llm.run_all(write, part, a.workers) if batch for row in batch]
        kept = [row for row, ok in zip(rows, llm.run_all(check, rows, a.workers)) if ok]
        with a.out.open("a") as fh:
            for row in kept:
                if row["q"].lower() in seen:
                    continue
                seen.add(row["q"].lower())
                r = random.Random(row["seed"] ^ zlib.crc32(row["q"].encode()))
                for _ in range(20):  # an anchor where the meaning has dates ("since 2027" asked in 2025 has none)
                    anchor = random_anchor(r)
                    ranges = ranges_json(row["code"], anchor)
                    if ranges or row["code"] == "none":
                        break
                else:
                    continue
                fh.write(json.dumps({"q": row["q"], "code": row["code"], "kind": row["kind"], "lang": row["lang"],
                                     "anchor": anchor.isoformat(), "ranges": ranges}, ensure_ascii=False) + "\n")
        progress.write_text(str(start + len(part)))
        print(f"{start + len(part)}/{len(jobs)} calls, {len(rows)} written, {len(kept)} kept, {len(seen)} saved", flush=True)
    progress.unlink(missing_ok=True)
    print(f"done: {len(seen)} rows in {a.out}")

if __name__ == "__main__":
    main()
