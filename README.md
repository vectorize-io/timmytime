# timmytime

**timmytime** (tiny time semantic parser) finds the dates a question talks about: "last week",
"上上周", "since March", "the 3rd of last month". A 10 MB model, numpy only, about 1 ms per question,
16 languages. Every model needs its timmy time.

```python
from datetime import date
import timmytime

timmytime.parse("what did I decide last week", anchor=date(2026, 11, 20))
# [DateRange(start=date(2026, 11, 9), end=date(2026, 11, 15))]

timmytime.parse("trips in May 2025 and June 2026", anchor=date(2026, 11, 20))
# [DateRange(2025-05-01, 2025-05-31), DateRange(2026-06-01, 2026-06-30)]

timmytime.parse("what is my wifi password")
# []   no date
```

## vs. dateparser

Hindsight finds dates with dateparser plus hand-written rules today. Both were run on the same
held-out questions (about 3,000 per language, every kind of date). **Exact** = every range right.
**False dates** = a date found in a question that has none.

| Language | dateparser exact | timmytime exact | dateparser false dates | timmytime false dates |
|---|---|---|---|---|
| English | 55.9% | **98.0%** | 24.4% | **1.1%** |
| Chinese | 66.9% | **97.6%** | 3.1% | **1.4%** |
| Chinese (traditional) | 63.4% | **97.5%** | 1.8% | **0.3%** |
| Spanish | 50.3% | **97.3%** | 19.6% | **0.7%** |
| German | 48.1% | **96.9%** | 18.4% | **2.2%** |
| Arabic | 42.2% | **95.5%** | 4.3% | **1.2%** |
| Japanese | 47.0% | **96.5%** | 8.9% | **1.7%** |
| Korean | 46.4% | **97.2%** | 5.4% | **1.8%** |
| Italian | 49.9% | **96.6%** | 10.4% | **1.0%** |
| Portuguese | 41.4% | **96.6%** | 18.3% | **1.0%** |
| French | 54.2% | **95.7%** | 14.2% | **1.0%** |
| Vietnamese | 40.4% | **94.6%** | 11.3% | **2.0%** |
| Indonesian | 40.0% | **95.6%** | 21.9% | **1.4%** |
| Thai | 43.2% | **95.3%** | 10.0% | **1.6%** |
| Hindi | 46.9% | **95.4%** | 16.9% | **1.0%** |
| Russian | 53.3% | **95.3%** | 16.9% | **0.9%** |
| hindsight's own tests (834) | 91.0% | **98.3%** | 5.2% | **1.5%** |

The last row is hindsight's query-analyzer tests and golden corpus (`data/test_hindsight.jsonl`).
They were written for dateparser, so it does well there. Time per question: dateparser 0.1-16 ms
(p95 up to 57 ms), timmytime about 1 ms (p95 1.5 ms).

timmytime numbers are the release candidate `bsoup` (one clean run from scratch, then the average of 3 short
fine-tunes of it), trained but not released yet. The shipped weights (`weights-g16p`) are 1-2 points lower,
cover 11 languages, and get 64% on hindsight's tests.

## Install

Not on PyPI yet. Build it from a checkout (the weights live on a GitHub release, not in git):

```sh
scripts/get_weights.sh && uv build && pip install dist/timmytime-*.whl
```

Python 3.11+, `numpy` and `tzdata`. No torch, no network, no LLM at run time.

## API

`timmytime.parse(text, anchor=None, tz="UTC") -> list[DateRange]`

- `anchor`: when the question was asked. A `date` is that day in `tz`; a naive `datetime` is UTC;
  `None` is now.
- `tz`: an IANA name like `"Europe/Rome"`. It sets what "today" is and the hemisphere for seasons.
- Returns whole-day ranges `(start, end)` of `datetime.date`, both included, in the order the
  question names them. `None` is an open end ("before 2020"). `[]` = no date. Never raises on odd text.
- Answers below `TIMMYTIME_MIN_CONFIDENCE` (default 0.5) become `[]`: a wrong date hides the right
  results, no date just searches everything.

`timmytime.Parser(path)` loads other weights. `timmytime.hindsight.extract_temporal_constraint` has
the shape hindsight's recall expects.

## What it understands

"Last week" and "this month" are whole calendar periods; weeks start on Monday. "N units ago" is
that whole unit ("two months ago" = all of that month). "Last Friday" or "in March" is the latest
one that has started. Early / mid / late are thirds. Fuzzy words have fixed windows ("recently" =
the last 30 days). Seasons follow the time zone's hemisphere. Holidays: fixed-date ones, Easter, US
Thanksgiving, Lunar New Year, Mid-Autumn, Dragon Boat, Ramadan, the Eids, Golden Week, Songkran, Obon.

<details>
<summary>Over 100 examples, each one checked by the tests (today = Wednesday 2026-11-18)</summary>

<!-- examples:start -->
| Kind | Question | Time zone | Dates |
|---|---|---|---|
| Days | what did I do today |  | 2026-11-18 |
| Days | what did I do yesterday |  | 2026-11-17 |
| Days | what did I do the day before yesterday |  | 2026-11-16 |
| Days | what's on tomorrow |  | 2026-11-19 |
| Days | what did I do 3 days ago |  | 2026-11-15 |
| Days | what's planned in 3 days |  | 2026-11-21 |
| Weeks | notes from this week |  | 2026-11-16 → 2026-11-22 |
| Weeks | what did I decide last week |  | 2026-11-09 → 2026-11-15 |
| Weeks | plans for next week |  | 2026-11-23 → 2026-11-29 |
| Weeks | what happened 3 weeks ago |  | 2026-10-26 → 2026-11-01 |
| Weeks | what do I have in two weeks |  | 2026-11-30 → 2026-12-06 |
| Weeks | the week before last |  | 2026-11-02 → 2026-11-08 |
| Months | expenses this month |  | 2026-11-01 → 2026-11-30 |
| Months | what did I spend last month |  | 2026-10-01 → 2026-10-31 |
| Months | what's due next month |  | 2026-12-01 → 2026-12-31 |
| Months | what happened 3 months ago |  | 2026-08-01 → 2026-08-31 |
| Months | what do I have in a month |  | 2026-12-01 → 2026-12-31 |
| Quarters | sales this quarter |  | 2026-10-01 → 2026-12-31 |
| Quarters | results from last quarter |  | 2026-07-01 → 2026-09-30 |
| Quarters | what did we ship in Q3 |  | 2026-07-01 → 2026-09-30 |
| Quarters | Q1 2025 revenue |  | 2025-01-01 → 2025-03-31 |
| Years | goals for this year |  | 2026-01-01 → 2026-12-31 |
| Years | what happened last year |  | 2025-01-01 → 2025-12-31 |
| Years | plans for next year |  | 2027-01-01 → 2027-12-31 |
| Years | trips 2 years ago |  | 2024-01-01 → 2024-12-31 |
| Years | photos from 2019 |  | 2019-01-01 → 2019-12-31 |
| Years | music from the 90s |  | 1990-01-01 → 1999-12-31 |
| Weekdays | what did I do last Friday |  | 2026-11-13 |
| Weekdays | what did I do on Monday |  | 2026-11-16 |
| Weekdays | what's on this coming Friday |  | 2026-11-20 |
| Weekdays | what happened Monday last week |  | 2026-11-09 |
| Weekdays | plans for Tuesday next week |  | 2026-11-24 |
| Weekends | what did I do last weekend |  | 2026-11-14 → 2026-11-15 |
| Weekends | plans for this weekend |  | 2026-11-21 → 2026-11-22 |
| Weekends | plans for next weekend |  | 2026-11-28 → 2026-11-29 |
| Months by name | what did I do in March |  | 2026-03-01 → 2026-03-31 |
| Months by name | trips in March last year |  | 2025-03-01 → 2025-03-31 |
| Months by name | plans for this June |  | 2026-06-01 → 2026-06-30 |
| Months by name | what happened in May 2025 |  | 2025-05-01 → 2025-05-31 |
| Dates | what did I do on March 12 |  | 2026-03-12 |
| Dates | what happened on 2024-03-12 |  | 2024-03-12 |
| Dates | meeting on 12 March 2024 |  | 2024-03-12 |
| Seasons | what did I do last summer |  | 2026-06-01 → 2026-08-31 |
| Seasons | photos from summer 2023 |  | 2023-06-01 → 2023-08-31 |
| Seasons | the winter of last year |  | 2025-12-01 → 2026-02-28 |
| Seasons | what did I do last summer | America/Argentina/Buenos_Aires | 2025-12-01 → 2026-02-28 |
| Windows | meetings in the last 6 months |  | 2026-05-18 → 2026-11-18 |
| Windows | what did I read in the past 3 weeks |  | 2026-10-28 → 2026-11-18 |
| Windows | anything planned for the next 3 days |  | 2026-11-18 → 2026-11-21 |
| Windows | what did I do between March and July |  | 2026-03-01 → 2026-07-31 |
| Windows | projects from 2019 to 2021 |  | 2019-01-01 → 2021-12-31 |
| Windows | what did I read since May |  | 2026-05-01 → 2026-11-18 |
| Windows | everything since last week |  | 2026-11-09 → 2026-11-18 |
| Windows | notes from before 2020 |  | … → 2019-12-31 |
| Windows | what happened after 2020 |  | 2021-01-01 → … |
| Parts | what did I do in early 2024 |  | 2024-01-01 → 2024-04-30 |
| Parts | what did I do in mid 2024 |  | 2024-05-01 → 2024-08-31 |
| Parts | notes from mid-March |  | 2026-03-12 → 2026-03-22 |
| Parts | what happened in late March |  | 2026-03-23 → 2026-03-31 |
| Parts | what happened at the end of last year |  | 2025-09-01 → 2025-12-31 |
| Parts | the first half of 2023 |  | 2023-01-01 → 2023-06-30 |
| Parts | the second half of last year |  | 2025-07-01 → 2025-12-31 |
| Parts | early this month |  | 2026-11-01 → 2026-11-10 |
| Parts | around March 12 2024 |  | 2024-03-09 → 2024-03-15 |
| Fuzzy | what did I read recently |  | 2026-10-19 → 2026-11-18 |
| Fuzzy | that thing I saw the other day |  | 2026-11-11 → 2026-11-17 |
| Fuzzy | something I said a while ago |  | 2026-05-22 → 2026-10-19 |
| Fuzzy | a couple of weeks ago |  | 2026-10-28 → 2026-11-11 |
| Fuzzy | a few months ago |  | 2026-06-21 → 2026-09-19 |
| Fuzzy | a long time ago |  | … → 2025-11-18 |
| Holidays | what did I do last Christmas |  | 2025-12-25 |
| Holidays | christmas 2023 photos |  | 2023-12-25 |
| Holidays | what did I do on Christmas Eve |  | 2025-12-24 |
| Holidays | New Year's Eve party |  | 2025-12-31 |
| Holidays | what did we do over Easter |  | 2026-04-03 → 2026-04-06 |
| Holidays | Easter 2024 |  | 2024-03-29 → 2024-04-01 |
| Holidays | dinner plans for Thanksgiving last year |  | 2025-11-27 |
| Holidays | where did I go for Chinese New Year |  | 2026-02-17 → 2026-02-23 |
| Holidays | what did I do during Ramadan |  | 2026-02-18 → 2026-03-19 |
| Holidays | Eid al-Adha 2025 |  | 2025-06-06 → 2025-06-09 |
| Holidays | what did we do during Golden Week | Asia/Tokyo | 2026-04-29 → 2026-05-05 |
| Holidays | Halloween costume ideas from last Halloween |  | 2026-10-31 |
| Several | trips in May 2025 and June 2026 |  | 2025-05-01 → 2025-05-31 ; 2026-06-01 → 2026-06-30 |
| Several | compare last week with the week before |  | 2026-11-09 → 2026-11-15 ; 2026-11-02 → 2026-11-08 |
| No date | what is my wifi password |  | no date |
| No date | may I ask you something |  | no date |
| No date | remind me what June said |  | no date |
| No date | install Office 2019 |  | no date |
| No date | show the last slide |  | no date |
| No date | room 2024 cleaning notes |  | no date |
| No date | my birthday party ideas |  | no date |
| No date | the meeting takes 2 weeks to prepare |  | no date |
| No date | every Monday standup notes |  | no date |
| Other languages | ¿qué hice la semana pasada? | Europe/Madrid | 2026-11-09 → 2026-11-15 |
| Other languages | was habe ich letzten Monat gemacht | Europe/Berlin | 2026-10-01 → 2026-10-31 |
| Other languages | 上个月的会议 | Asia/Shanghai | 2026-10-01 → 2026-10-31 |
| Other languages | ماذا فعلت الأسبوع الماضي | Asia/Riyadh | 2026-11-09 → 2026-11-15 |
| Other languages | 先週の会議のメモ | Asia/Tokyo | 2026-11-09 → 2026-11-15 |
| Other languages | 지난주 회의 메모 | Asia/Seoul | 2026-11-09 → 2026-11-15 |
| Other languages | पिछले हफ्ते की मीटिंग | Asia/Kolkata | 2026-11-09 → 2026-11-15 |
| Other languages | catatan rapat minggu lalu | Asia/Jakarta | 2026-11-09 → 2026-11-15 |
| Other languages | ghi chú cuộc họp tuần trước | Asia/Ho_Chi_Minh | 2026-11-09 → 2026-11-15 |
| Other languages | บันทึกการประชุมสัปดาห์ที่แล้ว | Asia/Bangkok | 2026-11-09 → 2026-11-15 |
| Other languages | 春节期间我去了哪里 | Asia/Shanghai | 2026-02-17 → 2026-02-23 |
| Other languages | ¿qué hice en Semana Santa? | Europe/Madrid | 2026-04-03 → 2026-04-06 |
<!-- examples:end -->

</details>

## Limits

- Whole days only: "yesterday at 3pm" gives yesterday.
- No Diwali, Holi or other Hindu-calendar days yet; personal events ("my birthday") give no date.
- Other languages don't crash (it reads raw bytes), but they are not trained or tested.
- Accuracy is measured on test sets; any single phrasing can still miss.

## How it works

The model reads the question's bytes and writes a short code (`week -1`, `last 6 month`,
`2025-05 ; 2026-06`, `none`). Plain Python (`timmytime/resolve.py`) turns the code into dates, so
leap years, week starts and month lengths are always right.

The model is a byte-level encoder-decoder transformer with 5.2M weights: a semantic parser distilled
from an LLM, trained on about 1.1M checked questions. Fixes and new languages come from training
data, never from rules in the code. See [TRAINING.md](TRAINING.md).

## Development

```sh
uv sync
scripts/get_weights.sh
PYTEST_ADDOPTS= uv run pytest -q
```
