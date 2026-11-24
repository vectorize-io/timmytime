# timmytime

**timmytime** (tiny time semantic parser) finds the dates and periods a question talks about ("last week", "上上周", "since March"), whole days only. A 9 MB model, numpy only, about 1.5 ms per question. Every model needs its timmy time.

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

## Install

Not released under the name timmytime yet (the last release, v0.1.0, is still called `tlm`). Until
then, build it from a checkout:

```sh
scripts/get_weights.sh && uv build && pip install dist/timmytime-*.whl
```

The package carries its weights (10 MB) inside, so nothing is downloaded at run time.
`pip install git+…` does not work, because the weights are not in git (see TRAINING.md).

Needs Python 3.11+, `numpy` and `tzdata`. No torch, no network, no LLM at run time.

## API

`timmytime.parse(text, anchor=None, tz="UTC") -> list[DateRange]`

- `anchor`: when the question was asked. A `date` is that calendar day in `tz`. A `datetime`
  without a timezone is taken as UTC. Leave it out to use now.
- `tz`: an IANA name like `"Europe/Rome"`. It decides what "today" is, and the hemisphere for
  seasons ("last summer" in Buenos Aires is Dec to Feb). An unknown name raises.
- Returns whole-day ranges in the order the question names them. `start` or `end` is `None` for
  open ranges ("before 2020", "since May"). `[]` means no date. It never raises on odd text.
- `DateRange` is a named tuple `(start, end)` of `datetime.date`, both days included.
- Long text is read in sentence pieces of up to 256 bytes; the ranges of all pieces are returned,
  each once.

Low-confidence answers become `[]`: a wrong date filters out the right results, while no date
just searches everything. The cut-off is `TIMMYTIME_MIN_CONFIDENCE` (default 0.5).

`timmytime.Parser(weights_path)` loads other weights. `timmytime.hindsight.extract_temporal_constraint`
has the shape hindsight's recall expects: the first range as naive datetimes, or `None`.

## What it understands

Every row below is run against the shipped model by `tests/test_readme.py`, so this table is what
the library does today. "Today" is Wednesday 2026-11-18; time zone UTC unless given. Both dates are
included; `…` is an open end.

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

The rules behind the table:

- Weeks run Monday to Sunday. "Last week", "this month" and "next year" are whole calendar periods.
- "Last Friday", "last summer", "last Christmas" and a month with no year ("in March") mean the most
  recent one that has started. "This coming Friday" is the next one.
- "In two weeks" is that whole calendar week; "in 3 days" is that single day.
- "Early", "mid" and "late" are the first, middle and last third of the period; "first / second
  half" are halves; "around" a date is 3 days either side.
- Fuzzy words have fixed windows: "recently" = the last 30 days, "the other day" = 1 to 7 days ago,
  "a while ago" = 1 to 6 months ago, "a couple / a few weeks ago" = 1-3 / 2-5 weeks ago,
  "a long time ago" = more than a year ago.
- Seasons follow the hemisphere of the time zone ("last summer" in Buenos Aires is Dec to Feb).
- Holidays: fixed-date ones are that day; Easter is Good Friday to Easter Monday; Chinese New Year
  is its 7-day holiday; Ramadan is the whole month; Eid al-Fitr 3 days, Eid al-Adha 4 days.

Languages: English, Spanish, German, Chinese, Arabic (including Arabizi), Japanese, Korean, Hindi
(including Hinglish), Indonesian, Vietnamese and Thai. The next weights add Italian, French, Russian,
Portuguese and traditional Chinese (see Accuracy). It reads raw bytes, so other languages don't crash,
but they were not trained or tested.

## Accuracy

Exact match on held-out test questions, about 2,900-3,100 per language, covering every kind below
(written with prompts never used for training). "False dates" = questions with no date that got one,
leaving out no-date questions where the LLM that checked the labels itself reads a date.

"Shipped" is the pinned weights (`weights-g16p`). "Next" is round g20, trained and scored but not
released yet.

| Language | Shipped exact | Shipped false dates | Next exact | Next false dates |
|---|---|---|---|---|
| English | 98.1% | 0.8% | 98.5% | 1.0% |
| Chinese | 97.0% | 0.4% | 97.5% | 0.9% |
| Spanish | 96.8% | 0.7% | 97.8% | 0.4% |
| German | 96.9% | 1.4% | 97.0% | 1.7% |
| Arabic | 96.2% | 0.9% | 96.8% | 1.1% |
| Korean | 96.4% | 1.6% | 96.8% | 1.6% |
| Japanese | 95.3% | 1.0% | 96.5% | 1.7% |
| Thai | 94.3% | 0.5% | 95.5% | 1.2% |
| Hindi | 94.2% | 0.7% | 95.4% | 0.9% |
| Indonesian | 94.3% | 2.0% | 95.4% | 1.8% |
| Vietnamese | 94.1% | 1.2% | 95.4% | 1.1% |
| Italian | - | - | 96.3% | 1.0% |
| Portuguese | - | - | 96.3% | 0.7% |
| Chinese (traditional) | - | - | 96.0% | 0.6% |
| French | - | - | 95.2% | 0.7% |
| Russian | - | - | 94.1% | 1.5% |

For comparison, dateparser-based extraction gets about 44% exact with 24% false dates in English,
and 5-48% exact in the other languages.

Hindsight's own query-analyzer tests and golden corpus, as timmytime codes, are in
`data/test_hindsight.jsonl` (835 questions, mostly short Chinese ones): 64% exact with the shipped
weights, 97.2% with the next ones.

## How it works

1. The model reads the question's UTF-8 bytes and writes a short code: `week -1`,
   `last 6 month`, `2025-05 ; 2026-06`, `none`.
2. Plain Python (`timmytime/resolve.py`) turns that code, the anchor and the timezone into dates.

The model never does calendar math, so leap years, week starts and month lengths are always
right. The code grammar is at the top of `timmytime/resolve.py`.

The model is a small byte-level encoder-decoder transformer (6.4M weights, fewer after pruning):
technically a semantic parser, distilled from an LLM. It was trained on about 1.1M questions, written
by an LLM and checked automatically, plus short template-made ones. Fixes and new languages come
from more training data, never from rules in the code. See [TRAINING.md](TRAINING.md) to retrain
or extend it.

## Limits

- Whole days only: "yesterday at 3pm" gives yesterday.
- Weeks start on Monday.
- Holidays: fixed-date ones (Christmas, New Year, Halloween, national days), Easter, US Thanksgiving,
  Lunar New Year (Chinese New Year, Seollal, Tet), Mid-Autumn / Chuseok, Dragon Boat, Ramadan, the two
  Eids (1950-2076), Golden Week, Songkran and Obon. Not yet: Diwali, Holi and other Hindu-calendar days.
  Personal events ("my birthday") give no date.
- Known weak spots (2026-11-24): two-date questions are weaker outside English; very short Chinese
  queries were the main gap until the next weights; "N units ago" is that whole unit ("two months
  ago" = all of that month), not a single day.
- The examples in this README were run against the shipped weights. The table above is the
  measured accuracy; single phrasings can still miss.

## Development

```sh
uv sync
scripts/get_weights.sh            # the weights are on a GitHub release, pinned by weights.lock
PYTEST_ADDOPTS= uv run pytest -q
```

`timmytime/` is the package. `training/` holds data generation, the torch model and scoring (see
TRAINING.md).
