"""ProcessParser: the same answers as text2date.parse, from a child process that never stalls the caller."""

import asyncio
import json
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

import text2date
from text2date import ProcessParser

QS = [json.loads(line)["q"] for line in (Path(__file__).parents[1] / "data" / "test_hindsight.jsonl").open()]
REF = date(2026, 11, 18)


@pytest.fixture(scope="module")
def parser():
    p = ProcessParser()
    p.load()
    yield p
    p.close()


def test_same_answers_as_in_process(parser):
    for q in QS[:300]:
        assert parser.parse(q, REF) == text2date.parse(q, REF), q


@pytest.mark.parametrize(
    "anchor",
    [REF, datetime(2026, 11, 18, 12), datetime(2024, 4, 5, 22, tzinfo=timezone(timedelta(hours=-5))), None],
    ids=["date", "naive", "aware", "none"],
)
def test_anchor_kinds(parser, anchor):
    q = "what did I do yesterday"
    assert parser.parse(q, anchor, "Europe/Rome") == text2date.parse(q, anchor, "Europe/Rome")


def test_open_ends_and_no_date(parser):
    assert parser.parse("notes from before 2020", REF) == [text2date.DateRange(None, date(2019, 12, 31))]
    assert parser.parse("what is my wifi password", REF) == []


def test_restarts_when_the_child_dies(parser):
    parser._proc.kill()
    parser._proc.wait()
    assert parser.parse("last week", REF) == text2date.parse("last week", REF)


def test_many_threads_at_once(parser):
    out: dict[int, list] = {}

    def run(i):
        out[i] = parser.parse(QS[i], REF)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(out[i] == text2date.parse(QS[i], REF) for i in range(32))


def test_aparse_keeps_the_event_loop_free(parser):
    """64 parses at once and the loop still runs: a 1 ms sleep wakes p99 ~2 ms late on a quiet machine
    (in-process on a worker thread the worst case is 30-60 ms). The bound is loose for busy CI machines."""

    want = {q: text2date.parse(q, REF) for q in QS[:640]}  # in-process, before the loop starts

    async def main():
        late, stop = [], asyncio.Event()

        async def ticker():
            while not stop.is_set():
                t = time.perf_counter()
                await asyncio.sleep(0.001)
                late.append((time.perf_counter() - t) * 1000 - 1)

        async def caller(i):
            for k in range(10):
                q = QS[i * 10 + k]
                assert await parser.aparse(q, REF) == want[q]

        tick = asyncio.create_task(ticker())
        await asyncio.gather(*(caller(i) for i in range(64)))
        stop.set()
        await tick
        return sorted(late)

    late = asyncio.run(main())
    assert late[int(len(late) * 0.99)] < 25, f"p99 {late[int(len(late) * 0.99)]:.1f} ms late"


def test_closed_parser_starts_again(parser):
    p = ProcessParser()
    assert p.parse("last week", REF) == text2date.parse("last week", REF)
    p.close()
    assert p.parse("yesterday", REF) == text2date.parse("yesterday", REF)
    p.close()
