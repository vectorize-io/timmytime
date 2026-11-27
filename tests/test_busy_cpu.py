"""text2date must stay fast when several processes share the CPU (test workers, a busy server).

numpy's math library on Linux (OpenBLAS) splits every matrix product across all cores. The model's
products are tiny, so with busy processes those threads fight: 0.1.0 took ~350 ms a parse with 6
processes on 6 cores, instead of ~5 ms. text2date runs its math on one thread, then restores the setting.
"""

import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest
from threadpoolctl import ThreadpoolController

import text2date
from text2date import runtime

QS = ["what did I decide last week", "上个月的会议", "¿qué hice el mes pasado?", "notes from early 2024"]
REF = date(2026, 11, 18)


_controller = ThreadpoolController()


def _blas_threads() -> list[int]:
    return [lib["num_threads"] for lib in _controller.info() if lib["user_api"] == "blas"]


needs_threaded_blas = pytest.mark.skipif(
    not any(n > 1 for n in _blas_threads()), reason="numpy's math library here is not multi-threaded (macOS uses Accelerate)"
)


def _ms_per_parse(on_worker_thread: bool, n: int = 40) -> float:
    def loop():
        text2date.parse("warm up", REF)
        t = time.perf_counter()
        for i in range(n):
            text2date.parse(QS[i % len(QS)], REF)
        return (time.perf_counter() - t) / n * 1000

    if not on_worker_thread:
        return loop()
    with ThreadPoolExecutor(1) as ex:  # how Hindsight calls it
        return ex.submit(loop).result()


@needs_threaded_blas
@pytest.mark.parametrize("on_worker_thread", [False, True], ids=["main-thread", "worker-thread"])
def test_fast_when_every_core_runs_a_parser(on_worker_thread):
    procs = max(4, os.cpu_count() or 4)
    with mp.get_context("spawn").Pool(procs) as pool:
        ms = pool.map(_ms_per_parse, [on_worker_thread] * procs)
    # ~5 ms a parse with the fix, ~350 ms without; 50 leaves room for slow CI machines.
    assert max(ms) < 50, f"{procs} busy processes: {max(ms):.0f} ms a parse"


@needs_threaded_blas
def test_one_thread_while_parsing_then_restored(monkeypatch):
    before, seen, real = _blas_threads(), [], runtime.Parser.code

    def spy(self, text):
        seen.append(_blas_threads())
        return real(self, text)

    monkeypatch.setattr(runtime.Parser, "code", spy)
    text2date.parse("last week", REF)
    assert seen == [[1] * len(before)]
    assert _blas_threads() == before


def test_threads_parsing_at_once_leave_the_setting_as_it_was():
    before = _blas_threads()
    threads = [threading.Thread(target=text2date.parse, args=(QS[i % len(QS)], REF)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _blas_threads() == before
    assert runtime._blas_users == 0
