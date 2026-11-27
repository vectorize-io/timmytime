"""The model in a child process, so it never holds up the caller's threads or its asyncio loop.

    from text2date import ProcessParser
    parser = ProcessParser()                      # starts a child running text2date.worker.main
    parser.parse("what did I decide last week")   # same answers as text2date.parse
    await parser.aparse("上个月的会议")            # from asyncio: the loop keeps running

In-process, the model's numpy and Python steps hold the GIL, so a worker thread running it still
freezes the event loop under load (30-60 ms with 64 parses at once). Here the caller only waits on a
pipe, which releases the GIL. A plain child process, not multiprocessing: it never re-imports the
caller's main module. One request at a time per ProcessParser (about 0.1 ms of pipe overhead); the child
restarts if it dies, and exits when the parent does (its stdin closes).
"""

import asyncio
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

from text2date.resolve import DateRange
from text2date.runtime import MIN_CONFIDENCE, WEIGHTS, Parser


def _anchor_out(anchor: datetime | date | None) -> list | None:
    if anchor is None:
        return None
    return ["datetime" if isinstance(anchor, datetime) else "date", anchor.isoformat()]


def _anchor_in(raw: list | None) -> datetime | date | None:
    if raw is None:
        return None
    kind, value = raw
    return datetime.fromisoformat(value) if kind == "datetime" else date.fromisoformat(value)


class ProcessParser:
    def __init__(self, weights: Path | str = WEIGHTS) -> None:
        self._cmd = [sys.executable, "-c", "from text2date.worker import main; main()", str(weights)]
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        # aparse waits here: requests go one at a time anyway, and more waiting threads only fight for the GIL.
        self._waiter = ThreadPoolExecutor(1, thread_name_prefix="text2date-wait")

    def _start(self) -> subprocess.Popen:
        proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                encoding="utf-8", bufsize=1)
        if proc.stdout.readline().strip() != "ready":  # the model loaded (or the child failed, on stderr)
            proc.kill()
            raise RuntimeError("text2date worker failed to start (its error is on stderr)")
        return proc

    def load(self) -> None:
        """Start the child now instead of on the first parse."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = self._start()

    def parse(self, text: str, anchor: datetime | date | None = None, tz: str = "UTC",
              min_confidence: float = MIN_CONFIDENCE) -> list[DateRange]:
        """Same as Parser.parse, run in the child. Safe from any thread."""
        request = json.dumps({"text": text, "anchor": _anchor_out(anchor), "tz": tz, "min": min_confidence})
        with self._lock:
            for attempt in (1, 2):  # a dead child is restarted once
                if self._proc is None or self._proc.poll() is not None:
                    self._proc = self._start()
                try:
                    self._proc.stdin.write(request + "\n")
                    self._proc.stdin.flush()
                    line = self._proc.stdout.readline()
                except OSError:
                    line = ""
                if line:
                    break
                self._proc = None
                if attempt == 2:
                    raise RuntimeError("text2date worker died twice in a row")
        answer = json.loads(line)
        if "error" in answer:  # what Parser.parse raised in the child (a bad tz, a bad anchor)
            raise ValueError(answer["error"])
        return [DateRange(date.fromisoformat(s) if s else None, date.fromisoformat(e) if e else None)
                for s, e in answer["ranges"]]

    async def aparse(self, text: str, anchor: datetime | date | None = None, tz: str = "UTC",
                     min_confidence: float = MIN_CONFIDENCE) -> list[DateRange]:
        """parse() for asyncio: waits on a thread that only reads a pipe, so the loop never stalls."""
        return await asyncio.get_running_loop().run_in_executor(self._waiter, self.parse, text, anchor, tz,
                                                                min_confidence)

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            self._proc = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def main() -> None:
    parser = Parser(sys.argv[1] if len(sys.argv) > 1 else WEIGHTS)
    print("ready", flush=True)
    for line in sys.stdin:
        r = json.loads(line)
        try:
            ranges = parser.parse(r["text"], _anchor_in(r["anchor"]), r["tz"], r["min"])
            out = {"ranges": [[s.isoformat() if s else None, e.isoformat() if e else None] for s, e in ranges]}
        except Exception as e:  # odd text never raises; a bad tz or anchor does, in the caller too
            out = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


