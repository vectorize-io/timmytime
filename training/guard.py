"""Run a command inside resource limits, so tlm jobs never starve the machine.

    python -m training.guard -- uv run python -m training.model.model --train data/train_en.jsonl

- low priority (nice 10) and at most --threads CPU threads (torch, numpy, BLAS);
- GPU memory capped (PYTORCH_MPS_HIGH_WATERMARK_RATIO; torch raises instead of growing);
- the job's memory footprint (all its processes, GPU memory included) is checked every few seconds:
  above --max-gb the job is killed;
- if the machine's free memory falls under --kill-below %, the job is killed. Pausing is not enough:
  a paused job keeps its memory, and on 2026-11-18 a memory-starved Mac panicked twice.
Every action is logged to stderr with a timestamp.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time


def _log(msg: str) -> None:
    print(f"[guard {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def tree_pids(root: int) -> list[int]:
    out = subprocess.run(["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True).stdout
    kids: dict[int, list[int]] = {}
    for line in out.splitlines():
        pid, ppid = map(int, line.split())
        kids.setdefault(ppid, []).append(pid)
    pids, todo = [], [root]
    while todo:
        pid = todo.pop()
        pids.append(pid)
        todo += kids.get(pid, [])
    return pids


def footprint_mb(pids: list[int]) -> float:
    """macOS memory footprint of these processes, in MB. Unlike RSS it counts GPU (Metal) memory,
    which is where a torch job on Apple silicon keeps most of its memory."""
    out = subprocess.run(["footprint", *[f"-p{p}" for p in pids]], capture_output=True, text=True).stdout
    units = {"B": 1 / 2**20, "KB": 1 / 1024, "MB": 1, "GB": 1024}
    return sum(float(n) * units[u] for n, u in re.findall(r"\[\d+\]: .*?Footprint: ([\d.]+) (B|KB|MB|GB)", out))  # per process; skip "Summary Footprint"


def free_memory_pct() -> float:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)%", out)
    return float(m[1]) if m else 100.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu-ratio", default="0.05", help="share of the GPU's recommended memory torch may use (~1.4 GB)")
    ap.add_argument("--kill-below", type=float, default=12.0, help="kill the job when the machine's free memory %% is under this")
    ap.add_argument("--kill-below-min-gb", type=float, default=1.0,
                    help="...but only if the job itself uses at least this much: killing a small job frees nothing")
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")

    t = str(a.threads)
    env = os.environ | {"OMP_NUM_THREADS": t, "VECLIB_MAXIMUM_THREADS": t, "OPENBLAS_NUM_THREADS": t,
                        "MKL_NUM_THREADS": t, "TLM_TORCH_THREADS": t, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": a.gpu_ratio,
                        "PYTORCH_MPS_LOW_WATERMARK_RATIO": str(float(a.gpu_ratio) * 0.8)}  # torch wants low <= high
    proc = subprocess.Popen(["nice", "-n", "10", *cmd], env=env, start_new_session=True)
    _log(f"pid {proc.pid}: max {a.max_gb} GB (GPU included), {a.threads} threads, gpu ratio {a.gpu_ratio}, "
         f"killed if the machine has under {a.kill_below}% free memory")
    peak = 0.0
    while proc.poll() is None:
        time.sleep(a.every)
        mb = footprint_mb(tree_pids(proc.pid))
        peak = max(peak, mb)
        free = free_memory_pct()
        why = (f"job memory {mb:.0f} MB over {a.max_gb} GB" if mb > a.max_gb * 1024
               else f"machine free memory {free:.0f}% under {a.kill_below}%" if free < a.kill_below and mb >= a.kill_below_min_gb * 1024 else "")
        if why:
            _log(f"{why}: killing the job")
            os.killpg(proc.pid, signal.SIGKILL)
            break
    code = proc.wait()
    _log(f"exit {code}, peak memory {peak:.0f} MB (GPU included)")
    sys.exit(code if code >= 0 else 1)


if __name__ == "__main__":
    main()
