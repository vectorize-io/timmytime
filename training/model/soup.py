"""Average the weights of runs that started from the same weights (same --seed, different --data-seed).

    python -m training.model.soup timmytime/weights/timmytime-r1.pt timmytime/weights/timmytime-r2.pt \
        --out timmytime/weights/timmytime-soup.npz --n-enc 6 --n-dec 1 --stride 4

Runs from different starting weights do not average (their units are in different orders); check the
average against each run before shipping it.
"""

import argparse
from pathlib import Path

import torch

from training.model.model import Cfg, TimmyTime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path, help=".pt checkpoints")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--n-dec", type=int, default=1)
    ap.add_argument("--n-enc", type=int, default=6)
    a = ap.parse_args()
    states = [torch.load(p, map_location="cpu") for p in a.runs]
    avg = {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}
    Cfg.stride, Cfg.n_dec, Cfg.n_enc = a.stride, a.n_dec, a.n_enc
    model = TimmyTime()
    model.load_state_dict(avg)
    model.export(a.out)
    torch.save(avg, a.out.with_suffix(".pt"))
    print(f"averaged {len(states)} runs into {a.out}")


if __name__ == "__main__":
    main()
