"""Remove encoder feed-forward units that never switch on, to make the model faster.

    python -m training.model.prune timmytime/weights/timmytime-g10.npz timmytime/weights/timmytime-g10p.npz data/train_*.jsonl

A ReLU unit that stays at 0 on every question in a large sample adds nothing to the output.
Dropping its column in f1 and its row in f2 leaves the answers unchanged for every question
where it stays off. Check with `training.eval.score` afterwards.
"""

import json
import random
import sys

import numpy as np

from timmytime.runtime import Parser, _attn_qkv, _conv, _norm, encode_bytes

SAMPLE = 80_000


def main() -> None:
    src, dst, *paths = sys.argv[1:]
    p = Parser(src)
    w = p.w
    rows = [json.loads(line)["q"] for path in paths for line in open(path)]
    qs = random.Random(0).sample(rows, min(SAMPLE, len(rows)))
    peak = [np.zeros(w[f"enc{i}.f1.w"].shape[1], np.float32) for i in range(p.n_enc)]
    for text in qs:
        if not text.strip():
            continue
        x = w["byte_emb"][encode_bytes(text)]
        x = np.maximum(_conv(x, w["conv1.w"], w["conv1.b"], w["conv1.k"], 1), 0)
        x = np.maximum(_conv(x, w["conv2.w"], w["conv2.b"], w["conv2.k"], p.stride), 0)
        x = x + w["enc_pos"][: x.shape[0]]
        for i in range(p.n_enc):
            k = f"enc{i}."
            x = x + _attn_qkv(_norm(x) @ w[k + "qkv.w"] + w[k + "qkv.b"], p.heads) @ w[k + "o.w"] + w[k + "o.b"]
            y = np.maximum(_norm(x) @ w[k + "f1.w"] + w[k + "f1.b"], 0)
            np.maximum(peak[i], y.max(0), out=peak[i])
            x = x + y @ w[k + "f2.w"] + w[k + "f2.b"]
    z = dict(np.load(src))
    for i, pk in enumerate(peak):
        keep = pk > 0
        k = f"enc{i}."
        z[k + "f1.w"], z[k + "f1.b"], z[k + "f2.w"] = z[k + "f1.w"][:, keep], z[k + "f1.b"][keep], z[k + "f2.w"][keep]
        print(f"enc{i}: kept {keep.sum()} of {keep.size} units", file=sys.stderr)
    np.savez(dst, **z)


if __name__ == "__main__":
    main()
