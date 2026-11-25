"""The torch model, training, and export to the numpy weights file.

    python -m training.model.model --train data/train_en.jsonl --out timmytime/weights/timmytime-en.npz

Training only; running the model needs numpy only (timmytime/runtime.py). Every layer
here is mirrored op for op there, so keep the two in step.
"""

import argparse
import calendar
import json
import math
import os
import random
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from timmytime.resolve import resolve
from timmytime.runtime import BOS, EOS, MAX_BYTES, MAX_OUT, PAD, UNK, VOCAB, encode_bytes, encode_code


class Cfg:
    byte_d = 64  # narrow byte embedding: the first conv is the costliest step at run time
    d = 256
    heads = 4
    ff = 1024
    n_enc = 4
    n_dec = 2
    stride = 2  # how much the second conv shortens the byte sequence before attention


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads

    def forward(self, q, k, v, key_pad=None, causal=False):
        b, t, d = q.shape
        s = k.shape[1]
        hd = d // self.heads
        q = q.view(b, t, self.heads, hd).transpose(1, 2)
        k = k.view(b, s, self.heads, hd).transpose(1, 2)
        v = v.view(b, s, self.heads, hd).transpose(1, 2)
        mask = None
        if key_pad is not None:
            mask = ~key_pad[:, None, None, :]
        if causal:
            c = torch.ones(t, s, dtype=torch.bool, device=q.device).tril()
            mask = c if mask is None else mask & c
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out.transpose(1, 2).reshape(b, t, d)


class EncLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv, self.o = nn.Linear(c.d, 3 * c.d), nn.Linear(c.d, c.d)
        self.f1, self.f2 = nn.Linear(c.d, c.ff), nn.Linear(c.ff, c.d)
        self.attn = Attn(c.d, c.heads)

    def forward(self, x, pad):
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        x = x + self.o(self.attn(q, k, v, key_pad=pad))
        return x + self.f2(F.relu(self.f1(self.ln2(x))))


class DecLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(c.d), nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv, self.o = nn.Linear(c.d, 3 * c.d), nn.Linear(c.d, c.d)
        self.q, self.kv, self.o2 = nn.Linear(c.d, c.d), nn.Linear(c.d, 2 * c.d), nn.Linear(c.d, c.d)
        self.f1, self.f2 = nn.Linear(c.d, c.ff), nn.Linear(c.ff, c.d)
        self.attn = Attn(c.d, c.heads)

    def forward(self, y, mem, mem_pad):
        q, k, v = self.qkv(self.ln1(y)).chunk(3, -1)
        y = y + self.o(self.attn(q, k, v, causal=True))
        mk, mv = self.kv(mem).chunk(2, -1)
        y = y + self.o2(self.attn(self.q(self.ln2(y)), mk, mv, key_pad=mem_pad))
        return y + self.f2(F.relu(self.f1(self.ln3(y))))


class TimmyTime(nn.Module):
    def __init__(self, c=Cfg):
        super().__init__()
        self.c = c
        self.byte_emb = nn.Embedding(257, c.byte_d)
        self.conv1 = nn.Conv1d(c.byte_d, c.d, 5, padding=2)
        self.conv2 = nn.Conv1d(c.d, c.d, 5, stride=c.stride, padding=2)
        self.enc_pos = nn.Parameter(torch.randn(MAX_BYTES // c.stride + 1, c.d) * 0.02)
        self.enc = nn.ModuleList(EncLayer(c) for _ in range(c.n_enc))
        self.enc_ln = nn.LayerNorm(c.d)
        self.tok_emb = nn.Embedding(len(VOCAB), c.d)
        self.dec_pos = nn.Parameter(torch.randn(MAX_OUT + 1, c.d) * 0.02)
        self.dec = nn.ModuleList(DecLayer(c) for _ in range(c.n_dec))
        self.dec_ln = nn.LayerNorm(c.d)
        for emb in (self.byte_emb, self.tok_emb):  # tok_emb is also the output layer: keep logits small at start
            nn.init.normal_(emb.weight, 0, 0.02)

    def encode(self, b):
        pad = b == 0
        keep = (~pad)[:, None, :].float()
        x = self.byte_emb(b).transpose(1, 2) * keep  # padding must look like the conv's own zero padding
        x = F.relu(self.conv1(x)) * keep
        x = F.relu(self.conv2(x)).transpose(1, 2)
        pad = pad[:, :: self.c.stride][:, : x.shape[1]]
        x = x + self.enc_pos[: x.shape[1]]
        for layer in self.enc:
            x = layer(x, pad)
        return self.enc_ln(x), pad

    def forward(self, b, tgt_in):
        mem, pad = self.encode(b)
        y = self.tok_emb(tgt_in) + self.dec_pos[: tgt_in.shape[1]]
        for layer in self.dec:
            y = layer(y, mem, pad)
        return self.dec_ln(y) @ self.tok_emb.weight.T

    def export(self, path: Path):
        """Weights in the numpy layout timmytime/runtime.py reads (float16 on disk)."""
        sd = {k: v.detach().float().cpu().numpy() for k, v in self.state_dict().items()}
        w = {"byte_emb": sd["byte_emb.weight"], "tok_emb": sd["tok_emb.weight"], "enc_pos": sd["enc_pos"],
             "dec_pos": sd["dec_pos"], "conv1.w": sd["conv1.weight"], "conv1.b": sd["conv1.bias"],
             "conv2.w": sd["conv2.weight"], "conv2.b": sd["conv2.bias"]}
        for name in ("enc_ln", "dec_ln"):
            w[f"{name}.g"], w[f"{name}.b"] = sd[f"{name}.weight"], sd[f"{name}.bias"]
        for prefix, n in (("enc", self.c.n_enc), ("dec", self.c.n_dec)):
            for i in range(n):
                for key, val in sd.items():
                    if not key.startswith(f"{prefix}.{i}."):
                        continue
                    sub, kind = key[len(f"{prefix}.{i}.") :].rsplit(".", 1)
                    if sub.startswith("ln"):
                        w[f"{prefix}{i}.{sub}.{'g' if kind == 'weight' else 'b'}"] = val
                    else:  # Linear: store (in, out) so the runtime does x @ w
                        w[f"{prefix}{i}.{sub}.{'w' if kind == 'weight' else 'b'}"] = val.T if kind == "weight" else val
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = np.array([self.c.heads, self.c.n_enc, self.c.n_dec, self.c.stride])
        tmp = path.with_suffix(".tmp.npz")  # write then rename: a crash mid-write must not leave a broken file
        np.savez_compressed(tmp, config=cfg, **{k: v.astype(np.float16) for k, v in w.items()})
        os.replace(tmp, path)


# --- data and augmentation


def _typo(s: str, r: random.Random) -> str:
    i = r.randrange(len(s))
    if s[i].isdigit():
        return s  # never touch digits: that changes the meaning
    op = r.randrange(4)
    if op == 0:
        return s[:i] + s[i + 1 :]
    if op == 1 and i + 1 < len(s) and not s[i + 1].isdigit():
        return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    if op == 2:
        return s[:i] + s[i] + s[i:]
    return s[:i] + s[i].swapcase() + s[i + 1 :]


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _swap_once(q: str, old: str, new: str) -> str | None:
    """q with the single whole-number occurrence of `old` replaced, or None if it isn't there exactly once."""
    hits = list(re.finditer(rf"(?<!\d){re.escape(old)}(?!\d)", q))
    if len(hits) != 1:
        return None
    m = hits[0]
    return q[: m.start()] + new + q[m.end() :]


# Longest count each unit draws; must match the range the output vocabulary can write.
_MAX_COUNT = {"day": 365, "week": 52, "month": 36, "year": 30}


def _new_year(r: random.Random) -> int:
    """Mostly recent years, and every year the vocabulary has some of the time."""
    return r.randint(1960, 2035) if r.random() < 0.8 else r.randint(1800, 2199)


def renumber(q: str, code: str, r: random.Random) -> tuple[str, str]:
    """Give the numbers a question shares with its code new random values, in both.

    Copying digits is the model's weakest skill; this makes endless fresh practice
    from every row without new LLM calls. Only numbers that appear literally, once,
    are touched, so the label stays right.
    """
    q0, words = q, code.split()
    for i, w in enumerate(words):
        if m := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", w):  # a full date
            y, mo, d = int(m[1]), int(m[2]), int(m[3])
            ny, nm = _new_year(r), r.randint(1, 12)  # all-digit forms: month too
            nd = r.randint(1, calendar.monthrange(ny, nm)[1])
            for old, new in ((w, f"{ny}-{nm:02d}-{nd:02d}"), (f"{mo}/{d}/{y}", f"{nm}/{nd}/{ny}"),
                             (f"{mo:02d}/{d:02d}/{y}", f"{nm:02d}/{nd:02d}/{ny}")):
                if (q2 := _swap_once(q, old, new)) is not None:
                    q, words[i] = q2, f"{ny}-{nm:02d}-{nd:02d}"
                    break
            else:  # "the 7th of May 2004": year and day separately
                nd = r.randint(1, calendar.monthrange(ny, mo)[1])
                q2 = _swap_once(q, str(y), str(ny))
                q3 = q2 and _swap_once(q2, str(d), str(nd))
                if q3:
                    q, words[i] = q3, f"{ny}-{mo:02d}-{nd:02d}"
        elif m := re.fullmatch(r"\*-(\d{2})-(\d{2})", w):
            nd = r.randint(1, calendar.monthrange(2023, int(m[1]))[1])  # no year: no Feb 29
            if (q2 := _swap_once(q, str(int(m[2])), str(nd))) is not None:
                q, words[i] = q2, f"*-{m[1]}-{nd:02d}"
        elif m := re.fullmatch(r"(\d{4})(-\d{2})?", w):  # a year, or year-month
            ny = _new_year(r)
            if (q2 := _swap_once(q, m[1], str(ny))) is not None:
                q, words[i] = q2, f"{ny}{m[2] or ''}"
        elif re.fullmatch(r"q[1-4]", w):
            nq = r.randint(1, 4)
            hits = re.findall(rf"\b[qQ]{w[1]}\b", q)
            if len(hits) == 1:
                q, words[i] = re.sub(rf"\b([qQ]){w[1]}\b", rf"\g<1>{nq}", q), f"q{nq}"
        elif re.fullmatch(r"-?\d+", w) and i > 0 and words[i - 1] in ("day", "week", "month", "year", "last", "next") \
                and abs(int(w)) >= 2:
            n = r.randint(2, 30 if words[i - 1] in ("day", "last", "next") else 12)
            unit = words[i - 1] if words[i - 1] in _MAX_COUNT else words[i + 1] if i + 1 < len(words) else ""
            if unit in _MAX_COUNT and r.random() < 0.2:  # counts the common range never reaches
                n = r.randint(13, _MAX_COUNT[unit])
            if (q2 := _swap_once(q, str(abs(int(w))), str(n))) is not None:
                q, words[i] = q2, str(-n if w.startswith("-") else n)
    new_code = " ".join(words)
    if new_code != code and not resolve(new_code, datetime(2026, 8, 12)):
        return q0, code  # e.g. a range swapped into the wrong order: keep the original
    return q, new_code


_MONTHS = "january|february|march|april|june|july|august|september|october|november|december"
_WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"


_SHORT = {"weeks": ["wks", "w", "wk"], "week": ["wk"], "days": ["d", "dys"], "months": ["mos", "mo", "mths"],
          "month": ["mo", "mth"], "years": ["yrs", "y"], "year": ["yr"]}


def respell(q: str, code: str, r: random.Random) -> str:
    """Ways people write dated words that the LLM rarely produces. The label stays the same."""
    if code == "none":
        if r.random() < 0.06:  # a year-like amount of money is not a year: "it cost 2019 usd"
            amount = f"{r.randint(1950, 2035)} {r.choice(['usd', 'dollars', 'eur', 'euros', 'bucks', 'gbp', 'chf'])}"
            q = r.choice([f"{q} - it cost {amount}", f"{q}, total {amount}", f"{amount} for {q}"])
        return q
    if r.random() < 0.3:  # "last 3 weeks" -> "last 3 wks" / "last 3w"
        q = re.sub(r"\b(\d+) ?(weeks|week|days|months|month|years|year)\b",
                   lambda m: f"{m[1]}{r.choice(['', ' '])}{r.choice(_SHORT[m[2].lower()])}", q, flags=re.I)
    if r.random() < 0.3:
        q = re.sub(r"\byesterday\b", lambda m: r.choice(["yday", "yest", "ystrdy", "yesterdy"]), q, flags=re.I)
    if r.random() < 0.2:
        q = re.sub(rf"\b({_WEEKDAYS})\b(?!\s+(morning|afternoon|evening|night))",
                   lambda m: f"{m[1]} {r.choice(['morning', 'afternoon', 'evening', 'night'])}", q, count=1, flags=re.I)
    if r.random() < 0.35:  # a typo inside a month name: "janary", "febuary"
        m = re.search(rf"\b({_MONTHS})\b", q, re.I)
        if m:
            w = m[1]
            i = r.randrange(1, len(w) - 1)
            q = q[: m.start()] + w[:i] + w[i + 1 :] + q[m.end() :]
    return q


def augment(q: str, code: str, pool_none: list[str], pool_dated: list[tuple[str, str]], r: random.Random):
    if code != "none" and r.random() < 0.5:
        q, code = renumber(q, code, r)
    q = respell(q, code, r)
    k = r.random()
    if k < 0.08 and pool_none:  # extra no-date text around the question; label unchanged
        other = r.choice(pool_none)
        q = f"{other} {q}" if r.random() < 0.5 else f"{q} {other}"
    elif k < 0.13 and code != "none" and ";" not in code and pool_dated:  # two dated questions -> two ranges
        q2, c2 = r.choice(pool_dated)
        if c2 != "none" and ";" not in c2:
            q, code = f"{q}{r.choice([' and ', '. ', ', also ', ' & '])}{q2}", f"{code} ; {c2}"
    case = r.random()
    if case < 0.3:
        q = q.lower()
    elif case < 0.35:  # people also type in ALL CAPS
        q = q.upper()
    if r.random() < 0.1:
        q = _strip_accents(q)
    for _ in range(r.choice([0, 0, 0, 1, 1, 2])):
        if q:
            q = _typo(q, r)
    return q, code


def batches(rows, size, r, augment_on, pools):
    """Padded batches of similar-length questions (less padding = faster), in random order."""
    items = [augment(q, c, *pools, r) if augment_on else (q, c) for q, c in rows]
    order = list(range(len(items)))
    r.shuffle(order)
    group = size * 32
    order = [j for g in range(0, len(order), group) for j in sorted(order[g : g + group], key=lambda j: len(items[j][0]))]
    starts = list(range(0, len(order), size))
    r.shuffle(starts)
    for i in starts:
        xs, ys = [], []
        for j in order[i : i + size]:
            q, code = items[j]
            ids = encode_code(code)
            if UNK in ids or len(ids) + 1 > MAX_OUT or len(q.encode()) > MAX_BYTES:
                continue
            xs.append(torch.from_numpy(encode_bytes(q)))
            ys.append(torch.tensor([BOS] + ids + [EOS]))
        if xs:  # round shapes up so the GPU sees only a few distinct ones (each new shape costs a compile)
            x = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
            y = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=PAD)
            yield (F.pad(x, (0, -x.shape[1] % 64), value=0), F.pad(y, (0, -y.shape[1] % 8), value=PAD))


def load_rows(paths):
    rows = []
    for p in paths:
        rows += [(r["q"], r["code"]) for r in map(json.loads, Path(p).open())]
    return rows


@torch.no_grad()
def greedy(model, b):
    """Batched greedy decode (training-time eval only)."""
    mem, pad = model.encode(b)
    out = torch.full((b.shape[0], 1), BOS, device=b.device)
    for _ in range(MAX_OUT):
        y = model.tok_emb(out) + model.dec_pos[: out.shape[1]]
        for layer in model.dec:
            y = layer(y, mem, pad)
        nxt = (model.dec_ln(y[:, -1]) @ model.tok_emb.weight.T).argmax(-1)
        out = torch.cat([out, nxt[:, None]], 1)
        if (out == EOS).any(1).all():
            break
    return out[:, 1:]


def seq_acc(model, rows, device, size=512):
    model.eval()
    good = total = 0
    for b, y in batches(rows, size, random.Random(0), False, None):
        pred = greedy(model, b.to(device)).cpu()
        for p, t in zip(pred.tolist(), y[:, 1:].tolist()):
            p = p[: p.index(EOS)] if EOS in p else p
            t = t[: t.index(EOS)]
            good += p == t
            total += 1
    model.train()
    return good / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("timmytime/weights/timmytime-en.npz"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0, help="starting weights and the dev split")
    ap.add_argument("--data-seed", type=int, default=None,
                    help="batch order and augmentation (default: --seed); runs that differ only here can be averaged")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--n-dec", type=int, default=2)
    ap.add_argument("--n-enc", type=int, default=4)
    ap.add_argument("--resume", type=Path, help="a .pt checkpoint to continue from")
    ap.add_argument("--start-epoch", type=int, default=0, help="epochs already done (keeps the learning-rate schedule)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(int(os.environ.get("TIMMYTIME_TORCH_THREADS", "4")))  # training.guard sets this
    r = random.Random(a.seed)
    rows = load_rows(a.train)
    r.shuffle(rows)
    n_dev = max(500, len(rows) // 20)
    dev, train = rows[:n_dev], rows[n_dev:]
    if a.data_seed is not None:
        r = random.Random(a.data_seed)
    pools = ([q for q, c in train if c == "none"], [(q, c) for q, c in train if c != "none"])
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    Cfg.stride, Cfg.n_dec, Cfg.n_enc = a.stride, a.n_dec, a.n_enc
    model = TimmyTime().to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M weights, {len(train)} train, {len(dev)} dev, {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    steps = a.epochs * math.ceil(len(train) / a.batch)
    warm = max(1, min(1000, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    if a.resume:
        state = torch.load(a.resume, map_location=device)
        old = state["tok_emb.weight"]
        if len(old) < len(VOCAB):  # words added to VOCAB since: keep the old rows, new ones start small
            state["tok_emb.weight"] = torch.cat([old, torch.randn(len(VOCAB) - len(old), old.shape[1], device=device) * 0.02])
        model.load_state_dict(state)
        for _ in range(a.start_epoch * math.ceil(len(train) / a.batch)):
            sched.step()
    best, t0 = -1.0, time.time()
    for epoch in range(a.start_epoch, a.epochs):
        total, n = 0.0, 0
        for b, y in batches(train, a.batch, r, True, pools):
            b, y = b.to(device), y.to(device)
            logits = model(b, y[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1),
                                   ignore_index=PAD, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
            if device == "mps" and n % 50 == 0:
                torch.mps.empty_cache()  # cached blocks for old batch shapes otherwise pile up until the cap
        acc = seq_acc(model, dev, device)
        if device == "mps":
            torch.mps.empty_cache()
        print(f"epoch {epoch + 1:2d}  loss {total / n:.4f}  dev exact {acc:.2%}  {time.time() - t0:.0f}s", flush=True)
        if acc >= best:
            best = acc
            model.export(a.out)
            torch.save(model.state_dict(), a.out.with_suffix(".tmp.pt"))
            os.replace(a.out.with_suffix(".tmp.pt"), a.out.with_suffix(".pt"))
    print(f"best dev exact {best:.2%}, saved {a.out}")


if __name__ == "__main__":
    main()
