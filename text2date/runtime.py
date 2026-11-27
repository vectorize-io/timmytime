"""Run the model with numpy only (no torch), then the calendar code.

    from text2date.runtime import Parser
    Parser().parse("what did I decide yesterday", anchor)   # -> [DateRange(...)]

The model reads the question's UTF-8 bytes and writes a meaning-code line, one
word per step (greedy). The vocabulary and the network here must match
text2date/model.py op for op; tests/test_runtime.py checks they agree.
"""

import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from text2date.resolve import DateRange, resolve

MAX_BYTES = 256
MIN_CONFIDENCE = float(os.environ.get("TEXT2DATE_MIN_CONFIDENCE", "0.5"))  # picked on data/tune_*.jsonl by text2date/tune.py
MAX_OUT = 24
WEIGHTS = Path(os.environ.get("TEXT2DATE_WEIGHTS", Path(__file__).parent / "weights" / "model.safetensors"))

# --- output vocabulary: one token per code word; dates split into year / month / day.
_WORDS = ("none now .. ; day week month quarter year last next before after around early mid late h1 h2 "
          "recent otherday while longago couple few weekend spring summer autumn winter q1 q2 q3 q4 * "
          "mon tue wed thu fri sat sun jan feb mar apr may jun jul aug sep oct nov dec").split()
VOCAB = (["<pad>", "<bos>", "<eos>", "<unk>"] + _WORDS + [str(n) for n in range(-99, 100)]
         + [str(y) for y in range(1900, 2100)] + [f"{y}s" for y in range(1900, 2100, 10)]
         + [f"M{m:02d}" for m in range(1, 13)] + [f"D{d:02d}" for d in range(1, 32)]
         # Added later: new words go at the end so older weights keep their ids.
         + "easter thanksgiving cny midautumn dragonboat ramadan eidfitr eidadha".split()
         + [str(y) for y in (*range(1800, 1900), *range(2100, 2200))] + [f"{y}s" for y in (*range(1800, 1900, 10), 2100)]
         + [str(n) for n in (*range(-365, -99), *range(100, 366))])  # "100 days ago"
TOKEN = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, EOS, UNK = 0, 1, 2, 3
_DATE = re.compile(r"(\d{4}|\*)(?:-(\d{2})(?:-(\d{2}))?)?$")


def encode_code(code: str) -> list[int]:
    """Code line -> token ids (no BOS/EOS). Unknown words map to <unk>."""
    out = []
    for word in code.split():
        m = _DATE.match(word)
        if m and (m[2] or word != "*") and word not in TOKEN:
            out.append(TOKEN.get(m[1], UNK))
            out += [TOKEN.get(f"M{m[2]}", UNK)] if m[2] else []
            out += [TOKEN.get(f"D{m[3]}", UNK)] if m[3] else []
        else:
            out.append(TOKEN.get(word, UNK))
    return out


def decode_code(ids: list[int]) -> str:
    words: list[str] = []
    for i in ids:
        t = VOCAB[i]
        if t[0] in "MD" and len(t) == 3 and words:
            words[-1] += "-" + t[1:]
        else:
            words.append(t)
    return " ".join(words)


def encode_bytes(text: str) -> np.ndarray:
    """UTF-8 bytes + 1 (0 is padding), cut at MAX_BYTES."""
    return np.frombuffer(text.encode("utf-8", "replace")[:MAX_BYTES], dtype=np.uint8).astype(np.int64) + 1


# --- the network, numpy version


def _norm(x):
    """LayerNorm without its scale and shift (those are folded into the next matmul at load)."""
    d = x - x.mean(-1, keepdims=True)
    return d / np.sqrt((d * d).mean(-1, keepdims=True) + 1e-5)


def _attn_qkv(qkv, heads):
    """Self-attention over the whole sequence from the fused q|k|v projection, in few numpy calls."""
    t = qkv.shape[0]
    q, k, v = qkv.reshape(t, 3, heads, -1).transpose(1, 2, 0, 3)  # each (heads, t, hd)
    a = q @ k.transpose(0, 2, 1)  # the 1/sqrt(head size) scale is folded into the query weights at load
    a -= a.max(-1, keepdims=True)
    np.exp(a, out=a)
    a /= a.sum(-1, keepdims=True)
    return (a @ v).transpose(1, 0, 2).reshape(t, -1)


def _attn1(q, kt, v, heads):
    """Attention for one query row. q: (1, d); kt: (heads, hd, s); v: (heads, s, hd)."""
    a = q.reshape(heads, 1, -1) @ kt
    e = np.exp(a - a.max(-1, keepdims=True))
    return ((e @ v) / e.sum(-1, keepdims=True)).reshape(1, -1)


def _conv(x, w, b, k, stride):
    """Conv1d over time, padding k//2. x: (T, C_in); w: (C_in * k, C_out), laid out like torch's (C_in, k)."""
    x = np.pad(x, ((k // 2, k // 2), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(x, k, axis=0)[::stride]  # (T', C_in, k)
    return win.reshape(win.shape[0], -1) @ w + b


_DTYPES = {"F16": np.float16, "F32": np.float32, "I64": np.int64}


def load_weights(path: Path | str) -> dict[str, np.ndarray]:
    """A .safetensors file (the released model) or a training .npz, as {name: array}, with numpy only."""
    if str(path).endswith(".npz"):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    head = json.loads(raw[8:8 + n])
    head.pop("__metadata__", None)
    return {k: np.frombuffer(raw, _DTYPES[v["dtype"]], int(np.prod(v["shape"])), 8 + n + v["data_offsets"][0])
            .reshape(v["shape"]) for k, v in head.items()}


class Parser:
    def __init__(self, weights: Path | str = WEIGHTS):
        if not Path(weights).exists():
            raise FileNotFoundError(f"no weights at {weights}. In a git checkout, run scripts/get_weights.sh; "
                                    "the installed package ships them.")
        z = load_weights(weights)
        # Row-major float32 copies: column-major weights make each matmul ~8x slower.
        self.w = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in z.items() if k != "config"}
        for name in ("conv1", "conv2"):  # (out, in, k) -> (in*k, out), the shape _conv multiplies by
            cw = self.w.pop(f"{name}.w")
            self.w[f"{name}.k"] = cw.shape[2]
            self.w[f"{name}.w"] = np.ascontiguousarray(cw.reshape(cw.shape[0], -1).T)
        self.w["tok_out"] = np.ascontiguousarray(self.w["tok_emb"].T)
        self.w["tok_out.b"] = np.zeros(self.w["tok_out"].shape[1], np.float32)
        cfg = [int(x) for x in z["config"]]
        self.heads, self.n_enc, self.n_dec = cfg[:3]
        self.stride = cfg[3] if len(cfg) > 3 else 2
        self._fold()

    def _fold(self):
        """Exact rewrites that save numpy calls per question (each costs microseconds, whatever its size).

        LayerNorm's scale g and shift b move into the matmul after it: (n*g+b) @ W + c = n @ (g*W) + (b@W + c).
        Attention's 1/sqrt(head size) moves into the query weights.
        """
        w = self.w

        def fold(ln, lin, wkey=None, bkey=None):
            wkey, bkey = wkey or f"{lin}.w", bkey or f"{lin}.b"
            g, b = w[f"{ln}.g"], w[f"{ln}.b"]
            w[bkey] = (b @ w[wkey] + w[bkey]).astype(np.float32)
            w[wkey] = np.ascontiguousarray(g[:, None] * w[wkey])

        d = w["tok_emb"].shape[1]
        scale = np.float32(1.0 / (d // self.heads) ** 0.5)
        for i in range(self.n_enc):
            fold(f"enc{i}.ln1", f"enc{i}.qkv")
            fold(f"enc{i}.ln2", f"enc{i}.f1")
            w[f"enc{i}.qkv.w"][:, :d] *= scale
            w[f"enc{i}.qkv.b"][:d] *= scale
        for i in range(self.n_dec):
            fold(f"dec{i}.ln1", f"dec{i}.qkv")
            fold(f"dec{i}.ln2", f"dec{i}.q")
            fold(f"dec{i}.ln3", f"dec{i}.f1")
            fold("enc_ln", f"dec{i}.kv")
            w[f"dec{i}.qkv.w"][:, :d] *= scale
            w[f"dec{i}.qkv.b"][:d] *= scale
            w[f"dec{i}.q.w"] *= scale
            w[f"dec{i}.q.b"] *= scale
        fold("dec_ln", "tok_out", "tok_out", "tok_out.b")

    def code(self, text: str) -> tuple[str, float]:
        """The meaning code for a question, and the lowest token probability on the way."""
        w, h = self.w, self.heads
        if not text.strip():  # blank text has no date; don't let the model guess
            return "none", 1.0
        b = encode_bytes(text)
        x = w["byte_emb"][b]
        x = np.maximum(_conv(x, w["conv1.w"], w["conv1.b"], w["conv1.k"], 1), 0)
        x = np.maximum(_conv(x, w["conv2.w"], w["conv2.b"], w["conv2.k"], self.stride), 0)
        x = x + w["enc_pos"][: x.shape[0]]
        for i in range(self.n_enc):
            p = f"enc{i}."
            x = x + _attn_qkv(_norm(x) @ w[p + "qkv.w"] + w[p + "qkv.b"], h) @ w[p + "o.w"] + w[p + "o.b"]
            y = np.maximum(_norm(x) @ w[p + "f1.w"] + w[p + "f1.b"], 0)
            x = x + y @ w[p + "f2.w"] + w[p + "f2.b"]
        mem = _norm(x)
        n, d = mem.shape
        hd = d // h
        # Keys pre-laid out per head as (h, hd, n) and values as (h, n, hd): the decoder adds one token
        # per step, and this saves the reshapes and transposes a general attention would redo each time.
        cross = []
        for i in range(self.n_dec):
            ck, cv = np.split(mem @ w[f"dec{i}.kv.w"] + w[f"dec{i}.kv.b"], 2, -1)
            cross.append((np.ascontiguousarray(ck.reshape(n, h, hd).transpose(1, 2, 0)),
                          np.ascontiguousarray(cv.reshape(n, h, hd).transpose(1, 0, 2))))
        self_k = [np.empty((h, hd, MAX_OUT), np.float32) for _ in range(self.n_dec)]
        self_v = [np.empty((h, MAX_OUT, hd), np.float32) for _ in range(self.n_dec)]
        out, tok, low = [], BOS, 1.0
        for step in range(MAX_OUT):
            y = w["tok_emb"][tok][None] + w["dec_pos"][step][None]
            for i in range(self.n_dec):
                p = f"dec{i}."
                q, k, v = np.split(_norm(y) @ w[p + "qkv.w"] + w[p + "qkv.b"], 3, -1)
                self_k[i][:, :, step] = k.reshape(h, hd)
                self_v[i][:, step] = v.reshape(h, hd)
                y = y + _attn1(q, self_k[i][:, :, : step + 1], self_v[i][:, : step + 1], h) @ w[p + "o.w"] + w[p + "o.b"]
                q = _norm(y) @ w[p + "q.w"] + w[p + "q.b"]
                y = y + _attn1(q, *cross[i], h) @ w[p + "o2.w"] + w[p + "o2.b"]
                z = np.maximum(_norm(y) @ w[p + "f1.w"] + w[p + "f1.b"], 0)
                y = y + z @ w[p + "f2.w"] + w[p + "f2.b"]
            logits = (_norm(y) @ w["tok_out"] + w["tok_out.b"])[0]
            tok = int(logits.argmax())
            e = np.exp(logits - logits[tok])
            low = min(low, float(1 / e.sum()))
            if tok == EOS:
                break
            out.append(tok)
        assert y.dtype == np.float32  # float64 creeping in makes this ~4x slower
        return decode_code(out), low

    def parse(self, text: str, anchor: datetime | date | None = None, tz: str = "UTC",
              min_confidence: float = MIN_CONFIDENCE) -> list[DateRange]:
        """Date ranges in a question. [] = no date. Never raises on odd text.

        A date the model is less sure of than `min_confidence` (its least likely output step) is
        dropped: a wrong window hurts recall more than a missing one.
        """
        if isinstance(anchor, date) and not isinstance(anchor, datetime):  # a calendar day in `tz`, not UTC
            anchor = datetime.combine(anchor, datetime.min.time(), ZoneInfo(tz))
        out: list[DateRange] = []
        for part in _chunks(text):  # the model reads MAX_BYTES at a time; longer text goes in pieces
            try:
                code, confidence = self.code(part)
            except Exception:  # a model bug must never break recall
                continue
            if code != "none" and confidence >= min_confidence:
                for r in resolve(code, anchor, tz):  # same range twice ("yesterday yesterday") counts once
                    if r not in out:
                        out.append(r)
        return out


_SENTENCE_END = re.compile(r"(?<=[.!?。！？\n])\s*")


def _chunks(text: str) -> list[str]:
    """Whole sentences packed into pieces of at most MAX_BYTES; a longer sentence is cut."""
    if len(text.encode("utf-8", "replace")) <= MAX_BYTES:
        return [text]
    pieces, cur = [], ""
    for sentence in filter(None, _SENTENCE_END.split(text)):
        while len(sentence.encode("utf-8", "replace")) > MAX_BYTES:  # cut at a space if there is one
            head = sentence.encode("utf-8", "replace")[:MAX_BYTES].decode("utf-8", "ignore")
            head = head[: head.rfind(" ")] if " " in head[MAX_BYTES // 4 :] else head
            pieces, cur, sentence = pieces + [p for p in (cur, head) if p], "", sentence[len(head) :].lstrip()
        if len((cur + " " + sentence).encode("utf-8", "replace")) > MAX_BYTES and cur:
            pieces.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    return pieces + [cur] if cur else pieces
