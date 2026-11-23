"""The numpy runner must match the torch model, and the vocabulary must round-trip."""

import random
from datetime import date

import numpy as np
import pytest

from tlm.meanings import sample
from tlm.runtime import UNK, Parser, decode_code, encode_bytes, encode_code

torch = pytest.importorskip("torch")


def test_codes_round_trip_through_tokens():
    r = random.Random(0)
    for _ in range(5000):
        code = sample(r)[1]
        ids = encode_code(code)
        assert UNK not in ids, code
        assert decode_code(ids) == code
    assert decode_code(encode_code("*-06-04 ; 2024-03 .. now")) == "*-06-04 ; 2024-03 .. now"


def test_numpy_matches_torch(tmp_path):
    from tlm.model import TLM, greedy

    torch.manual_seed(0)
    model = TLM().eval()
    for p in model.parameters():  # random weights, scaled so decoding picks varied tokens
        p.data.normal_(0, 0.3)
    path = tmp_path / "w.npz"
    model.export(path)
    parser = Parser(path)
    texts = ["what did I decide yesterday", "x", "vorige week, wat heb ik gedaan?", "上周一", "a" * 300]
    batch = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(encode_bytes(t)) for t in texts], batch_first=True)
    want = greedy(model, batch).tolist()
    for text, ids in zip(texts, want):
        ids = ids[: ids.index(2)] if 2 in ids else ids
        got_code, _ = parser.code(text)
        assert got_code == decode_code(ids), text

    assert len({tuple(ids[:5]) for ids in want}) > 1, "random model too uniform to prove anything"


def test_odd_input_never_crashes():
    from tlm.runtime import WEIGHTS

    if not WEIGHTS.exists():
        pytest.skip("no trained weights")
    parser = Parser()
    for text in ["", " ", "\x00\x00", "🧠" * 500, "a" * 10_000, "\ud800 broken surrogate", "yesterday" * 100,
                 "SELECT * FROM t WHERE d > '2024-01-01'", "\n\t\r", "٣ أيام", "1" * 300]:
        ranges = parser.parse(text, None)
        assert isinstance(ranges, list)


def test_date_anchor_is_a_day_in_tz():
    import tlm
    for tz in ("UTC", "America/New_York", "Asia/Tokyo"):
        assert tlm.parse("what did I do yesterday", date(2026, 11, 20), tz) == [(date(2026, 11, 19), date(2026, 11, 19))]
