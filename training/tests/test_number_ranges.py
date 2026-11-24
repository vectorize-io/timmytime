"""Days 29-31, counts above 30 and far years must reach the training targets.

With tied embeddings, an answer token that is never a target stays the same vector as <unk>,
and the model can never write it (issue #1).
"""

import random
import re

from training.data import en_templates, zh_templates
from training.data.meanings import sample
from training.model.model import renumber

DAY = re.compile(r"(?:\d{4}|\*)-\d{2}-(\d{2})")


def _days(codes):
    return {int(m[1]) for c in codes for m in DAY.finditer(c)}


def test_sampler_draws_every_day_of_the_month():
    r = random.Random(0)
    assert {29, 30, 31} <= _days(sample(r)[1] for _ in range(20_000))


def test_sampler_draws_counts_above_30():
    r = random.Random(0)
    codes = [sample(r)[1] for _ in range(20_000)]
    counts = {int(m[1]) for c in codes for m in re.finditer(r"\blast (\d+) (?:day|week|month|year)\b", c)}
    assert max(counts) > 30


def test_templates_draw_every_day_of_the_month():
    r = random.Random(0)
    assert {29, 30, 31} <= _days(en_templates.row(r)[1] for _ in range(20_000))
    assert {29, 30, 31} <= _days(zh_templates.t_before_after(r)[1] for _ in range(5_000))
    assert {29, 30, 31} <= _days(zh_templates.t_dayrange(r)[1] for _ in range(5_000))


def test_renumber_reaches_every_day_count_and_year():
    r = random.Random(0)
    rows = [("meeting on 2024-05-14", "2024-05-14"), ("what did I do 12 days ago", "day -12"),
            ("photos from 2019", "2019"), ("dinner on 14 May 2004", "2004-05-14")]
    out = [renumber(q, code, r)[1] for q, code in rows for _ in range(3_000)]
    assert {29, 30, 31} <= _days(out)
    assert any(abs(int(m[1])) > 30 for c in out if (m := re.fullmatch(r"day (-?\d+)", c)))
    assert any(not 1960 <= int(c) <= 2035 for c in out if re.fullmatch(r"\d{4}", c))
