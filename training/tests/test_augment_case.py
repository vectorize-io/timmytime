"""Training questions must sometimes be uppercase (issue #3).

Augmentation only lowercased, so ALL CAPS questions were out of distribution and mostly
came back as no date (Spanish dropped from 96.7% to 78.7%).
"""

import random

from training.model.model import augment


def test_augment_sometimes_uppercases():
    r = random.Random(0)
    out = [augment("qué hice la semana pasada", "week -1", [], [], r)[0] for _ in range(2_000)]
    assert any(q == q.upper() and q != q.lower() for q in out)
