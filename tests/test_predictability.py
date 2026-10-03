import random

import pytest

from bench.predictability import gzip_ratio, repeated_ngram_rate


def test_repeated_ngram_rate_on_hand_built_cases() -> None:
    assert repeated_ngram_rate("a b c d e f g h") == 0.0
    # 12 words -> 9 four-grams; the first 4 ("a b c d", "b c d a", "c d a b", "d a b c") are
    # new and the remaining 5 repeat one of them.
    assert repeated_ngram_rate("a b c d a b c d a b c d") == pytest.approx(5 / 9)
    assert repeated_ngram_rate("a b c") == 0.0


def test_gzip_ratio_orders_redundancy() -> None:
    rng = random.Random(0)
    random_text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(4000))
    repeated = "the same line again\n" * 200
    assert gzip_ratio(repeated) < 0.05
    assert gzip_ratio(random_text) > 0.55  # ~log2(27)/8 = 0.59 bits per char at best
    assert gzip_ratio(repeated) < gzip_ratio(random_text)
