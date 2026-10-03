import random

import pytest
import torch

from baselines import cached_autoregressive
from cache.kv import KVCache
from lm.load import tiny_model


@pytest.fixture(scope="module")
def tiny():
    return tiny_model(seed=0, dtype=torch.float64)


def new_cache(model) -> KVCache:
    return KVCache.for_model(model.config, num_blocks=128, block_size=16, dtype=torch.float64, device="cpu")


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_batched_plain_decoding_matches_one_at_a_time(tiny, batch: int) -> None:
    rng = random.Random(0)
    prompts = [[rng.randrange(256) for _ in range(rng.randint(1, 40))] for _ in range(24)]
    expected = [cached_autoregressive.generate(tiny, new_cache(tiny), p, 12) for p in prompts]
    cache = new_cache(tiny)
    for i in range(0, len(prompts), batch):
        assert cached_autoregressive.generate_batch(tiny, cache, prompts[i:i + batch], 12) == expected[i:i + batch]
    assert cache.allocator.num_used == 0 and not cache.tables


def test_batched_plain_decoding_drops_a_row_at_eos(tiny) -> None:
    prompts = [[5, 6, 7], list(range(30, 60))]
    full = [cached_autoregressive.generate(tiny, new_cache(tiny), p, 12) for p in prompts]
    eos = full[0][3]
    want = [cached_autoregressive.generate(tiny, new_cache(tiny), p, 12, eos_id=eos) for p in prompts]
    assert len(want[0]) <= 4
    assert cached_autoregressive.generate_batch(tiny, new_cache(tiny), prompts, 12, eos_id=eos) == want
