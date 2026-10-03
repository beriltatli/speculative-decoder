import pytest
import torch

from baselines import cached_autoregressive
from cache.kv import KVCache
from lm.load import tiny_model
from scripts.block_size import SharedPool, waste, waste_table


def test_waste_counts_the_unused_tail_of_the_last_block() -> None:
    # 16 fills one block exactly; 17 needs a second block with 15 slots unused.
    result = waste([16, 17], 16)
    assert result["waste_fraction"] == pytest.approx(1 - 33 / 48)
    assert result["blocks_per_seq_p50"] == 1.5
    assert waste([16, 17], 1)["waste_fraction"] == 0.0


def test_contiguous_reserves_the_longest_sequence_for_everyone() -> None:
    table = waste_table({"c": [10, 30]}, max_new=5, k=4)["c"]["contiguous"]
    # Lengths reached: 14 and 34; every sequence reserves 30 + 5 + 4 + 1 = 40.
    assert table["reserved_slots"] == 40
    assert table["waste_fraction"] == pytest.approx(1 - 48 / 80)


def test_shared_pool_matches_a_private_cache_at_every_block_size() -> None:
    model = tiny_model(seed=0, dtype=torch.float64)
    prompts = [[1, 2, 3, 4, 5], list(range(7, 40)), [9] * 18]
    expected = [cached_autoregressive.generate(
        model, KVCache.for_model(model.config, 16, 16, torch.float64, "cpu"), p, 12) for p in prompts]
    pool = SharedPool(torch.device("cpu"))
    caches = [pool(model, -(-64 // b) + 1, b) for b in (64, 1, 4, 16)]
    assert len({c.k.data_ptr() for c in caches}) == 1
    for cache in caches:
        assert [cached_autoregressive.generate(model, cache, p, 12) for p in prompts] == expected
