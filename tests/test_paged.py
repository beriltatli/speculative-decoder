import math
import random

import pytest
import torch

from cache.kv import KVCache
from cache.paged import BlockAllocator, BlockTable, OutOfBlocks
from lm.load import tiny_model


def test_exhaustion_raises() -> None:
    allocator = BlockAllocator(3)
    for _ in range(3):
        allocator.allocate()
    with pytest.raises(OutOfBlocks):
        allocator.allocate()


def test_double_free_and_foreign_free_raise() -> None:
    allocator = BlockAllocator(2)
    block = allocator.allocate()
    allocator.free(block)
    with pytest.raises(ValueError):
        allocator.free(block)
    with pytest.raises(ValueError):
        allocator.free(99)


def test_failed_grow_changes_nothing() -> None:
    allocator = BlockAllocator(3)
    table = BlockTable()
    table.grow_to(8, allocator, block_size=4)
    with pytest.raises(OutOfBlocks):
        table.grow_to(13, allocator, block_size=4)
    assert table.length == 8 and len(table.blocks) == 2 and allocator.num_free == 1


@pytest.mark.parametrize(("length", "blocks"), [(9, 3), (8, 2), (5, 2), (4, 1), (1, 1), (0, 0)])
def test_shrink_keeps_ceil_length_over_block_size(length: int, blocks: int) -> None:
    allocator = BlockAllocator(4)
    table = BlockTable()
    table.grow_to(9, allocator, block_size=4)
    table.shrink_to(length, allocator, block_size=4)
    assert len(table.blocks) == blocks
    assert allocator.num_free == 4 - blocks


def test_slot_arithmetic() -> None:
    table = BlockTable(blocks=[7, 2], length=6)
    assert table.slots(0, 6, block_size=4).tolist() == [28, 29, 30, 31, 8, 9]
    assert table.slots(3, 5, block_size=4).tolist() == [31, 8]


def _cache(block_size: int, num_blocks: int) -> KVCache:
    return KVCache(n_layers=2, n_kv_heads=2, head_dim=8, num_blocks=num_blocks, block_size=block_size,
                   dtype=torch.float32, device="cpu")


def test_fragmentation_hand_case() -> None:
    cache = _cache(block_size=4, num_blocks=8)
    for seq_id, length in enumerate([5, 8, 1]):
        cache.add(seq_id)
        cache.append([seq_id], [length])
    stats = cache.stats()
    # Blocks: ceil(5/4) + ceil(8/4) + ceil(1/4) = 2 + 2 + 1. Slots: 20 allocated, 14 used.
    assert stats.blocks_in_use == 5
    assert stats.slots_allocated == 20
    assert stats.slots_used == 14
    assert stats.waste == pytest.approx(0.3)
    # 2 (k and v) * 2 layers * 2 heads * 8 dims * 4 bytes
    assert stats.bytes_per_slot == 256
    assert stats.bytes_in_use == 20 * 256


def test_contiguous_cache_is_one_block_per_sequence() -> None:
    config = tiny_model(seed=0).config
    cache = KVCache.contiguous(config, max_batch=4, max_seq_len=64, dtype=torch.float32, device="cpu")
    for seq_id, length in enumerate([5, 8, 1]):
        cache.add(seq_id)
        cache.append([seq_id], [length])
    stats = cache.stats()
    assert stats.blocks_in_use == 3
    assert stats.waste == pytest.approx(1 - 14 / 192)


def test_peak_survives_free() -> None:
    cache = _cache(block_size=4, num_blocks=8)
    cache.add(0)
    cache.append([0], [12])
    cache.free(0)
    assert cache.stats().blocks_in_use == 0
    assert cache.stats().peak_blocks_in_use == 3


def test_random_operations_keep_free_list_consistent() -> None:
    rng = random.Random(0)
    block_size = 3
    cache = _cache(block_size=block_size, num_blocks=40)
    live: list[int] = []
    next_id = 0
    for _ in range(3000):
        op = rng.random()
        if op < 0.2 or not live:
            cache.add(next_id)
            live.append(next_id)
            next_id += 1
        elif op < 0.6:
            seq_id = rng.choice(live)
            try:
                cache.append([seq_id], [rng.randint(1, 7)])
            except OutOfBlocks:
                pass
        elif op < 0.85:
            seq_id = rng.choice(live)
            cache.truncate(seq_id, rng.randint(0, cache.length(seq_id)))
        else:
            seq_id = live.pop(rng.randrange(len(live)))
            cache.free(seq_id)
        cache.allocator.check()
        owned = [b for t in cache.tables.values() for b in t.blocks]
        assert len(owned) == len(set(owned)) == cache.allocator.num_used
        for table in cache.tables.values():
            assert len(table.blocks) == math.ceil(table.length / block_size)
