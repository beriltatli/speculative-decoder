import math
from dataclasses import dataclass, field

import torch


class OutOfBlocks(RuntimeError):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        # LIFO free list: the most recently freed block is handed out next, so a fixed
        # sequence of operations always produces the same block ids.
        self._free = list(range(num_blocks - 1, -1, -1))
        self._used: set[int] = set()
        self.peak_used = 0

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return len(self._used)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocks(f"all {self.num_blocks} blocks in use")
        block = self._free.pop()
        self._used.add(block)
        self.peak_used = max(self.peak_used, len(self._used))
        return block

    def free(self, block: int) -> None:
        if block not in self._used:
            raise ValueError(f"block {block} is not allocated (double free or foreign id)")
        self._used.remove(block)
        self._free.append(block)

    def check(self) -> None:
        free = set(self._free)
        assert len(free) == len(self._free), "duplicate id in free list"
        assert not (free & self._used), "block both free and used"
        assert free | self._used == set(range(self.num_blocks)), "block leaked"


@dataclass
class BlockTable:
    blocks: list[int] = field(default_factory=list)
    length: int = 0

    def grow_to(self, length: int, allocator: BlockAllocator, block_size: int) -> None:
        needed = math.ceil(length / block_size) - len(self.blocks)
        # Checked up front so a failed grow leaves the table and allocator untouched;
        # the scheduler relies on this to preempt instead of half-admitting a sequence.
        if needed > allocator.num_free:
            raise OutOfBlocks(f"need {needed} blocks, {allocator.num_free} free")
        for _ in range(needed):
            self.blocks.append(allocator.allocate())
        self.length = length

    def shrink_to(self, length: int, allocator: BlockAllocator, block_size: int) -> None:
        if length > self.length:
            raise ValueError(f"cannot shrink length {self.length} to {length}")
        keep = math.ceil(length / block_size)
        for block in self.blocks[keep:]:
            allocator.free(block)
        del self.blocks[keep:]
        self.length = length

    def slots(self, start: int, stop: int, block_size: int) -> torch.Tensor:
        positions = torch.arange(start, stop)
        table = torch.tensor(self.blocks, dtype=torch.long)
        return table[positions // block_size] * block_size + positions % block_size


@dataclass(frozen=True)
class FragmentationStats:
    block_size: int
    bytes_per_slot: int
    blocks_in_use: int
    peak_blocks_in_use: int
    slots_used: int

    @property
    def slots_allocated(self) -> int:
        return self.blocks_in_use * self.block_size

    @property
    def waste(self) -> float:
        # Internal fragmentation only: the unused tail of each sequence's last block.
        # Fixed-size blocks have no external fragmentation.
        if self.slots_allocated == 0:
            return 0.0
        return 1.0 - self.slots_used / self.slots_allocated

    @property
    def bytes_in_use(self) -> int:
        return self.slots_allocated * self.bytes_per_slot

    @property
    def peak_bytes(self) -> int:
        return self.peak_blocks_in_use * self.block_size * self.bytes_per_slot
