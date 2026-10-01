from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from cache.paged import BlockAllocator, BlockTable, FragmentationStats


@dataclass(frozen=True)
class SlotMap:
    write: torch.Tensor        # [n_valid] pool slots receiving the new tokens
    write_index: torch.Tensor  # [n_valid] rows of the flattened [B*T] new-token block that are real, not padding
    read: torch.Tensor         # [B, S] pool slot for every cached position after this append
    past: torch.Tensor         # [B] cached length before the append
    total: torch.Tensor        # [B] cached length after it


class KVCache:
    """Paged K/V for independent sequences. A slot is one position in a flat pool of
    `num_blocks * block_size` entries; each sequence maps its positions to slots through
    its block table."""

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.block_size = block_size
        self.device = torch.device(device)
        self.allocator = BlockAllocator(num_blocks)
        shape = (n_layers, num_blocks * block_size, n_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v = torch.zeros(shape, dtype=dtype, device=self.device)
        self.tables: dict[int, BlockTable] = {}
        self.bytes_per_slot = 2 * n_layers * n_kv_heads * head_dim * self.k.element_size()

    @classmethod
    def for_model(
        cls,
        config: PretrainedConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "KVCache":
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return cls(config.num_hidden_layers, config.num_key_value_heads, head_dim, num_blocks, block_size, dtype, device)

    @classmethod
    def contiguous(
        cls,
        config: PretrainedConfig,
        max_batch: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "KVCache":
        # The naive pre-allocated cache: one max_seq_len block per sequence. Expressing it
        # as a paged cache with a single huge block keeps the memory comparison on identical code.
        return cls.for_model(config, max_batch, max_seq_len, dtype, device)

    def add(self, seq_id: int) -> None:
        if seq_id in self.tables:
            raise ValueError(f"sequence {seq_id} already present")
        self.tables[seq_id] = BlockTable()

    def free(self, seq_id: int) -> None:
        self.tables[seq_id].shrink_to(0, self.allocator, self.block_size)
        del self.tables[seq_id]

    def length(self, seq_id: int) -> int:
        return self.tables[seq_id].length

    def append(self, seq_ids: list[int], n_new: list[int]) -> SlotMap:
        """Reserve slots for `n_new[b]` tokens on each sequence. Lengths advance immediately."""
        width = max(n_new)
        past = [self.tables[s].length for s in seq_ids]
        for seq_id, n in zip(seq_ids, n_new):
            table = self.tables[seq_id]
            table.grow_to(table.length + n, self.allocator, self.block_size)
        total = [p + n for p, n in zip(past, n_new)]

        write = torch.cat([self.tables[s].slots(p, t, self.block_size) for s, p, t in zip(seq_ids, past, total)])
        write_index = torch.cat([b * width + torch.arange(n) for b, n in enumerate(n_new)])
        # Padded read entries point at slot 0, which may hold another sequence's data.
        # They are harmless only because the attention mask excludes positions >= total.
        read = torch.zeros(len(seq_ids), max(total), dtype=torch.long)
        for b, (seq_id, t) in enumerate(zip(seq_ids, total)):
            read[b, :t] = self.tables[seq_id].slots(0, t, self.block_size)

        def put(x: list[int] | torch.Tensor) -> torch.Tensor:
            return torch.as_tensor(x, dtype=torch.long).to(self.device)

        return SlotMap(put(write), put(write_index), put(read), put(past), put(total))

    def write(self, layer: int, slots: SlotMap, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: [B, H_kv, T, d] for the tokens passed to `append`, right-padded to T."""
        b, h, t, d = k.shape
        rows = slots.write_index
        self.k[layer][slots.write] = k.transpose(1, 2).reshape(b * t, h, d)[rows]
        self.v[layer][slots.write] = v.transpose(1, 2).reshape(b * t, h, d)[rows]

    def gather(self, layer: int, slots: SlotMap) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, H_kv, S, d] copies of every cached position. A real paged-attention kernel
        reads blocks in place; this copy is O(length) per layer per step and is a known
        source of overhead in the benchmark."""
        return self.k[layer][slots.read].transpose(1, 2), self.v[layer][slots.read].transpose(1, 2)

    def truncate(self, seq_id: int, length: int) -> None:
        # Blocks wholly past `length` return to the free list. Entries past `length` inside
        # the kept last block are stale but unreachable: reads stop at `total`, and the next
        # append writes exactly those positions before any layer gathers them.
        self.tables[seq_id].shrink_to(length, self.allocator, self.block_size)

    def stats(self) -> FragmentationStats:
        return FragmentationStats(
            block_size=self.block_size,
            bytes_per_slot=self.bytes_per_slot,
            blocks_in_use=self.allocator.num_used,
            peak_blocks_in_use=self.allocator.peak_used,
            slots_used=sum(t.length for t in self.tables.values()),
        )
