import torch
from transformers import PreTrainedModel

from cache.kv import KVCache
from lm.forward import forward
from lm.sampling import probabilities


def verify(
    target: PreTrainedModel,
    cache: KVCache,
    seq_ids: list[int],
    contexts: list[list[int]],
    draft: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """One target forward over [context[-1], d_1, ..., d_k] per row. Output row i is the
    target's distribution for the token after d_i (row 0: for d_1 itself), and row k is the
    bonus position. Returns p [B, k+1, V] on CPU.

    Every row has exactly k+1 new tokens, so no padding is needed; positions run from
    len(context) - 1 to len(context) + k - 1, set by the cache's per-row past length."""
    for seq_id, ctx in zip(seq_ids, contexts):
        if cache.length(seq_id) != len(ctx) - 1:
            raise RuntimeError(f"seq {seq_id}: target cache holds {cache.length(seq_id)}, expected {len(ctx) - 1}")
    last = torch.tensor([[ctx[-1]] for ctx in contexts])
    input_ids = torch.cat([last, draft], dim=1).to(cache.device)
    k1 = input_ids.shape[1]
    logits = forward(target, cache, seq_ids, input_ids, [k1] * len(seq_ids))
    return probabilities(logits, temperature).cpu()
