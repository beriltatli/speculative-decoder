from collections.abc import Callable

import torch
from transformers import PreTrainedModel

from cache.kv import KVCache
from lm.forward import forward
from lm.sampling import next_token


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    cache: KVCache,
    prompt: list[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    eos_id: int | None = None,
    seq_id: int = 0,
    on_event: Callable[[str], None] | None = None,
) -> list[int]:
    device = cache.device
    cache.add(seq_id)
    try:
        logits = forward(model, cache, [seq_id], torch.tensor([prompt], device=device), [len(prompt)])[0, -1]
        out: list[int] = []
        for _ in range(max_new_tokens):
            token = next_token(logits, temperature, generator)
            out.append(token)
            if len(out) == 1 and on_event:
                on_event("first_token")
            if token == eos_id or len(out) == max_new_tokens:
                break
            logits = forward(model, cache, [seq_id], torch.tensor([[token]], device=device), [1])[0, 0]
        return out
    finally:
        cache.free(seq_id)
