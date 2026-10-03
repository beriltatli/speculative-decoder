from collections.abc import Callable

import torch
from transformers import PreTrainedModel

from lm.forward import logits as output_logits
from lm.sampling import next_token


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    prompt: list[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    eos_id: int | None = None,
    on_event: Callable[[str], None] | None = None,
) -> list[int]:
    """Recompute the whole prefix every step: O(n^2) attention work in total, on purpose.
    This is the floor that shows how much of any speedup is the cache, not the speculation."""
    ids = torch.tensor([prompt], device=model.device)
    out: list[int] = []
    for _ in range(max_new_tokens):
        # Same fp32 output projection as the cached path, so the two differ only in the cache.
        hidden = model.model(ids, use_cache=False).last_hidden_state[:, -1]
        logits = output_logits(model, hidden)[0]
        token = next_token(logits, temperature, generator)
        out.append(token)
        if len(out) == 1 and on_event:
            on_event("first_token")
        if token == eos_id:
            break
        ids = torch.cat([ids, torch.tensor([[token]], device=ids.device)], dim=1)
    return out
