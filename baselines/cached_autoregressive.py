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
        logits = forward(model, cache, [seq_id], torch.tensor([prompt], device=device), [len(prompt)],
                         last_only=True)[0]
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


@torch.no_grad()
def generate_batch(
    model: PreTrainedModel,
    cache: KVCache,
    prompts: list[list[int]],
    max_new_tokens: int,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    eos_id: int | None = None,
    on_event: Callable[[str], None] | None = None,
) -> list[list[int]]:
    """Plain cached decoding of several prompts at once: one forward per step over every
    sequence still running. A sequence leaves the batch when it emits eos or reaches
    max_new_tokens, as in the speculative loop, so the two are compared on the same work."""
    device = cache.device
    seq_ids = list(range(len(prompts)))
    for seq_id in seq_ids:
        cache.add(seq_id)
    try:
        width = max(len(p) for p in prompts)
        ids = torch.tensor([p + [0] * (width - len(p)) for p in prompts], device=device)
        logits = forward(model, cache, seq_ids, ids, [len(p) for p in prompts], last_only=True)
        outputs: list[list[int]] = [[] for _ in prompts]
        active = seq_ids
        while True:
            if temperature == 0.0:
                # One argmax and one device sync for the whole batch, not one per row.
                tokens = logits.argmax(dim=-1).tolist()
            else:
                tokens = [next_token(row, temperature, generator) for row in logits]
            for s, t in zip(active, tokens):
                outputs[s].append(t)
            if on_event and active is seq_ids:
                on_event("first_token")
            still = []
            for s, t in zip(active, tokens):
                if t == eos_id or len(outputs[s]) == max_new_tokens:
                    cache.free(s)
                else:
                    still.append(s)
            active = still
            if not active:
                return outputs
            step = torch.tensor([[outputs[s][-1]] for s in active], device=device)
            logits = forward(model, cache, active, step, [1] * len(active))[:, 0]
    finally:
        for seq_id in seq_ids:
            if seq_id in cache.tables:
                cache.free(seq_id)
