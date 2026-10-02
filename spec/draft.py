from typing import Protocol

import torch
from transformers import PreTrainedModel

from cache.kv import KVCache
from lm.forward import forward
from lm.sampling import probabilities
from spec.accept import sample_inverse_cdf


class Drafter(Protocol):
    def prefill(self, seq_ids: list[int], prompts: list[list[int]]) -> None: ...

    def propose(
        self, seq_ids: list[int], contexts: list[list[int]], k: int, temperature: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns draft [B, k] and q [B, k, V] on CPU: the distributions the draft tokens
        were actually sampled from. Accept-reject is exact only against those."""
        ...

    def rollback(self, seq_id: int, keep: int) -> None: ...

    def release(self, seq_id: int) -> None: ...


def sample(probs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    u = torch.rand(probs.shape[0], generator=generator, dtype=probs.dtype)
    return sample_inverse_cdf(probs, u)


class ModelDraft:
    def __init__(self, model: PreTrainedModel, cache: KVCache) -> None:
        self.model = model
        self.cache = cache

    def prefill(self, seq_ids: list[int], prompts: list[list[int]]) -> None:
        for seq_id in seq_ids:
            self.cache.add(seq_id)
        run(self.model, self.cache, seq_ids, prompts)

    def propose(
        self, seq_ids: list[int], contexts: list[list[int]], k: int, temperature: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The draft cache lags the context by one token normally and by two after a round in
        # which every draft was accepted (d_k was sampled but never fed). Feeding every
        # uncached token in the first step closes either gap.
        pending = [ctx[self.cache.length(s):] for s, ctx in zip(seq_ids, contexts)]
        tokens, qs = [], []
        for _ in range(k):
            q = probabilities(run(self.model, self.cache, seq_ids, pending), temperature).cpu()
            token = sample(q, generator)
            tokens.append(token)
            qs.append(q)
            pending = [[t] for t in token.tolist()]
        return torch.stack(tokens, dim=1), torch.stack(qs, dim=1)

    def rollback(self, seq_id: int, keep: int) -> None:
        self.cache.truncate(seq_id, min(self.cache.length(seq_id), keep))

    def release(self, seq_id: int) -> None:
        self.cache.free(seq_id)


class UniformDraft:
    """The worst useful draft: acceptance near 1/V, output must still be exact."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def prefill(self, seq_ids: list[int], prompts: list[list[int]]) -> None:
        pass

    def propose(
        self, seq_ids: list[int], contexts: list[list[int]], k: int, temperature: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = torch.full((len(seq_ids), k, self.vocab_size), 1.0 / self.vocab_size)
        return torch.randint(0, self.vocab_size, (len(seq_ids), k), generator=generator), q

    def rollback(self, seq_id: int, keep: int) -> None:
        pass

    def release(self, seq_id: int) -> None:
        pass


def run(model: PreTrainedModel, cache: KVCache, seq_ids: list[int], chunks: list[list[int]]) -> torch.Tensor:
    """Forward right-padded chunks; returns the logits after each row's last real token, [B, V]."""
    n_new = [len(c) for c in chunks]
    width = max(n_new)
    input_ids = torch.tensor([c + [0] * (width - len(c)) for c in chunks], device=cache.device)
    logits = forward(model, cache, seq_ids, input_ids, n_new)
    last = torch.tensor(n_new, device=cache.device) - 1
    return logits[torch.arange(len(chunks), device=cache.device), last]
