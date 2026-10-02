from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel

from cache.kv import KVCache
from spec.accept import accept_reject
from spec.draft import Drafter, run, sample
from spec.verify import verify
from lm.sampling import probabilities


@dataclass
class SpecStats:
    k: int
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    # attempts[i]: rounds that reached draft position i; accepts[i]: of those, accepted there.
    attempts: list[int] = field(default_factory=list)
    accepts: list[int] = field(default_factory=list)
    # One entry per (sequence, round): tokens emitted that round, after eos/budget trimming.
    emitted_per_round: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.attempts = [0] * self.k
        self.accepts = [0] * self.k

    @property
    def discarded(self) -> int:
        return self.drafted - self.accepted


@torch.no_grad()
def generate(
    target: PreTrainedModel,
    cache: KVCache,
    drafter: Drafter,
    prompts: list[list[int]],
    max_new_tokens: int,
    k: int,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    eos_id: int | None = None,
) -> tuple[list[list[int]], SpecStats]:
    generator = generator or torch.Generator().manual_seed(0)
    stats = SpecStats(k)
    seq_ids = list(range(len(prompts)))
    for seq_id in seq_ids:
        cache.add(seq_id)
    # The first token comes from the target's prefill logits, exactly as in plain decoding,
    # so time to first token is the same quantity for every method.
    first = sample(probabilities(run(target, cache, seq_ids, prompts), temperature).cpu(), generator).tolist()
    drafter.prefill(seq_ids, prompts)
    contexts = [p + [t] for p, t in zip(prompts, first)]
    outputs = [[t] for t in first]
    active = [s for s in seq_ids if not finished(outputs[s], max_new_tokens, eos_id)]
    for s in seq_ids:
        if s not in active:
            release(cache, drafter, s)

    while active:
        ctx = [contexts[s] for s in active]
        draft, q = drafter.propose(active, ctx, k, temperature, generator)
        p = verify(target, cache, active, ctx, draft, temperature)
        verdict = accept_reject(p, q, draft, generator=generator)
        stats.rounds += 1
        still_active = []
        for row, s in enumerate(active):
            n = int(verdict.n_accepted[row])
            record(stats, n)
            new = draft[row, :n].tolist() + [int(verdict.next_token[row])]
            new = trim(new, len(outputs[s]), max_new_tokens, eos_id)
            stats.emitted_per_round.append(len(new))
            length_before = len(contexts[s])
            contexts[s] = contexts[s] + new
            outputs[s] = outputs[s] + new
            if finished(outputs[s], max_new_tokens, eos_id):
                release(cache, drafter, s)
                continue
            # The target cache holds context[:-1] (length L - 1) plus the k + 1 verified
            # positions: L + k entries. After keeping n drafts and the corrected token the
            # context has L + n + 1 tokens, and the invariant wants all but the last cached:
            # L + n. Entries for d_{n+1}..d_k (and, at n = k, nothing) are dropped. One more
            # and the next round conditions on a rejected draft; one fewer and the target's
            # positions shift by one for the rest of the sequence.
            cache.truncate(s, length_before + n)
            drafter.rollback(s, length_before + n)
            still_active.append(s)
        active = still_active
    return outputs, stats


def record(stats: SpecStats, n: int) -> None:
    stats.drafted += stats.k
    stats.accepted += n
    for i in range(min(n + 1, stats.k)):
        stats.attempts[i] += 1
    for i in range(n):
        stats.accepts[i] += 1


def trim(new: list[int], have: int, max_new_tokens: int, eos_id: int | None) -> list[int]:
    new = new[: max_new_tokens - have]
    if eos_id is not None and eos_id in new:
        new = new[: new.index(eos_id) + 1]
    return new


def finished(output: list[int], max_new_tokens: int, eos_id: int | None) -> bool:
    return len(output) >= max_new_tokens or (eos_id is not None and output[-1] == eos_id)


def release(cache: KVCache, drafter: Drafter, seq_id: int) -> None:
    cache.free(seq_id)
    drafter.release(seq_id)
