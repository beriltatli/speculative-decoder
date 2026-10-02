import torch
from scipy.stats import binomtest

from cache.kv import KVCache
from spec.draft import ModelDraft


class RecordingDraft(ModelDraft):
    """Remembers each round's proposal (batch size 1) by the absolute position of its first token."""

    def __init__(self, model: torch.nn.Module, cache: KVCache) -> None:
        super().__init__(model, cache)
        self.proposals: list[tuple[int, list[int]]] = []

    def propose(self, seq_ids, contexts, k, temperature, generator):
        draft, q = super().propose(seq_ids, contexts, k, temperature, generator)
        self.proposals.append((len(contexts[0]), draft[0].tolist()))
        return draft, q

    def proposed_at(self, position: int) -> int | None:
        # The round that decided a position is the last one starting at or before it. An
        # earlier round can also cover it with a draft that was rejected upstream and never
        # judged there, so the first match would be the wrong proposal. None for positions
        # outside the deciding round's window: the prefill token and bonus tokens.
        for start, tokens in reversed(self.proposals):
            if start <= position:
                return tokens[position - start] if position < start + len(tokens) else None
        return None


def first_divergence(out: list[int], want: list[int]) -> int | None:
    if out == want:
        return None
    return next((i for i, (a, b) in enumerate(zip(out, want)) if a != b), min(len(out), len(want)))


def draft_bias(rows: list[dict]) -> tuple[int, int, float]:
    """At each first divergence the draft proposed one token and the speculative path emitted
    one. Rounding flips a near-tie without regard to which token the draft named, so the
    emitted token is the draft's at most about half the time; less in practice, because the
    draft tends to agree with the reference's side of the tie (22-27% in a simulated-rounding
    control on the tiny model). A verification step that is too lenient (accepting a
    near-miss draft) diverges only by emitting the draft's token, pushing the fraction to 1.

    Returns (divergences where the emitted token is the draft's, divergences the draft
    covered, one-sided binomial p-value against 1/2)."""
    covered = [r for r in rows if not r["identical"] and r.get("draft_token") is not None]
    hits = sum(r["chosen_token"] == r["draft_token"] for r in covered)
    if not covered:
        return 0, 0, 1.0
    return hits, len(covered), binomtest(hits, len(covered), 0.5, alternative="greater").pvalue
