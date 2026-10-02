import pytest
import torch

from baselines import cached_autoregressive
from conftest import new_cache, random_prompts
from spec.draft import ModelDraft
from spec.loop import generate

MAX_NEW = 30
K = 4


class ScriptedDraft:
    """Proposes the target's own greedy continuation for the first j positions and a wrong
    token after that, so every round accepts exactly j drafts."""

    def __init__(self, reference: list[int], prompt_len: int, j: int, vocab: int) -> None:
        self.reference, self.prompt_len, self.j, self.vocab = reference, prompt_len, j, vocab

    def prefill(self, seq_ids, prompts) -> None:
        pass

    def propose(self, seq_ids, contexts, k, temperature, generator):
        done = len(contexts[0]) - self.prompt_len
        right = self.reference[done:done + k] + [0] * k
        draft = [right[i] if i < self.j else (right[i] + 1) % self.vocab for i in range(k)]
        tokens = torch.tensor([draft])
        return tokens, torch.nn.functional.one_hot(tokens, self.vocab).float()

    def rollback(self, seq_id, keep) -> None:
        pass

    def release(self, seq_id) -> None:
        pass


@pytest.mark.parametrize("j", range(K + 1))
def test_exactly_j_accepted_every_round(tiny, j: int) -> None:
    prompt = random_prompts(1, tiny.config.vocab_size, seed=3)[0]
    # Two extra reference tokens so the script can look past the budget at the final round.
    want = cached_autoregressive.generate(tiny, new_cache(tiny), prompt, MAX_NEW + K + 1)
    draft = ScriptedDraft(want, len(prompt), j, tiny.config.vocab_size)
    # verify() raises if the target cache is not exactly len(context) - 1 at a round start,
    # so a rollback off by one in either direction fails here before the comparison.
    out, stats = generate(tiny, new_cache(tiny), draft, [prompt], MAX_NEW, k=K)
    assert out[0] == want[:MAX_NEW]
    assert stats.accepts[:j] == stats.attempts[:j]
    assert all(a == 0 for a in stats.accepts[j:])
    assert all(e == j + 1 for e in stats.emitted_per_round[:-1])


class CheckedDraft(ModelDraft):
    """A wrong draft cache never changes greedy output, because the target corrects every
    token; it only lowers acceptance. So the draft's q is compared with an uncached forward
    of the draft model over exactly the tokens it should be conditioning on."""

    def propose(self, seq_ids, contexts, k, temperature, generator):
        draft, q = super().propose(seq_ids, contexts, k, temperature, generator)
        for row, ctx in enumerate(contexts):
            full = torch.tensor([ctx + draft[row, :-1].tolist()])
            logits = self.model(full, use_cache=False).logits[0, -k:]
            assert torch.equal(q[row], torch.nn.functional.one_hot(logits.argmax(-1), q.shape[-1]).float())
        return draft, q


def test_draft_cache_conditions_on_the_right_tokens(tiny, tiny_draft) -> None:
    prompts = random_prompts(6, tiny.config.vocab_size, seed=4)
    out, stats = generate(tiny, new_cache(tiny), CheckedDraft(tiny_draft, new_cache(tiny_draft)), prompts, MAX_NEW, k=K)
    # Both partial and full acceptance must have occurred for this to cover both lag cases.
    assert 0 < stats.accepted < stats.drafted
    assert any(e == K + 1 for e in stats.emitted_per_round)
