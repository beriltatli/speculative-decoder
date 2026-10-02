import math

import pytest
import torch

from baselines import cached_autoregressive
from conftest import new_cache, random_prompts
from spec.draft import ModelDraft
from spec.loop import generate


@pytest.mark.parametrize(("k", "max_new"), [(1, 20), (4, 21), (4, 23), (7, 30)])
def test_full_acceptance_emits_k_plus_one(tiny, k: int, max_new: int) -> None:
    prompt = random_prompts(1, tiny.config.vocab_size, seed=5)[0]
    want = cached_autoregressive.generate(tiny, new_cache(tiny), prompt, max_new)
    out, stats = generate(tiny, new_cache(tiny), ModelDraft(tiny, new_cache(tiny)), [prompt], max_new, k=k)
    assert out[0] == want
    # The first token comes from prefill; each round then yields k accepted + 1 bonus.
    # Forgetting the bonus gives k per round; taking it twice duplicates a token and breaks
    # the equality above.
    assert stats.rounds == math.ceil((max_new - 1) / (k + 1))
    assert all(e == k + 1 for e in stats.emitted_per_round[:-1])


def test_bonus_comes_from_position_k_plus_one(tiny) -> None:
    """With temperature > 0 and a copy draft, every draft is accepted and the token after the
    window is sampled from the target's k+1-th row. Its distribution is checked in
    test_accept_toy; here, that sampling with a copy draft never rejects."""
    prompts = random_prompts(20, tiny.config.vocab_size, seed=6)
    _, stats = generate(tiny, new_cache(tiny), ModelDraft(tiny, new_cache(tiny)), prompts, 20, k=4,
                        temperature=1.0, generator=torch.Generator().manual_seed(0))
    assert stats.accepted == stats.drafted
