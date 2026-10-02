import json
from pathlib import Path

import pytest
import torch
import yaml

from baselines import cached_autoregressive
from bench.quality import RecordingDraft, draft_bias, first_divergence
from conftest import early_exit_draft, new_cache, random_prompts
from lm.forward import forward
from spec.draft import ModelDraft, UniformDraft
from spec.loop import generate
from spec.ngram import NGramDraft

MAX_NEW = 24


def reference(target: torch.nn.Module, prompts: list[list[int]]) -> list[list[int]]:
    return [cached_autoregressive.generate(target, new_cache(target), p, MAX_NEW) for p in prompts]


@pytest.fixture(scope="module")
def prompts(tiny: torch.nn.Module) -> list[list[int]]:
    return random_prompts(200, tiny.config.vocab_size, seed=10)


@pytest.fixture(scope="module")
def expected(tiny: torch.nn.Module, prompts: list[list[int]]) -> list[list[int]]:
    return reference(tiny, prompts)


def test_200_prompts_identical_one_at_a_time(tiny, tiny_draft, prompts, expected) -> None:
    accepted = drafted = 0
    for prompt, want in zip(prompts, expected):
        out, stats = generate(tiny, new_cache(tiny), ModelDraft(tiny_draft, new_cache(tiny_draft)), [prompt], MAX_NEW, k=4)
        assert out[0] == want
        accepted += stats.accepted
        drafted += stats.drafted
    # Guard against a vacuous pass: identity is only informative if drafts are both
    # accepted and rejected along the way.
    assert 0.1 < accepted / drafted < 0.95


@pytest.mark.parametrize("batch", [8, 25])
def test_200_prompts_identical_batched(tiny, tiny_draft, prompts, expected, batch: int) -> None:
    for i in range(0, len(prompts), batch):
        out, _ = generate(tiny, new_cache(tiny), ModelDraft(tiny_draft, new_cache(tiny_draft)), prompts[i:i + batch], MAX_NEW, k=4)
        assert out == expected[i:i + batch]


@pytest.mark.parametrize("k", [1, 2, 3, 8])
def test_identical_across_draft_lengths(tiny, tiny_draft, prompts, expected, k: int) -> None:
    out, _ = generate(tiny, new_cache(tiny), ModelDraft(tiny_draft, new_cache(tiny_draft)), prompts[:40], MAX_NEW, k=k)
    assert out == expected[:40]


def test_copy_of_target_accepts_everything(tiny, prompts, expected) -> None:
    out, stats = generate(tiny, new_cache(tiny), ModelDraft(tiny, new_cache(tiny)), prompts[:40], MAX_NEW, k=4)
    assert out == expected[:40]
    assert stats.accepted == stats.drafted


def test_uniform_draft_is_still_exact(tiny, prompts, expected) -> None:
    out, stats = generate(tiny, new_cache(tiny), UniformDraft(tiny.config.vocab_size), prompts[:40], MAX_NEW, k=4,
                          generator=torch.Generator().manual_seed(0))
    assert out == expected[:40]
    assert stats.accepted / stats.drafted < 0.05


def test_ngram_draft_is_still_exact(tiny, prompts, expected) -> None:
    out, _ = generate(tiny, new_cache(tiny), NGramDraft(tiny.config.vocab_size), prompts[:40], MAX_NEW, k=4)
    assert out == expected[:40]


def test_eos_stops_mid_window(tiny, tiny_draft, prompts, expected) -> None:
    # Pick an eos that the reference emits partway through, so it lands inside a window.
    want = expected[0]
    eos = want[5]
    cut = want[: want.index(eos) + 1]
    out, _ = generate(tiny, new_cache(tiny), ModelDraft(tiny_draft, new_cache(tiny_draft)), [prompts[0]], MAX_NEW, k=4, eos_id=eos)
    assert out[0] == cut



def lenient_verify(delta: float):
    """Known-bad: the target accepts a draft whose logit is within delta of the maximum.
    Fluent output, a few flipped near-ties, and every flip lands on the draft's token."""

    def verify(target, cache, seq_ids, contexts, draft, temperature):
        ids = torch.cat([torch.tensor([[c[-1]] for c in contexts]), draft], dim=1)
        logits = forward(target, cache, seq_ids, ids, [ids.shape[1]] * len(seq_ids))
        choice = logits.argmax(-1)
        near = logits[:, :-1].gather(-1, draft[..., None]).squeeze(-1) >= logits[:, :-1].max(-1).values - delta
        choice[:, :-1] = torch.where(near, draft, choice[:, :-1])
        return torch.nn.functional.one_hot(choice, logits.shape[-1]).double()

    return verify


def divergence_rows(target, draft, prompts, expected) -> list[dict]:
    rows = []
    for prompt, want in zip(prompts, expected):
        drafter = RecordingDraft(draft, new_cache(draft))
        out, _ = generate(target, new_cache(target), drafter, [prompt], MAX_NEW, k=4)
        at = first_divergence(out[0], want)
        row = {"identical": at is None}
        if at is not None:
            row.update(chosen_token=out[0][at], draft_token=drafter.proposed_at(len(prompt) + at))
        rows.append(row)
    return rows


def test_draft_bias_check_catches_lenient_verification(tiny, tiny_draft, prompts, expected, monkeypatch) -> None:
    # delta = 0.001 flips only a handful of the 60 prompts, close to the size of a real
    # rounding effect, so the check has to work from few divergences.
    monkeypatch.setattr("spec.loop.verify", lenient_verify(0.001))
    rows = divergence_rows(tiny, tiny_draft, prompts[:60], expected[:60])
    hits, covered, pvalue = draft_bias(rows)
    assert covered >= 5
    assert hits == covered
    assert pvalue < 0.05


def load_real_run(name: str) -> dict:
    path = Path("results/greedy_identity.json")
    record = json.loads(path.read_text()) if path.exists() else {}
    if name not in record:
        pytest.skip(f"results/greedy_identity.json has no '{name}' run; python -m scripts.greedy_identity")
    return record[name]


def configured_prompts(name: str) -> int:
    return sum(yaml.safe_load(open("config.yaml"))["greedy_identity"][name]["prompts_per_category"])


def test_real_weights_fp64_all_identical() -> None:
    run = load_real_run("exact")
    assert run["prompts"] == configured_prompts("exact")
    assert run["identical"] == run["prompts"]


def test_real_weights_low_precision_flips_are_rare() -> None:
    run = load_real_run("main")
    assert run["prompts"] == configured_prompts("main")
    assert (run["prompts"] - run["identical"]) / run["prompts"] < 0.05


def test_real_weights_low_precision_flips_are_rounding_ties() -> None:
    # A flip is only excused if the reference's top two logits sit within two grid steps of
    # each other at the divergent position: the verify pass (k+1 tokens) and the decode pass
    # (1 token) round differently by about one step. A larger gap would be a logic bug.
    for row in load_real_run("main")["rows"]:
        if not row["identical"]:
            assert row["gap"] <= 2 * row["ulp"], row


def test_real_weights_low_precision_flips_do_not_favour_the_draft() -> None:
    # A lenient accept step also flips near-ties, so the gap test alone cannot tell it from
    # rounding. What separates them is which token wins: see bench.quality.draft_bias.
    hits, covered, pvalue = draft_bias(load_real_run("main")["rows"])
    if covered == 0:
        pytest.skip("no draft-covered flips in the main run")
    assert pvalue >= 0.05, f"{hits}/{covered} flips emitted the draft's token"
