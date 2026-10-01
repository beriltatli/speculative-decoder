import pytest
import torch
import yaml

from baselines import cached_autoregressive, no_cache
from cache.kv import KVCache
from lm.forward import forward

# fp64, not the brief's 1e-4 in fp32. On SmolLM2-135M in fp32 the cached path differs from
# recompute by 1.12e-4 on an Apple M3 and by less than 1e-4 on GitHub's x86 runner: 1-row and
# n-row matmuls round differently, and how much depends on the BLAS. In fp64 the same
# comparison is 2.2e-14, so the fp32 gap is kernel rounding and says nothing about the cache.
# The fp32 figure is a platform measurement and is reported with its hardware in results/.
TOL_FP64 = 1e-10


@torch.no_grad()
def recompute(model: torch.nn.Module, tokens: list[int]) -> torch.Tensor:
    """HF's own uncached forward over the whole prefix: [T, V]."""
    return model(torch.tensor([tokens]), use_cache=False).logits[0]


def new_cache(model: torch.nn.Module, block_size: int) -> KVCache:
    return KVCache.for_model(model.config, num_blocks=256, block_size=block_size, dtype=model.dtype, device="cpu")


def max_step_diff(model: torch.nn.Module, tokens: list[int], prefill: int, block_size: int) -> float:
    cache = new_cache(model, block_size)
    cache.add(0)
    logits = forward(model, cache, [0], torch.tensor([tokens[:prefill]]), [prefill])[0]
    worst = (logits - recompute(model, tokens[:prefill])).abs().max().item()
    for t in range(prefill, len(tokens)):
        step = forward(model, cache, [0], torch.tensor([[tokens[t]]]), [1])[0, 0]
        # Literal per-step recompute over tokens[:t+1], not one causal pass over everything.
        worst = max(worst, (step - recompute(model, tokens[: t + 1])[-1]).abs().max().item())
    return worst


@pytest.mark.parametrize("block_size", [1, 3, 16])
def test_incremental_matches_recompute_batch1(tiny: torch.nn.Module, block_size: int) -> None:
    tokens = torch.randint(0, tiny.config.vocab_size, (24,), generator=torch.Generator().manual_seed(1)).tolist()
    assert max_step_diff(tiny, tokens, prefill=5, block_size=block_size) <= TOL_FP64


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_padded_batch_of_mixed_lengths(tiny: torch.nn.Module, block_size: int) -> None:
    gen = torch.Generator().manual_seed(2)
    vocab = tiny.config.vocab_size
    seqs = [torch.randint(0, vocab, (n,), generator=gen).tolist() for n in (3, 9, 6)]
    cache = new_cache(tiny, block_size)
    for seq_id in range(3):
        cache.add(seq_id)
    # Prefill, then rounds of 1-to-4-token appends: the last shape is the k+1 verify pass.
    rounds = [[len(s) for s in seqs], [1, 3, 2], [2, 1, 1], [1, 1, 4]]
    worst = 0.0
    for r, n_new in enumerate(rounds):
        if r > 0:
            for s, n in zip(seqs, n_new):
                s.extend(torch.randint(0, vocab, (n,), generator=gen).tolist())
        chunks = [s[len(s) - n:] for s, n in zip(seqs, n_new)]
        width = max(n_new)
        input_ids = torch.tensor([c + [0] * (width - len(c)) for c in chunks])
        logits = forward(tiny, cache, [0, 1, 2], input_ids, n_new)
        for b, (s, n) in enumerate(zip(seqs, n_new)):
            expected = recompute(tiny, s)[len(s) - n:]
            worst = max(worst, (logits[b, :n] - expected).abs().max().item())
    assert worst <= TOL_FP64


@pytest.mark.parametrize("block_size", [1, 4])
def test_truncate_then_append_matches_recompute(tiny: torch.nn.Module, block_size: int) -> None:
    gen = torch.Generator().manual_seed(3)
    vocab = tiny.config.vocab_size
    prefix = torch.randint(0, vocab, (10,), generator=gen).tolist()
    rejected = torch.randint(0, vocab, (4,), generator=gen).tolist()
    replacement = torch.randint(0, vocab, (3,), generator=gen).tolist()
    cache = new_cache(tiny, block_size)
    cache.add(0)
    forward(tiny, cache, [0], torch.tensor([prefix]), [10])
    forward(tiny, cache, [0], torch.tensor([rejected]), [4])
    cache.truncate(0, 11)  # keep the first appended token, as after accepting 1 of 4
    logits = forward(tiny, cache, [0], torch.tensor([replacement]), [3])[0]
    expected = recompute(tiny, prefix + rejected[:1] + replacement)[-3:]
    assert cache.length(0) == 14
    assert (logits - expected).abs().max().item() <= TOL_FP64


def test_cached_greedy_equals_no_cache_greedy(tiny: torch.nn.Module) -> None:
    prompt = [5, 17, 99, 3, 42]
    plain = no_cache.generate(tiny, prompt, max_new_tokens=40)
    cached = cached_autoregressive.generate(tiny, new_cache(tiny, 4), prompt, max_new_tokens=40)
    assert cached == plain


def real_draft(dtype: str) -> tuple[torch.nn.Module, list[int]]:
    from lm.load import load_model, load_tokenizer

    config = yaml.safe_load(open("config.yaml"))
    spec = config["models"]["draft"]
    try:
        model = load_model(spec, config["models"]["cache_dir"], torch.device("cpu"), dtype=dtype, local_files_only=True)
        tokenizer = load_tokenizer(spec, config["models"]["cache_dir"], local_files_only=True)
    except OSError:
        pytest.skip("draft weights not downloaded")
    tokens = tokenizer("def fibonacci(n):\n    if n < 2:\n        return n\n    return")["input_ids"]
    return model, tokens + no_cache.generate(model, tokens, max_new_tokens=24)


@pytest.mark.model
def test_real_draft_model_fp64() -> None:
    model, tokens = real_draft("float64")
    assert max_step_diff(model, tokens, prefill=8, block_size=16) <= TOL_FP64

