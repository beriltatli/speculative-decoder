import pytest

from sanity_guard import pytest_runtest_logreport, pytest_sessionfinish  # noqa: F401  (hooks)
import torch

from lm.load import tiny_model


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "model: needs downloaded weights; skipped when absent")


@pytest.fixture(scope="session")
def tiny() -> torch.nn.Module:
    # fp64 so equivalence can be asserted at 1e-10. In fp32 the gap between a 1-token and an
    # n-token matmul is ~1e-4 on a real model, which would hide a small cache bug.
    return tiny_model(seed=0, dtype=torch.float64)


def early_exit_draft(target: torch.nn.Module, n_layers: int = 1) -> torch.nn.Module:
    """The target truncated to its first layers: same embeddings and head, so it agrees with
    the target often but not always, which makes acceptance land strictly between 0 and 1."""
    import copy

    draft = copy.deepcopy(target)
    draft.model.layers = draft.model.layers[:n_layers]
    draft.config.num_hidden_layers = n_layers
    return draft


@pytest.fixture(scope="session")
def tiny_draft(tiny: torch.nn.Module) -> torch.nn.Module:
    return early_exit_draft(tiny)


def random_prompts(n: int, vocab: int, seed: int, low: int = 3, high: int = 20) -> list[list[int]]:
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(low, high, (n,), generator=gen).tolist()
    return [torch.randint(0, vocab, (m,), generator=gen).tolist() for m in lengths]


def new_cache(model: torch.nn.Module, num_blocks: int = 4096, block_size: int = 4):
    from cache.kv import KVCache

    return KVCache.for_model(model.config, num_blocks=num_blocks, block_size=block_size, dtype=model.dtype, device="cpu")
