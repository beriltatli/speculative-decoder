import pytest
import torch

from lm.load import tiny_model


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "model: needs downloaded weights; skipped when absent")


@pytest.fixture(scope="session")
def tiny() -> torch.nn.Module:
    # fp64 so equivalence can be asserted at 1e-10. In fp32 the gap between a 1-token and an
    # n-token matmul is ~1e-4 on a real model, which would hide a small cache bug.
    return tiny_model(seed=0, dtype=torch.float64)
