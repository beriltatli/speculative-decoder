import math

import pytest
import torch
from scipy.stats import chi2_contingency, chisquare

from spec.accept import accept_reject

N = 100_000
V = 32
K = 4
# Significance for "within sampling noise". The streams are seeded, so the outcome is
# deterministic; alpha only sets how surprising a correct run would have to be to fail.
ALPHA = 1e-3


def resample_from_target(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The common bug: on rejection, sample p instead of norm(max(0, p - q))."""
    return p


def argmax_of_residual(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Right residual, wrong sampling: always its most likely token."""
    r = torch.clamp(p - q, min=0.0)
    return torch.nn.functional.one_hot(r.argmax(dim=-1), p.shape[-1]).to(p.dtype)


def toy_pair(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    dirichlet = torch.distributions.Dirichlet(torch.ones(V, dtype=torch.float64))
    return dirichlet.sample(), dirichlet.sample()


def tv(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * (a - b).abs().sum().item()


def tv_noise(p: torch.Tensor, n: int) -> float:
    """E[TV] between two independent n-sample empirical distributions of p, from the normal
    approximation p_hat1 - p_hat2 ~ N(0, 2 p (1 - p) / n) and E|N(0, s^2)| = s sqrt(2/pi)."""
    return 0.5 * sum(math.sqrt(2 / math.pi) * math.sqrt(2 * x * (1 - x) / n) for x in p.tolist())


def emitted_tokens(p: torch.Tensor, q: torch.Tensor, residual, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One speculative round per row. Returns the first emitted token for every row, and the
    second emitted token for rows that accepted at least one draft. Whether the first draft
    is accepted depends only on draft[0] and its uniform, so conditioning on it leaves the
    second token's distribution untouched."""
    gen = torch.Generator().manual_seed(seed)
    draft = torch.multinomial(q, N * K, replacement=True, generator=gen).view(N, K)
    kwargs = {} if residual is None else {"residual": residual}
    verdict = accept_reject(p.expand(N, K + 1, V), q.expand(N, K, V), draft, generator=gen, **kwargs)
    n = verdict.n_accepted
    first = torch.where(n >= 1, draft[:, 0], verdict.next_token)
    second = torch.where(n >= 2, draft[:, 1], verdict.next_token)[n >= 1]
    return first, second


def stats(samples: torch.Tensor, p: torch.Tensor, plain: torch.Tensor) -> dict[str, float]:
    counts = torch.bincount(samples, minlength=V).double()
    plain_counts = torch.bincount(plain, minlength=V).double()
    return {
        "gof_p": chisquare(counts.numpy(), (p * counts.sum()).numpy()).pvalue,
        "homogeneity_p": chi2_contingency(torch.stack([counts, plain_counts]).numpy())[1],
        "tv": tv(counts / counts.sum(), plain_counts / plain_counts.sum()),
        "tv_noise": tv_noise(p, len(samples)),
    }


@pytest.fixture(scope="module")
def setup() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p, q = toy_pair(seed=0)
    plain = torch.multinomial(p, N, replacement=True, generator=torch.Generator().manual_seed(1))
    return p, q, plain


@pytest.mark.parametrize("position", [0, 1])
def test_correct_implementation_is_within_noise(setup, position: int) -> None:
    p, q, plain = setup
    samples = emitted_tokens(p, q, None, seed=2)[position]
    s = stats(samples, p, plain)
    assert s["gof_p"] > ALPHA
    assert s["homogeneity_p"] > ALPHA
    # E[TV] under the null is tv_noise; 2x is a loose band for a single draw of a sum of 32 terms.
    assert s["tv"] < 2 * s["tv_noise"]


def mixed_draft(p: torch.Tensor, q: torch.Tensor, w: float) -> torch.Tensor:
    """A draft that agrees with the target more as w -> 1: overlap sum min(p, q_w) rises
    from 0.447 at w=0 to 0.989 at w=0.98 for the seed-0 pair."""
    return w * p + (1 - w) * q


@pytest.mark.parametrize("broken", [resample_from_target, argmax_of_residual])
def test_broken_residual_is_caught_on_a_poor_draft(setup, broken) -> None:
    p, q, plain = setup
    s = stats(emitted_tokens(p, q, broken, seed=2)[0], p, plain)
    assert s["gof_p"] < ALPHA
    assert s["homogeneity_p"] < ALPHA
    assert s["tv"] > 2 * s["tv_noise"]


# Goodness of fit against the known p is the decisive statistic: it carries one sample's
# noise, where the two-sample homogeneity test and TV carry two. At w=0.9 (overlap 0.945)
# TV no longer clears 2x noise but GoF rejects at p=9e-19.
@pytest.mark.parametrize("w", [0.0, 0.8, 0.9, 0.95])
def test_gof_catches_resampling_from_target_as_draft_improves(setup, w: float) -> None:
    p, q, plain = setup
    s = stats(emitted_tokens(p, mixed_draft(p, q, w), resample_from_target, seed=2)[0], p, plain)
    assert s["gof_p"] < ALPHA


def test_known_blind_spot_at_100k_samples(setup) -> None:
    """At overlap 0.989 the bug shifts the output by TV 0.0034, under the 0.0085 sampling noise
    of 100k draws, and no test here sees it; detecting it needs on the order of (0.0085 /
    0.0034)^2 = 6x more samples. This is why test_accept_toy asserts exact equality on a
    coupled stream: that test has no power limit. Asserted so a change in power is noticed."""
    p, q, plain = setup
    s = stats(emitted_tokens(p, mixed_draft(p, q, 0.98), resample_from_target, seed=2)[0], p, plain)
    assert s["gof_p"] > ALPHA


def test_broken_bias_matches_its_closed_form(setup) -> None:
    """Resampling p on rejection emits min(p, q) + beta * p, so its distance from p is
    0.5 * sum |beta * p - max(0, p - q)|. The test above should be detecting exactly that."""
    p, q, plain = setup
    beta = torch.clamp(p - q, min=0).sum()
    predicted = (torch.minimum(p, q) + beta * p)
    samples = emitted_tokens(p, q, resample_from_target, seed=2)[0]
    empirical = torch.bincount(samples, minlength=V).double() / N
    assert tv(empirical, predicted) < 2 * tv_noise(predicted, N)
