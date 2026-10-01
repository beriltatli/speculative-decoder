from fractions import Fraction

import torch

from spec.accept import accept_reject, residual_distribution, sample_inverse_cdf

P = [Fraction(4, 10), Fraction(3, 10), Fraction(2, 10), Fraction(1, 10), Fraction(0)]
Q = [Fraction(1, 10), Fraction(2, 10), Fraction(3, 10), Fraction(1, 10), Fraction(3, 10)]
# By hand: max(0, p - q) = [.3, .1, 0, 0, 0], beta = .4, residual = [.75, .25, 0, 0, 0].
RESIDUAL = [Fraction(3, 4), Fraction(1, 4), Fraction(0), Fraction(0), Fraction(0)]
# min(1, p/q) per drafted token.
ACCEPT = [Fraction(1), Fraction(1), Fraction(2, 3), Fraction(1), Fraction(0)]


def tensor(xs: list[Fraction]) -> torch.Tensor:
    return torch.tensor([float(x) for x in xs], dtype=torch.float64)


def test_residual_matches_hand_computation() -> None:
    r = residual_distribution(tensor(P), tensor(Q))
    assert torch.allclose(r, tensor(RESIDUAL), atol=1e-15, rtol=0)


def test_beta_is_one_minus_overlap() -> None:
    beta = sum(max(Fraction(0), p - q) for p, q in zip(P, Q))
    overlap = sum(min(p, q) for p, q in zip(P, Q))
    assert beta == Fraction(2, 5) and overlap == 1 - beta


def test_scripted_uniforms() -> None:
    p = tensor(P).expand(1, 2, 5)
    q = tensor(Q).expand(1, 1, 5)
    draft = torch.tensor([[2]])
    run = lambda ua, ur: accept_reject(p, q, draft, u_accept=torch.tensor([[ua]], dtype=torch.float64),
                                       u_residual=torch.tensor([ur], dtype=torch.float64))
    accepted = run(0.6, 0.5)                    # 0.6 < 2/3
    assert accepted.n_accepted.item() == 1
    assert accepted.next_token.item() == 1      # bonus from p: cdf [.4, .7, ...], 0.5 -> token 1
    rejected = run(0.7, 0.8)                    # 0.7 >= 2/3
    assert rejected.n_accepted.item() == 0
    assert rejected.next_token.item() == 1      # residual cdf [.75, 1], 0.8 -> token 1
    assert run(0.7, 0.5).next_token.item() == 0


def test_zero_target_mass_is_always_rejected_and_never_emitted() -> None:
    p = tensor(P).expand(1000, 2, 5)
    q = tensor(Q).expand(1000, 1, 5)
    draft = torch.full((1000, 1), 4)
    verdict = accept_reject(p, q, draft, generator=torch.Generator().manual_seed(0))
    assert verdict.n_accepted.sum().item() == 0
    assert set(verdict.next_token.tolist()) <= {0, 1}


def test_coupled_stream_reproduces_target_exactly() -> None:
    """Every draft token y is enumerated with weight q(y), and the two uniforms run over the
    midpoints of a 120-point grid. 120 is a multiple of every denominator involved (2/3, 3/4,
    2/5, 7/10, 9/10), so each acceptance and inverse-CDF interval holds a whole number of grid
    points and the output distribution is computed exactly, as fractions."""
    n = 120
    grid = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    u_accept = grid.repeat_interleave(n).unsqueeze(-1)
    u_residual = grid.repeat(n)
    p = tensor(P).expand(n * n, 2, 5)
    q = tensor(Q).expand(n * n, 1, 5)
    emitted = [Fraction(0)] * 5
    for y in range(5):
        if Q[y] == 0:
            continue
        draft = torch.full((n * n, 1), y)
        verdict = accept_reject(p, q, draft, u_accept=u_accept, u_residual=u_residual)
        first = torch.where(verdict.n_accepted == 1, draft[:, 0], verdict.next_token)
        accepted = int(verdict.n_accepted.sum())
        assert Fraction(accepted, n * n) == ACCEPT[y]
        for x, count in enumerate(torch.bincount(first, minlength=5).tolist()):
            emitted[x] += Q[y] * Fraction(count, n * n)
    assert emitted == P


def test_greedy_reduces_to_argmax_verification() -> None:
    v = 5
    target_argmax = torch.tensor([[3, 1, 4]])  # positions 0, 1 and the bonus position
    p = torch.nn.functional.one_hot(target_argmax, v).double()
    draft = torch.tensor([[3, 2]])
    q = torch.nn.functional.one_hot(draft, v).double()
    verdict = accept_reject(p, q, draft, generator=torch.Generator().manual_seed(0))
    assert verdict.n_accepted.item() == 1       # 3 matches, 2 != 1
    assert verdict.next_token.item() == 1       # the target's own choice at the mismatch
    all_match = accept_reject(p, torch.nn.functional.one_hot(torch.tensor([[3, 1]]), v).double(),
                              torch.tensor([[3, 1]]), generator=torch.Generator().manual_seed(0))
    assert all_match.n_accepted.item() == 2
    assert all_match.next_token.item() == 4     # bonus token from the k+1-th position


def test_inverse_cdf_skips_zero_mass() -> None:
    probs = torch.tensor([[0.0, 0.5, 0.0, 0.5, 0.0]], dtype=torch.float64)
    for u, expected in [(0.0, 1), (0.49, 1), (0.5, 3), (0.999, 3)]:
        assert sample_inverse_cdf(probs, torch.tensor([u], dtype=torch.float64)).item() == expected
