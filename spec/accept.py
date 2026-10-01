"""Speculative sampling acceptance (Leviathan et al. 2023, Alg. 1; Chen et al. 2023, Alg. 2).

For drafted x ~ q, accepting with probability min(1, p(x)/q(x)) emits x with probability
min(p(x), q(x)). Rejection happens with probability beta = sum_x max(0, p(x) - q(x)), and
sampling the residual r = max(0, p - q) / beta on rejection emits x with probability
max(0, p(x) - q(x)). The two add to p(x) for every x, whatever q is. That sum is the whole
guarantee: resampling from p instead of r counts the mass where q > p twice and drifts the
output toward the draft, with fluent text and no error anywhere.
"""
from collections.abc import Callable
from dataclasses import dataclass

import torch

Residual = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """norm(max(0, p - q)) over the last dim.

    If p == q exactly the residual is all zeros, but then p(x)/q(x) = 1 for every x and a
    rejection cannot happen, so the fallback to p is never sampled from. It exists so a
    float near-tie returns a valid distribution instead of NaN."""
    r = torch.clamp(p - q, min=0.0)
    total = r.sum(dim=-1, keepdim=True)
    return torch.where(total > 0, r / total.clamp(min=torch.finfo(r.dtype).tiny), p)


def sample_inverse_cdf(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """probs [B, V] (need not sum to exactly 1), u [B] in [0, 1). Token i is chosen iff
    u * total lies in [cdf[i-1], cdf[i]), so zero-probability tokens have an empty interval
    and are never chosen. Explicit uniforms make the coupled test possible."""
    cdf = probs.cumsum(dim=-1)
    target = (u * cdf[:, -1]).unsqueeze(-1)
    return torch.searchsorted(cdf, target, right=True).squeeze(-1).clamp(max=probs.shape[-1] - 1)


@dataclass(frozen=True)
class Verdict:
    n_accepted: torch.Tensor  # [B] leading draft tokens kept, 0..k
    next_token: torch.Tensor  # [B] residual sample at the first rejection, or the bonus token if none


def accept_reject(
    p: torch.Tensor,
    q: torch.Tensor,
    draft: torch.Tensor,
    generator: torch.Generator | None = None,
    u_accept: torch.Tensor | None = None,
    u_residual: torch.Tensor | None = None,
    residual: Residual = residual_distribution,
) -> Verdict:
    """p [B, k+1, V] target probabilities at each drafted position plus one past the window,
    q [B, k, V] draft probabilities the draft tokens were sampled from, draft [B, k].

    At temperature 0 both p and q are one-hot; the ratio is then 1 if the target's argmax is
    the drafted token and 0 otherwise, and the residual is the target's one-hot, so this
    reduces exactly to greedy verification with no separate code path."""
    b, k = draft.shape
    if u_accept is None:
        u_accept = torch.rand(b, k, generator=generator, dtype=p.dtype, device=p.device)
    if u_residual is None:
        u_residual = torch.rand(b, generator=generator, dtype=p.dtype, device=p.device)

    p_x = p[:, :k].gather(-1, draft.unsqueeze(-1)).squeeze(-1)
    q_x = q.gather(-1, draft.unsqueeze(-1)).squeeze(-1)
    if not bool((q_x > 0).all()):
        raise ValueError("a drafted token has zero draft probability; q is not what it was sampled from")
    # u < p/q is u < min(1, p/q) because u < 1. Written as a product to avoid dividing by q.
    accepted = u_accept * q_x < p_x
    n_accepted = accepted.long().cumprod(dim=-1).sum(dim=-1)

    # The bonus position is a rejection against a draft that put zero mass everywhere:
    # max(0, p - 0) = p. Padding q with a zero row makes the bonus token the same residual
    # sample as a rejection, so it can neither be forgotten nor taken twice.
    q_padded = torch.cat([q, torch.zeros_like(q[:, :1])], dim=1)
    rows = torch.arange(b, device=p.device)
    r = residual(p[rows, n_accepted], q_padded[rows, n_accepted])
    return Verdict(n_accepted, sample_inverse_cdf(r, u_residual))
