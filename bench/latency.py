"""Latency definitions. Every timestamp is wall-clock seconds from the request's start, read
after a device sync (bench.harness.Stopwatch).

TTFT  request start to the first output token existing on the host: everything that runs
      before that token counts. Which work that is depends on the method:
        cached, no_cache       target prefill.
        spec_draft, spec_ngram target prefill only. The loop emits the first token from the
                               target's prefill logits before the draft does anything.
        hf_assisted            target prefill, draft prefill, k draft steps and the first
                               verify: transformers emits nothing until its first round.

TTFT (first round)  request start to the end of the first speculative round: target
      prefill, draft prefill, k draft steps and the first verify pass. Equal to TTFT for the
      non-speculative methods. Reported beside TTFT, not instead of it, because the choice
      of stopwatch position moves the result.

TPOT  (total - TTFT) / (output tokens - 1), for each TTFT definition. Under the first-round
      definition the numerator loses the first round's time while the denominator keeps its
      up to k + 1 tokens, so that TPOT is biased low by construction.

draft prefill  the draft's own prefill, between "first_token" and "draft_prefilled". A
      one-time cost per request; under the first-token TTFT it is spread over every token
      of TPOT, which is why it is also reported on its own.

End-to-end  total request time. Nothing can be moved out of it.

Headline tables use the first-token TTFT, the moment a user can see output, with draft
prefill and end-to-end beside it so the draft's one-time cost stays visible.
"""
import statistics
from collections.abc import Sequence

from bench.harness import Run, Sample


def percentile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default, 'linear'): rank
    h = (n - 1) q / 100 between the sorted values at floor(h) and ceil(h)."""
    if not values:
        raise ValueError("percentile of no values")
    xs = sorted(values)
    h = (len(xs) - 1) * q / 100
    lo = int(h)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def spread(values: Sequence[float]) -> dict[str, float]:
    return {f"p{q}": percentile(values, q) for q in (50, 95, 99)}


def summarize(run: Run, edge_rounds: int = 1) -> dict[str, dict]:
    """Per configuration, in milliseconds, each as p50/p95/p99 and never a bare mean, plus
    the TPOT median over the first and the last `edge_rounds` rounds. A last/first ratio
    well above 1 is the machine slowing down, not the configuration."""
    out: dict[str, dict] = {}
    for name in dict.fromkeys(s.config for s in run.samples):
        mine = [s for s in run.samples if s.config == name]

        def ttft_round(s: Sample) -> float:
            return s.marks.get("first_round", s.ttft)

        def tpot(s: Sample, ttft: float) -> float | None:
            return (s.total - ttft) / (s.tokens - 1) * 1e3 if s.tokens > 1 else None

        def ms(values) -> dict[str, float] | None:
            values = [v for v in values if v is not None]
            return spread(values) if values else None

        def edge(rounds) -> float | None:
            values = [tpot(s, s.ttft) for s in mine if s.round in rounds and s.tokens > 1]
            return statistics.median(values) if values else None

        prefill = [(s.marks["draft_prefilled"] - s.ttft) * 1e3 for s in mine if "draft_prefilled" in s.marks]
        out[name] = {
            "n": len(mine),
            "tokens": ms(s.tokens for s in mine),
            "ttft_ms": ms(s.ttft * 1e3 for s in mine),
            "ttft_first_round_ms": ms(ttft_round(s) * 1e3 for s in mine),
            "tpot_ms": ms(tpot(s, s.ttft) for s in mine),
            "tpot_first_round_ms": ms(tpot(s, ttft_round(s)) for s in mine),
            "draft_prefill_ms": ms(prefill),
            "e2e_ms": ms(s.total * 1e3 for s in mine),
            "tpot_ms_first_rounds_p50": edge(range(edge_rounds)),
            "tpot_ms_last_rounds_p50": edge(range(run.repeats - edge_rounds, run.repeats)),
        }
    return out
