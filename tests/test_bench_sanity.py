"""Sanity checks on the measured run in results/latency.json. They are allowed to fail: a
failure is a finding about the measurement and is reported in the README, not hidden by
loosening a threshold."""
import json
from pathlib import Path

import pytest

CONDITIONS = ["long", "short"]
CATEGORIES = ["code", "prose", "repetitive"]
METHODS = ["no_cache", "cached", "spec_ngram", "spec_draft", "spec_self", "hf_assisted"]


def get(record: dict, *path: str):
    """record[path[0]][path[1]]...; a missing key skips with the key path named, and
    sanity_guard turns that skip into a failure whenever the results file exists."""
    for depth, key in enumerate(path):
        if not isinstance(record, dict) or key not in record:
            pytest.skip(f"missing key {path[: depth + 1]} in results/latency.json")
        record = record[key]
    return record


@pytest.fixture(scope="module")
def latency() -> dict:
    path = Path("results/latency.json")
    if not path.exists():
        pytest.skip("results/latency.json not generated yet; python -m scripts.benchmark")
    return json.loads(path.read_text())


@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.parametrize("condition", CONDITIONS)
def test_kv_cache_is_the_first_order_win(latency, condition: str, category: str) -> None:
    no_cache = get(latency, "conditions", condition, "categories", category, "no_cache", "tpot_ms", "p50")
    cached = get(latency, "conditions", condition, "categories", category, "cached", "tpot_ms", "p50")
    assert no_cache / cached >= 5, (
        f"{condition}/{category}: no-cache p50 TPOT {no_cache:.1f} ms is only {no_cache / cached:.2f}x cached {cached:.1f} ms; "
        "the cache is not where the speed comes from at these context lengths"
    )


@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.parametrize("condition", CONDITIONS)
def test_draft_equal_to_target_is_slower_than_baseline(latency, condition: str, category: str) -> None:
    cached = get(latency, "conditions", condition, "categories", category, "cached", "e2e_ms", "p50")
    self_draft = get(latency, "conditions", condition, "categories", category, "spec_self", "e2e_ms", "p50")
    speedup = cached / self_draft
    assert speedup < 1.0, (
        f"{condition}/{category}: drafting with the target itself measured {speedup:.3f}x end-to-end. It runs the same model "
        "twice and must be slower; the harness is measuring something other than the work being done"
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_acceptance_separates_categories(latency, condition: str) -> None:
    alpha = {c: round(get(latency, "conditions", condition, "categories", c, "spec_draft", "alpha"), 3) for c in CATEGORIES}
    assert alpha["repetitive"] > alpha["code"] > alpha["prose"], f"{condition} acceptance by category: {alpha}"
    assert alpha["repetitive"] - alpha["prose"] >= 0.15, f"{condition} acceptance by category: {alpha}"


@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("condition", CONDITIONS)
def test_enough_samples_for_a_p99(latency, condition: str, category: str, method: str) -> None:
    n = get(latency, "conditions", condition, "categories", category, method, "n")
    assert n >= 100, f"{condition}/{category}/{method}: {n} measured iterations; a p99 needs at least 100"


@pytest.mark.parametrize("metric", ["e2e_ms", "ttft_ms", "tpot_ms"])
@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("condition", CONDITIONS)
def test_tail_is_above_median(latency, condition: str, category: str, method: str, metric: str) -> None:
    # p99 within 20% of p50 means every iteration looked the same: warmup leaking into a
    # steady state, or too few samples for a tail to exist.
    cell = get(latency, "conditions", condition, "categories", category, method, metric)
    assert cell["p99"] >= 1.2 * cell["p50"], f"{condition}/{category}/{method} {metric}: p50 {cell['p50']:.1f}, p99 {cell['p99']:.1f}"
