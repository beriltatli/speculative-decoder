import statistics
import subprocess
import time

import numpy as np
import pytest
import torch

from bench.harness import Stopwatch, environment, measure, synchronizer
from bench.latency import percentile, summarize


class FakeClock:
    """Time moves only when a workload says so, so every duration is known exactly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.log: list[str] = []

    def __call__(self) -> float:
        self.log.append("clock")
        return self.now

    def sync(self) -> None:
        self.log.append("sync")


def fixed(clock: FakeClock, ttft: float, tpot: float, tokens: int):
    def workload(watch: Stopwatch) -> int:
        clock.now += ttft
        watch.first_token()
        clock.now += tpot * (tokens - 1)
        return tokens

    return workload


def test_percentiles_on_1_to_100() -> None:
    xs = list(range(1, 101))
    # Rank h = 99 q / 100: p50 at 49.5 -> 50.5, p95 at 94.05 -> 95.05, p99 at 98.01 -> 99.01.
    assert percentile(xs, 50) == pytest.approx(50.5)
    assert percentile(xs, 95) == pytest.approx(95.05)
    assert percentile(xs, 99) == pytest.approx(99.01)
    assert percentile([7.0], 99) == 7.0
    assert percentile([1.0, 3.0], 50) == 2.0


def test_percentiles_agree_with_numpy() -> None:
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 37, 1000):
        xs = rng.lognormal(size=n).tolist()
        for q in (0, 50, 95, 99, 100):
            assert percentile(xs, q) == pytest.approx(float(np.percentile(xs, q)))


def test_ttft_and_tpot_are_exact_on_a_known_workload() -> None:
    clock = FakeClock()
    run = measure({"a": fixed(clock, ttft=0.100, tpot=0.020, tokens=10)}, warmup=0, repeats=3, sync=clock.sync, seed=0, clock=clock)
    for s in run.samples:
        assert s.ttft == pytest.approx(0.100)
        assert s.total == pytest.approx(0.100 + 9 * 0.020)
    summary = summarize(run)["a"]
    assert summary["tpot_ms"]["p50"] == pytest.approx(20.0)
    assert summary["e2e_ms"]["p50"] == pytest.approx(280.0)


def test_percentiles_through_the_harness() -> None:
    # Repeat r takes (r + 1) ms to first token, so measured TTFT is exactly 1..100 ms.
    clock = FakeClock()
    calls = iter(range(1, 101))

    def workload(watch: Stopwatch) -> int:
        clock.now += next(calls) / 1e3
        watch.first_token()
        return 1

    summary = summarize(measure({"a": workload}, warmup=0, repeats=100, sync=clock.sync, seed=0, clock=clock))["a"]
    assert summary["ttft_ms"]["p50"] == pytest.approx(50.5)
    assert summary["ttft_ms"]["p95"] == pytest.approx(95.05)
    assert summary["ttft_ms"]["p99"] == pytest.approx(99.01)
    assert summary["n"] == 100


def test_warmup_is_excluded_per_configuration() -> None:
    # The first two calls of each configuration are 1000x slower. None may be measured, and
    # each configuration must get its own warmup rather than sharing one.
    clock = FakeClock()
    counts = {"a": 0, "b": 0, "c": 0}

    def workload(name: str):
        def run(watch: Stopwatch) -> int:
            counts[name] += 1
            clock.now += 10.0 if counts[name] <= 2 else 0.01
            watch.first_token()
            return 1

        return run

    run = measure({n: workload(n) for n in counts}, warmup=2, repeats=5, sync=clock.sync, seed=0, clock=clock)
    assert len(run.samples) == 15
    assert all(s.ttft == pytest.approx(0.01) for s in run.samples)
    assert counts == {"a": 7, "b": 7, "c": 7}


def test_every_timestamp_follows_a_sync() -> None:
    clock = FakeClock()
    measure({"a": fixed(clock, 0.1, 0.01, 4), "b": fixed(clock, 0.2, 0.01, 4)}, warmup=1, repeats=3,
            sync=clock.sync, seed=0, clock=clock)
    reads = [i for i, e in enumerate(clock.log) if e == "clock"]
    assert len(reads) == 3 * (2 * 4)  # start, first token, stop for 2 configs x (1 warmup + 3 repeats)
    assert all(clock.log[i - 1] == "sync" for i in reads)


def test_rounds_are_reshuffled_and_reproducible() -> None:
    names = ["a", "b", "c", "d"]

    def orders(seed: int) -> list[list[str]]:
        clock = FakeClock()
        return measure({n: fixed(clock, 0.1, 0.01, 2) for n in names}, warmup=0, repeats=12,
                       sync=clock.sync, seed=seed, clock=clock).orders

    first = orders(0)
    assert all(sorted(o) == names for o in first)
    assert len({tuple(o) for o in first}) > 1
    assert first == orders(0)


def test_interleaving_spreads_drift_and_reports_it() -> None:
    # A machine that slows by 1% per call (throttling). Two identical configurations must
    # come out equal; run back to back, the second would look ~2x slower than the first.
    clock = FakeClock()
    calls = [0]

    def workload(watch: Stopwatch) -> int:
        calls[0] += 1
        clock.now += 0.01 * (1 + 0.01 * calls[0])
        watch.first_token()
        clock.now += 0.01 * (1 + 0.01 * calls[0])
        return 2

    run = measure({"a": workload, "b": workload}, warmup=0, repeats=100, sync=clock.sync, seed=0, clock=clock)
    summary = summarize(run, edge_rounds=10)
    a, b = summary["a"]["tpot_ms"]["p50"], summary["b"]["tpot_ms"]["p50"]
    assert abs(a - b) / a < 0.05
    for name in ("a", "b"):
        assert summary[name]["tpot_ms_last_rounds_p50"] / summary[name]["tpot_ms_first_rounds_p50"] > 1.5

    sequential_a = statistics.median(10 * (1 + 0.01 * i) for i in range(1, 101))
    sequential_b = statistics.median(10 * (1 + 0.01 * i) for i in range(101, 201))
    assert sequential_b / sequential_a > 1.4


def test_real_sleep_workload() -> None:
    # Real clock, real sleeps: the harness must report at least the slept time, and not
    # wildly more. Upper bounds are loose because CI runners are noisy.
    def workload(watch: Stopwatch) -> int:
        time.sleep(0.02)
        watch.first_token()
        for _ in range(4):
            time.sleep(0.01)
        return 5

    def slow_first(watch: Stopwatch) -> int:
        time.sleep(0.2 if not hasattr(slow_first, "seen") else 0.02)
        slow_first.seen = True
        watch.first_token()
        return 1

    run = measure({"sleep": workload, "warm": slow_first}, warmup=1, repeats=10, sync=lambda: None, seed=0)
    summary = summarize(run)
    assert 20 <= summary["sleep"]["ttft_ms"]["p50"] < 40
    assert 10 <= summary["sleep"]["tpot_ms"]["p50"] < 20
    assert summary["warm"]["ttft_ms"]["p99"] < 100


def test_workload_must_report_first_token() -> None:
    with pytest.raises(RuntimeError, match="first token"):
        measure({"a": lambda watch: 3}, warmup=0, repeats=1, sync=lambda: None, seed=0)


def test_synchronizer_matches_device() -> None:
    assert synchronizer(torch.device("cpu"))() is None
    if torch.backends.mps.is_available():
        assert synchronizer(torch.device("mps")) is torch.mps.synchronize
    if torch.cuda.is_available():
        assert synchronizer(torch.device("cuda")) is torch.cuda.synchronize


def test_environment_records_the_checkout() -> None:
    record = environment(torch.device("cpu"), seed=0)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert record["git_commit"] == head
    assert isinstance(record["git_dirty"], bool)
    assert {"torch", "transformers"} <= set(record["packages"])


def test_two_ttft_definitions_on_a_known_speculative_timeline() -> None:
    # Target prefill 100 ms -> first token; draft prefill 30 ms; first round (k = 4 drafts
    # at 5 ms, verify 20 ms) ends at 170 ms with 3 more tokens; 6 later tokens at 10 ms each.
    clock = FakeClock()

    def workload(watch: Stopwatch) -> int:
        clock.now += 0.100
        watch.mark("first_token")
        clock.now += 0.030
        watch.mark("draft_prefilled")
        clock.now += 4 * 0.005 + 0.020
        watch.mark("first_round")
        clock.now += 6 * 0.010
        return 1 + 3 + 6

    summary = summarize(measure({"spec": workload}, warmup=0, repeats=2, sync=clock.sync, seed=0, clock=clock))["spec"]
    assert summary["ttft_ms"]["p50"] == pytest.approx(100.0)
    assert summary["ttft_first_round_ms"]["p50"] == pytest.approx(170.0)
    assert summary["draft_prefill_ms"]["p50"] == pytest.approx(30.0)
    assert summary["e2e_ms"]["p50"] == pytest.approx(230.0)
    assert summary["tpot_ms"]["p50"] == pytest.approx(130.0 / 9)
    # The first-round TPOT drops 70 ms of work but keeps its 3 tokens in the denominator.
    assert summary["tpot_first_round_ms"]["p50"] == pytest.approx(60.0 / 9)


def test_non_speculative_definitions_coincide() -> None:
    clock = FakeClock()
    summary = summarize(measure({"a": fixed(clock, 0.05, 0.01, 5)}, warmup=0, repeats=2, sync=clock.sync, seed=0, clock=clock))["a"]
    assert summary["ttft_first_round_ms"] == summary["ttft_ms"]
    assert summary["tpot_first_round_ms"] == summary["tpot_ms"]
    assert summary["draft_prefill_ms"] is None


def test_loop_emits_first_token_before_the_draft_runs(tiny, tiny_draft) -> None:
    from conftest import new_cache
    from spec.draft import ModelDraft
    from spec.loop import generate

    events: list[str] = []

    class Watched(ModelDraft):
        def prefill(self, seq_ids, prompts):
            events.append("draft_work")
            super().prefill(seq_ids, prompts)

    generate(tiny, new_cache(tiny), Watched(tiny_draft, new_cache(tiny_draft)), [[1, 2, 3]], 12, k=4,
             on_event=events.append)
    assert events == ["first_token", "draft_work", "draft_prefilled", "first_round"]


def test_after_each_runs_outside_the_timed_window() -> None:
    # A hook that takes 5 s (freeing caches, logging) must not appear in any measurement.
    clock = FakeClock()
    calls: list[tuple[str, int]] = []

    def after(name: str, round_: int) -> None:
        clock.now += 5.0
        calls.append((name, round_))

    run = measure({"a": fixed(clock, 0.1, 0.01, 3)}, warmup=2, repeats=3, sync=clock.sync, seed=0, clock=clock,
                  after_each=after)
    assert all(s.total == pytest.approx(0.12) for s in run.samples)
    assert calls == [("a", -1), ("a", -1), ("a", 0), ("a", 1), ("a", 2)]
