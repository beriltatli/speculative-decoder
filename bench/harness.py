"""Timing harness: device synchronisation, warmup, interleaved repeats, environment record.

Every number in results/ comes through here, so the three ways a laptop benchmark lies are
handled in one place:
  - asynchronous devices: MPS and CUDA return as soon as work is queued, so a timer read
    without a sync measures queueing, not compute;
  - warmup: kernel selection, allocator growth and lazy weight paging land on the first
    calls of each configuration and are excluded per configuration;
  - drift: thermal throttling slows whatever runs late, so configurations are reshuffled
    every round instead of run back to back, and first- and last-round medians are kept
    so the drift itself is visible.
"""
import importlib.metadata
import platform
import random
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

Clock = Callable[[], float]
Sync = Callable[[], None]


def synchronizer(device: torch.device) -> Sync:
    if device.type == "mps":
        return torch.mps.synchronize
    if device.type == "cuda":
        return torch.cuda.synchronize
    return lambda: None


class Stopwatch:
    """One measurement window. Every timestamp is taken after a device sync, so it marks
    finished work. Each sync mid-generation drains the queue once; a speculative method
    marks three events against the baselines' one, so it pays two extra pipeline bubbles
    per request, a small bias against speculation."""

    def __init__(self, sync: Sync, clock: Clock = time.perf_counter) -> None:
        self.sync, self.clock = sync, clock
        self.start_at = self.stop_at = None
        self.marks: dict[str, float] = {}
        self.tokens = 0

    def start(self) -> None:
        self.sync()
        self.start_at = self.clock()

    def mark(self, event: str) -> None:
        """Seconds since start of the first occurrence of `event`; later ones are ignored."""
        if event not in self.marks:
            self.sync()
            self.marks[event] = self.clock() - self.start_at

    def first_token(self) -> None:
        self.mark("first_token")

    def stop(self) -> None:
        self.sync()
        self.stop_at = self.clock()


@dataclass(frozen=True)
class Sample:
    config: str
    round: int
    slot: int  # position of this config within its round's shuffled order
    total: float  # seconds, request start to last token
    tokens: int
    marks: dict[str, float]  # seconds from request start; see bench.latency for the definitions

    @property
    def ttft(self) -> float:
        return self.marks["first_token"]


# A workload receives the stopwatch, marks events as they happen ("first_token" is
# required), and returns how many output tokens it produced. start/stop are the harness's job.
Workload = Callable[[Stopwatch], int]


@dataclass
class Run:
    samples: list[Sample]
    warmup: int
    repeats: int
    orders: list[list[str]] = field(default_factory=list)


def measure(
    workloads: dict[str, Workload],
    warmup: int,
    repeats: int,
    sync: Sync,
    seed: int,
    clock: Clock = time.perf_counter,
) -> Run:
    rng = random.Random(seed)
    names = list(workloads)

    def once(name: str) -> Stopwatch:
        watch = Stopwatch(sync, clock)
        watch.start()
        tokens = workloads[name](watch)
        watch.stop()
        if "first_token" not in watch.marks:
            raise RuntimeError(f"{name}: workload never reported its first token")
        watch.tokens = tokens
        return watch

    # Warmup also interleaved, so no configuration gets a cooler machine for its first
    # measured round just because it warmed up first.
    for _ in range(warmup):
        for name in rng.sample(names, len(names)):
            once(name)

    run = Run([], warmup, repeats)
    for r in range(repeats):
        order = rng.sample(names, len(names))
        run.orders.append(order)
        for slot, name in enumerate(order):
            w = once(name)
            run.samples.append(Sample(name, r, slot, w.stop_at - w.start_at, w.tokens, dict(w.marks)))
    return run


def environment(device: torch.device, seed: int, models: dict | None = None) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    packages = ["torch", "transformers", "numpy", "scipy"]
    record = {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "device": str(device),
        "gpu": gpu_name(device),
        "seed": seed,
    }
    if models is not None:
        record["models"] = models
    return record


def cpu_name() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if out.returncode == 0:
            return out.stdout.strip()
    return platform.processor() or platform.machine()


def gpu_name(device: torch.device) -> str | None:
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(device)} (driver CUDA {torch.version.cuda})"
    if device.type == "mps":
        return f"{cpu_name()} integrated GPU (MPS)"
    return None
