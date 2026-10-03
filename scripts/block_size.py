"""Block-size sweep: results/block_size.json.

Two costs pull the block size in opposite directions:
  waste        the unused tail of each sequence's last block. Computed without a model from
               the tokenized prompts of both conditions, through the same BlockTable the cache
               uses, at the length a sequence reaches under plain decoding (prompt + max_new - 1)
               and at the speculative peak (k more, just after a verify append). A contiguous
               cache that reserves the workload's longest sequence for every request is the
               reference row.
  bookkeeping  per-step work that grows with the number of blocks: slot tables are built in
               Python from each sequence's block list on every append. Measured as latency of
               `cached` and `spec_draft` on long prompts, with every block size interleaved
               in the same rounds so drift lands on all of them alike.

All block sizes share one K/V pool per model. The pool's contents are irrelevant between
calls (every read is of a slot written in the same call), and one pool per size would cost
~630 MB for the target on a machine that does not have it.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import yaml

from bench.harness import environment, measure, synchronizer
from bench.latency import summarize
from cache.kv import KVCache
from cache.paged import BlockAllocator, BlockTable
from lm.load import load_model, load_tokenizer, resolve_device, tiny_model
from scripts.benchmark import between, load_prompts, workloads

SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
TIMED_SIZES = [1, 4, 16, 64, 256]
METHODS = ["cached", "spec_draft"]


def allocated(length: int, block_size: int) -> int:
    allocator = BlockAllocator(math.ceil(length / block_size) + 1)
    table = BlockTable()
    table.grow_to(length, allocator, block_size)
    return len(table.blocks) * block_size


def waste(lengths: list[int], block_size: int) -> dict[str, float]:
    slots = [allocated(n, block_size) for n in lengths]
    return {
        "waste_fraction": 1 - sum(lengths) / sum(slots),
        "wasted_slots_per_seq_p50": statistics.median(s - n for s, n in zip(slots, lengths)),
        "blocks_per_seq_p50": statistics.median(s // block_size for s in slots),
    }


def waste_table(prompt_lengths: dict[str, list[int]], max_new: int, k: int) -> dict:
    out: dict = {}
    for condition, lengths in prompt_lengths.items():
        plain = [n + max_new - 1 for n in lengths]
        peak = [n + k for n in plain]
        reserve = max(lengths) + max_new + k + 1
        out[condition] = {
            "sequences": len(lengths),
            "paged": {str(b): {"plain": waste(plain, b), "spec_peak": waste(peak, b)} for b in SIZES},
            "contiguous": {"reserved_slots": reserve,
                           "waste_fraction": 1 - sum(plain) / (reserve * len(plain))},
        }
    return out


class SharedPool:
    """Cache factory for scripts.benchmark.workloads: every cache it makes for a model shares
    that model's one K/V pool, sized for the largest request so far, and has its own
    allocator and block size. Safe only while caches of one model are used one call at a
    time, which is how the harness runs them: a call writes every slot it reads, and frees
    its blocks before returning."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.pools: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(self, model, num_blocks: int, block_size: int) -> KVCache:
        cache = KVCache.for_model(model.config, num_blocks=1, block_size=1, dtype=model.dtype, device=self.device)
        cache.allocator = BlockAllocator(num_blocks)
        cache.block_size = block_size
        pool = self.pools.get(id(model))
        if pool is None or pool[0].shape[1] < num_blocks * block_size:
            shape = (cache.k.shape[0], num_blocks * block_size, *cache.k.shape[2:])
            pool = self.pools[id(model)] = (torch.zeros(shape, dtype=model.dtype, device=self.device),
                                            torch.zeros(shape, dtype=model.dtype, device=self.device))
        cache.k, cache.v = pool
        return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(open("config.yaml"))
    settings = {key: value for key, value in config["bench"].items() if key != "conditions"}
    settings.update(warmup=2, repeats=30, edge_rounds=3)
    conditions = config["bench"]["conditions"]

    if args.tiny:
        device = torch.device("cpu")
        target, draft = tiny_model(seed=0, n_layers=4), tiny_model(seed=0, n_layers=4)
        draft.model.layers = draft.model.layers[:1]
        draft.config.num_hidden_layers = 1
        tokenize = lambda text: [b % target.config.vocab_size for b in text.encode()][:128]  # noqa: E731
        eos_id, models = None, "tiny random Llama, 4-layer target, 1-layer early-exit draft"
        settings.update(warmup=1, repeats=3, max_new_tokens=16, edge_rounds=1)
    else:
        device = resolve_device(config["device"])
        spec, cache_dir = config["models"], config["models"]["cache_dir"]
        tokenizer = load_tokenizer(spec["target"], cache_dir, local_files_only=True)
        target = load_model(spec["target"], cache_dir, device, local_files_only=True)
        draft = load_model(spec["draft"], cache_dir, device, local_files_only=True)
        tokenize = lambda text: tokenizer(text)["input_ids"]  # noqa: E731
        eos_id, models = tokenizer.eos_token_id, {"target": spec["target"], "draft": spec["draft"]}

    texts = {c: load_prompts(path) for c, path in conditions.items()}
    lengths = {c: [len(tokenize(t)) for items in cats.values() for t in items] for c, cats in texts.items()}
    record = {"settings": settings, "environment": environment(device, config["seed"], models),
              "waste": waste_table(lengths, settings["max_new_tokens"], settings["k"])}

    # 10 long prompts per category, interleaved, so call j of every configuration sees the
    # same prompt and the categories are spread evenly over the rounds.
    per = [[tokenize(t) for t in items[:10]] for items in texts["long"].values()]
    prompts = [p for group in zip(*per) for p in group]
    pooled = SharedPool(device)

    longest = max(len(p) for p in prompts) + settings["max_new_tokens"] + settings["k"] + 1
    # Largest pool first, so every later cache reuses it.
    by_slots = sorted(TIMED_SIZES, key=lambda b: (math.ceil(longest / b) + 1) * b, reverse=True)
    timed = {}
    for b in by_slots:
        w = workloads(target, draft, prompts, device, settings, eos_id, {}, {}, block_size=b, new_cache=pooled)
        for method in METHODS:
            timed[f"{method}@{b}"] = w[method]
    timed = {name: timed[name] for name in sorted(timed)}
    run = measure(timed, settings["warmup"], settings["repeats"], synchronizer(device), seed=config["seed"],
                  after_each=between(device, "block size"))
    record["latency"] = summarize(run, settings["edge_rounds"])
    out_path = Path("results/block_size.tiny.raw.json" if args.tiny else "results/block_size.json")
    out_path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({name: [s[key]["p50"] for key in ("tpot_ms", "e2e_ms")] for name, s in record["latency"].items()}))


if __name__ == "__main__":
    main()
