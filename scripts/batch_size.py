"""Batch-size sweep: results/batch_size.json.

Speculative decoding trades extra target compute (k + 1 positions verified per round, most
of them for nothing when acceptance is low) for fewer sequential target passes. At batch 1
a decode step is bound by reading the weights, so the extra positions are nearly free; as
the batch grows each step does more arithmetic and the trade gets worse. This measures how
fast the advantage shrinks.

Methods: `cached` (plain decoding of the whole batch, baselines.cached_autoregressive
.generate_batch), `spec_ngram` and `spec_draft`, at batch sizes 1, 2, 4 and 8, every
configuration interleaved in the same rounds. A call decodes one batch of prompts from one
category; within a round every method at a given batch size gets the same batch.

Short prompts only. On the 1.7B target K/V costs 192 KiB per token, so eight 600-token
sequences need ~1 GB of cache next to 4.7 GiB of weights and buffers, past the 5.33 GiB
MPS limit of an 8 GB machine. 20 measured rounds give a usable median, not a p99.
"""
import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch
import yaml

from baselines.cached_autoregressive import generate_batch
from bench.harness import Stopwatch, Workload, environment, measure, synchronizer
from bench.latency import summarize
from lm.load import load_model, load_tokenizer, resolve_device, tiny_model
from scripts.benchmark import between, load_prompts
from scripts.block_size import SharedPool
from spec.draft import ModelDraft
from spec.loop import generate
from spec.ngram import NGramDraft

BATCHES = [1, 2, 4, 8]
METHODS = ["cached", "spec_ngram", "spec_draft"]


def workloads(target, draft, prompts: list[list[int]], device, settings: dict, eos_id: int | None,
              acceptance: dict[str, list[tuple[int, int]]]) -> dict[str, Workload]:
    k, max_new = settings["k"], settings["max_new_tokens"]
    per_seq = -(-(max(len(p) for p in prompts) + max_new + k + 1) // 16) + 1
    pool = SharedPool(device)
    out: dict[str, Workload] = {}
    for b in sorted(BATCHES, reverse=True):  # largest pool first; smaller batches reuse it
        target_cache = pool(target, per_seq * b, 16)
        drafters = {"spec_ngram": NGramDraft(target.config.vocab_size),
                    "spec_draft": ModelDraft(draft, pool(draft, per_seq * b, 16))}

        def batches(b=b):
            # Call j decodes prompts j*b .. j*b + b - 1 (mod n): the same batch for every
            # method at this batch size in the same round.
            for j in itertools.count():
                yield [prompts[(j * b + i) % len(prompts)] for i in range(b)]

        def cached(watch: Stopwatch, batch=batches(), cache=target_cache) -> int:
            outs = generate_batch(target, cache, next(batch), max_new, eos_id=eos_id, on_event=watch.mark)
            return sum(map(len, outs))

        out[f"cached@{b}"] = cached
        for name, drafter in drafters.items():
            def spec(watch: Stopwatch, batch=batches(), cache=target_cache, drafter=drafter, key=f"{name}@{b}") -> int:
                outs, stats = generate(target, cache, drafter, next(batch), max_new, k=k, eos_id=eos_id,
                                       on_event=watch.mark)
                acceptance.setdefault(key, []).append((stats.accepted, stats.drafted))
                return sum(map(len, outs))

            out[f"{name}@{b}"] = spec
    return {name: out[name] for name in sorted(out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(open("config.yaml"))
    settings = {key: value for key, value in config["bench"].items() if key != "conditions"}
    settings.update(warmup=2, repeats=20, edge_rounds=3, batches=BATCHES)

    if args.tiny:
        device = torch.device("cpu")
        target, draft = tiny_model(seed=0, n_layers=4), tiny_model(seed=0, n_layers=4)
        draft.model.layers = draft.model.layers[:1]
        draft.config.num_hidden_layers = 1
        tokenize = lambda text: [b % target.config.vocab_size for b in text.encode()][:128]  # noqa: E731
        eos_id, models = None, "tiny random Llama, 4-layer target, 1-layer early-exit draft"
        settings.update(warmup=1, repeats=2, max_new_tokens=16, edge_rounds=1)
    else:
        device = resolve_device(config["device"])
        spec, cache_dir = config["models"], config["models"]["cache_dir"]
        tokenizer = load_tokenizer(spec["target"], cache_dir, local_files_only=True)
        target = load_model(spec["target"], cache_dir, device, local_files_only=True)
        draft = load_model(spec["draft"], cache_dir, device, local_files_only=True)
        tokenize = lambda text: tokenizer(text)["input_ids"]  # noqa: E731
        eos_id, models = tokenizer.eos_token_id, {"target": spec["target"], "draft": spec["draft"]}

    out_path = Path("results/batch_size.tiny.raw.json" if args.tiny else "results/batch_size.json")
    record = json.loads(out_path.read_text()) if out_path.exists() else {}
    if record.get("settings") != settings:
        record = {"settings": settings, "categories": {}}
    record["environment"] = environment(device, config["seed"], models)
    path = config["bench"]["conditions"]["short"]
    for category, items in load_prompts(path).items():
        if category in record["categories"]:
            continue
        prompts = [tokenize(t) for t in items]
        acceptance: dict[str, list[tuple[int, int]]] = {}
        run = measure(workloads(target, draft, prompts, device, settings, eos_id, acceptance),
                      settings["warmup"], settings["repeats"], synchronizer(device), seed=config["seed"],
                      after_each=between(device, f"batch {category}"))
        summary = summarize(run, settings["edge_rounds"])
        for name, cell in summary.items():
            cell["tokens_per_s_p50"] = statistics.median(s.tokens / s.total for s in run.samples if s.config == name)
        for name, calls in acceptance.items():
            measured = calls[settings["warmup"]:]
            summary[name]["alpha"] = sum(a for a, _ in measured) / max(sum(d for _, d in measured), 1)
        record["categories"][category] = summary
        out_path.write_text(json.dumps(record, indent=1) + "\n")
        print(category, json.dumps({name: round(cell["tokens_per_s_p50"], 1) for name, cell in summary.items()}),
              flush=True)


if __name__ == "__main__":
    main()
