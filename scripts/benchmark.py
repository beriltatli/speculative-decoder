"""Batch-size-1 latency of every method on every prompt category, under each prompt-length
condition in config.yaml: results/latency.json. Written after every (condition, category)
cell group, and a rerun skips groups already measured with the same settings, so a
multi-hour run survives interruption.

Methods, all on the same prompts within a round and interleaved by bench.harness:
  no_cache           full-prefix recomputation each step (the floor)
  cached             paged KV cache, plain autoregressive decoding (the real baseline)
  spec_ngram         speculative decoding with a prompt-lookup draft, no draft model
  spec_draft         speculative decoding with the draft model
  spec_self          speculative decoding with the target as its own draft: acceptance ~1,
                     two models' cost, so it must come out slower than `cached`
  hf_assisted        transformers' assistant_model path, the one permitted reference row

--tiny swaps in random-weight tiny models on CPU: a check that the pipeline runs, whose
numbers mean nothing.
"""
import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch
import yaml
from transformers.generation.streamers import BaseStreamer

from baselines import cached_autoregressive, no_cache
from bench.harness import Stopwatch, Workload, environment, measure, synchronizer
from bench.latency import summarize
from bench.predictability import gzip_ratio, repeated_ngram_rate
from cache.kv import KVCache
from lm.load import check_shared_vocab, load_config, load_model, load_tokenizer, resolve_device, tiny_model
from spec.draft import ModelDraft
from spec.loop import generate
from spec.ngram import NGramDraft


class FirstTokenStreamer(BaseStreamer):
    """generate() calls put() once with the prompt, then with each step's new tokens. In
    assisted generation the first step is a whole speculative round, so its first token and
    its first round end at the same moment."""

    def __init__(self, watch: Stopwatch) -> None:
        self.watch, self.calls = watch, 0

    def put(self, value) -> None:
        self.calls += 1
        if self.calls == 2:
            self.watch.mark("first_token")
            self.watch.mark("first_round")

    def end(self) -> None:
        pass


def workloads(target, draft, prompts: list[list[int]], device, settings: dict, eos_id: int | None,
              acceptance: dict[str, list[tuple[int, int]]], reference: dict[int, list[int]]) -> dict[str, Workload]:
    k, max_new = settings["k"], settings["max_new_tokens"]
    longest = max(len(p) for p in prompts) + max_new + k + 1

    def cache(model) -> KVCache:
        # Allocated once per method, outside the timed region; generate() frees its blocks.
        blocks = -(-longest // 16) + 1
        return KVCache.for_model(model.config, num_blocks=blocks, block_size=16, dtype=model.dtype, device=device)

    drafters = {
        "spec_ngram": NGramDraft(target.config.vocab_size),
        "spec_draft": ModelDraft(draft, cache(draft)),
        "spec_self": ModelDraft(target, cache(target)),
    }
    target_caches = {name: cache(target) for name in ["cached", *drafters]}

    def cycling(body):
        # Every method is called equally often in the same rounds, so call j of each method
        # sees prompt j: within a round all methods run the same prompt.
        order = itertools.cycle(range(len(prompts)))

        def workload(watch: Stopwatch) -> int:
            return body(next(order), watch)

        return workload

    def run_no_cache(i, watch):
        return len(no_cache.generate(target, prompts[i], max_new, eos_id=eos_id, on_event=watch.mark))

    def run_cached(i, watch):
        out = cached_autoregressive.generate(target, target_caches["cached"], prompts[i], max_new, eos_id=eos_id,
                                             on_event=watch.mark)
        reference[i] = out
        return len(out)

    def run_spec(name: str):
        def body(i, watch):
            out, stats = generate(target, target_caches[name], drafters[name], [prompts[i]], max_new, k=k, eos_id=eos_id,
                                  on_event=watch.mark)
            acceptance.setdefault(name, []).append((stats.accepted, stats.drafted))
            return len(out[0])

        return body

    draft.generation_config.num_assistant_tokens = k
    draft.generation_config.num_assistant_tokens_schedule = "constant"
    draft.generation_config.assistant_confidence_threshold = 0.0

    def run_hf(i, watch):
        prompt = prompts[i]
        ids = torch.tensor([prompt], device=device)
        out = target.generate(ids, attention_mask=torch.ones_like(ids), assistant_model=draft, do_sample=False,
                              max_new_tokens=max_new, eos_token_id=eos_id, pad_token_id=eos_id or 0,
                              streamer=FirstTokenStreamer(watch))
        return out.shape[1] - len(prompt)

    return {
        "no_cache": cycling(run_no_cache),
        "cached": cycling(run_cached),
        **{name: cycling(run_spec(name)) for name in drafters},
        "hf_assisted": cycling(run_hf),
    }


def load_prompts(path: str) -> dict[str, list[str]]:
    record = json.loads(Path(path).read_text())
    return record.get("prompts", record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(open("config.yaml"))
    settings = {key: value for key, value in config["bench"].items() if key != "conditions"}
    conditions = config["bench"]["conditions"]

    if args.tiny:
        device = torch.device("cpu")
        target, draft = tiny_model(seed=0, n_layers=4), tiny_model(seed=0, n_layers=4)
        draft.model.layers = draft.model.layers[:1]
        draft.config.num_hidden_layers = 1
        tokenize = lambda text: [b % target.config.vocab_size for b in text.encode()][:128]  # noqa: E731
        detokenize = None
        eos_id, models = None, "tiny random Llama, 4-layer target, 1-layer early-exit draft"
        settings.update(warmup=1, repeats=4, max_new_tokens=16, edge_rounds=1)
    else:
        device = resolve_device(config["device"])
        spec, cache_dir = config["models"], config["models"]["cache_dir"]
        tokenizer = load_tokenizer(spec["target"], cache_dir, local_files_only=True)
        check_shared_vocab(load_config(spec["target"], cache_dir), load_config(spec["draft"], cache_dir),
                           tokenizer, load_tokenizer(spec["draft"], cache_dir, local_files_only=True))
        target = load_model(spec["target"], cache_dir, device, local_files_only=True)
        draft = load_model(spec["draft"], cache_dir, device, local_files_only=True)
        tokenize = lambda text: tokenizer(text)["input_ids"]  # noqa: E731
        detokenize = tokenizer.decode
        eos_id, models = tokenizer.eos_token_id, {"target": spec["target"], "draft": spec["draft"]}

    out_path = Path("results/latency.tiny.raw.json" if args.tiny else "results/latency.json")
    record = json.loads(out_path.read_text()) if out_path.exists() else {}
    if record.get("settings") != settings:
        record = {"settings": settings, "conditions": {}}
    record["environment"] = environment(device, config["seed"], models)

    for condition, path in conditions.items():
        texts = load_prompts(path)
        done = record["conditions"].setdefault(condition, {"prompts": path, "categories": {}, "prompt_tokens": {},
                                                           "predictability": {}})
        for category, items in texts.items():
            if category in done["categories"]:
                continue
            # More rounds than prompts cycles through the category again: call j uses prompt j mod n.
            prompts = [tokenize(t) for t in items]
            acceptance: dict[str, list[tuple[int, int]]] = {}
            reference: dict[int, list[int]] = {}
            run = measure(workloads(target, draft, prompts, device, settings, eos_id, acceptance, reference),
                          settings["warmup"], settings["repeats"], synchronizer(device), seed=config["seed"])
            summary = summarize(run, settings["edge_rounds"])
            for name, calls in acceptance.items():
                # Measured calls only: warmup calls come first for every method.
                measured = calls[settings["warmup"]:]
                summary[name]["alpha"] = sum(a for a, _ in measured) / max(sum(d for _, d in measured), 1)
            lengths = sorted(len(p) for p in prompts)
            done["prompt_tokens"][category] = {"min": lengths[0], "p50": lengths[len(lengths) // 2], "max": lengths[-1]}
            if detokenize:
                # Prompt plus the target's own greedy continuation, scored without a model.
                joined = [items[i] + detokenize(out) for i, out in sorted(reference.items())]
                done["predictability"][category] = {
                    "gzip_ratio_p50": statistics.median(map(gzip_ratio, joined)),
                    "repeated_4gram_rate_p50": statistics.median(map(repeated_ngram_rate, joined)),
                }
            done["categories"][category] = summary
            out_path.write_text(json.dumps(record, indent=1) + "\n")
            print(condition, category, json.dumps({m: [(s[key] or {}).get("p50") for key in ("e2e_ms", "ttft_ms", "tpot_ms")]
                                                   for m, s in summary.items()}), flush=True)


if __name__ == "__main__":
    main()
