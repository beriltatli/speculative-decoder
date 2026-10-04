"""Try speculative decoding on your own prompts.

    python -m scripts.demo "Write a Python function that reverses a string."
    python -m scripts.demo --max-new 96 "First prompt" "Second prompt"
    python -m scripts.demo --raw "Text to continue as-is, without the chat template"

For each prompt this runs plain cached decoding, then speculative decoding with the 135M
draft model and with prompt lookup, and prints:
  - the answer, with the tokens the draft proposed and the target accepted in green;
  - whether every method produced exactly the same tokens (greedy, so they must, up to the
    bf16 rounding ties described in the README);
  - each method's time, tokens per second and speedup over plain decoding.

The models load once (~20 s), and one untimed warmup call runs before the first prompt so
kernel compilation is not counted against whichever method happens to go first.
"""
import argparse
import time

import torch
import yaml

from baselines import cached_autoregressive
from bench.harness import synchronizer
from cache.kv import KVCache
from lm.load import load_model, load_tokenizer, resolve_device
from spec.draft import ModelDraft
from spec.loop import generate
from spec.ngram import NGramDraft

GREEN, DIM, BOLD, RED, RESET = "\033[32m", "\033[2m", "\033[1m", "\033[31m", "\033[0m"


def colored(tokenizer, tokens: list[int], drafted_per_round: list[int], emitted_per_round: list[int]) -> str:
    """The first token comes from the target's prefill; after that each round emits its
    accepted draft tokens first, then the target's own token."""
    from_draft = [False]
    for drafted, emitted in zip(drafted_per_round, emitted_per_round):
        from_draft += [True] * drafted + [False] * (emitted - drafted)
    out, prev = [], ""
    for i in range(len(tokens)):
        text = tokenizer.decode(tokens[: i + 1], skip_special_tokens=True)
        piece = text[len(prev):]
        prev = text
        out.append(f"{GREEN}{piece}{RESET}" if from_draft[i] else piece)
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--raw", action="store_true", help="skip the chat template")
    args = parser.parse_args()

    config = yaml.safe_load(open("config.yaml"))
    spec, cache_dir = config["models"], config["models"]["cache_dir"]
    device = resolve_device(config["device"])
    sync = synchronizer(device)
    print(f"{DIM}Loading {spec['target']['repo']} and {spec['draft']['repo']} on {device}...{RESET}", flush=True)
    tokenizer = load_tokenizer(spec["target"], cache_dir, local_files_only=True)
    target = load_model(spec["target"], cache_dir, device, local_files_only=True)
    draft = load_model(spec["draft"], cache_dir, device, local_files_only=True)
    eos = tokenizer.eos_token_id

    def encode(text: str) -> list[int]:
        if args.raw:
            return tokenizer(text)["input_ids"]
        return tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                             tokenize=True, return_dict=False)

    encoded = [encode(p) for p in args.prompts]
    blocks = -(-(max(map(len, encoded)) + args.max_new + args.k + 1) // 16) + 1
    target_cache = KVCache.for_model(target.config, blocks, 16, target.dtype, device)
    draft_cache = KVCache.for_model(draft.config, blocks, 16, draft.dtype, device)

    def plain(ids):
        return cached_autoregressive.generate(target, target_cache, ids, args.max_new, eos_id=eos), None

    def with_draft(ids):
        out, stats = generate(target, target_cache, ModelDraft(draft, draft_cache), [ids], args.max_new, k=args.k,
                              eos_id=eos)
        return out[0], stats

    def with_lookup(ids):
        out, stats = generate(target, target_cache, NGramDraft(target.config.vocab_size), [ids], args.max_new,
                              k=args.k, eos_id=eos)
        return out[0], stats

    methods = {"plain decoding": plain, "draft model (135M)": with_draft, "prompt lookup": with_lookup}
    for run in methods.values():  # warmup, untimed
        run(encoded[0][:16])

    for text, ids in zip(args.prompts, encoded):
        print(f"\n{BOLD}Prompt:{RESET} {text}  {DIM}({len(ids)} tokens){RESET}")
        results = {}
        for name, run in methods.items():
            sync()
            start = time.perf_counter()
            tokens, stats = run(ids)
            sync()
            results[name] = (tokens, stats, time.perf_counter() - start)

        reference, _, base = results["plain decoding"]
        _, draft_stats, _ = results["draft model (135M)"]
        print(f"{BOLD}Answer{RESET} {DIM}(green: proposed by the draft model and accepted by the target){RESET}")
        print(colored(tokenizer, results["draft model (135M)"][0], draft_stats.drafted_per_round,
                      draft_stats.emitted_per_round))
        print()
        print(f"  {'method':20s} {'time':>8s} {'tok/s':>7s} {'speedup':>8s} {'accepted':>9s}  same tokens")
        for name, (tokens, stats, seconds) in results.items():
            accepted = f"{stats.accepted / max(stats.drafted, 1):.0%}" if stats else "–"
            same = f"{GREEN}yes{RESET}" if tokens == reference else f"{RED}no, differs at token " \
                f"{next(i for i, (a, b) in enumerate(zip(tokens + [-1], reference + [-1])) if a != b)}{RESET}"
            print(f"  {name:20s} {seconds:7.2f}s {len(tokens) / seconds:7.1f} {base / seconds:7.2f}× {accepted:>9s}  {same}")


if __name__ == "__main__":
    main()
