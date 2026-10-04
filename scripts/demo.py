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

import yaml

from baselines import cached_autoregressive
from bench.harness import synchronizer
from cache.kv import KVCache
from lm.load import load_model, load_tokenizer, resolve_device
from spec.draft import ModelDraft
from spec.loop import generate
from spec.ngram import NGramDraft

GREEN, DIM, BOLD, RED, RESET = "\033[32m", "\033[2m", "\033[1m", "\033[31m", "\033[0m"


METHODS = {"plain": "plain decoding", "draft": "draft model (135M)", "lookup": "prompt lookup"}
MAX_PROMPT_TOKENS = 1024
MAX_NEW, MAX_K = 256, 8  # K/V for 1024 + 256 + 9 positions is ~250 MB, inside the 8 GB budget


def segments(tokenizer, tokens: list[int], stats) -> list[tuple[str, bool]]:
    """(text, from_draft) per token. The first token comes from the target's prefill; after
    that each round emits its accepted draft tokens first, then the target's own token."""
    from_draft = [False] * len(tokens)
    if stats is not None:
        flags = [False]
        for drafted, emitted in zip(stats.drafted_per_round, stats.emitted_per_round):
            flags += [True] * drafted + [False] * (emitted - drafted)
        from_draft = flags
    out, prev = [], ""
    for i in range(len(tokens)):
        text = tokenizer.decode(tokens[: i + 1], skip_special_tokens=True)
        out.append((text[len(prev):], from_draft[i]))
        prev = text
    return out


class Demo:
    """Loads the configured target and draft once and runs all three methods on a prompt."""

    def __init__(self) -> None:
        config = yaml.safe_load(open("config.yaml"))
        spec, cache_dir = config["models"], config["models"]["cache_dir"]
        self.device = resolve_device(config["device"])
        self.sync = synchronizer(self.device)
        self.names = {"target": spec["target"]["repo"], "draft": spec["draft"]["repo"], "device": str(self.device)}
        self.tokenizer = load_tokenizer(spec["target"], cache_dir, local_files_only=True)
        self.target = load_model(spec["target"], cache_dir, self.device, local_files_only=True)
        self.draft = load_model(spec["draft"], cache_dir, self.device, local_files_only=True)
        # Allocated once for the largest request. Allocating per request put the cost of
        # fresh GPU memory on whichever method ran first, which inflated plain decoding by
        # seconds and the speedups with it.
        blocks = -(-(MAX_PROMPT_TOKENS + MAX_NEW + MAX_K + 1) // 16) + 1
        self.target_cache = KVCache.for_model(self.target.config, blocks, 16, self.target.dtype, self.device)
        self.draft_cache = KVCache.for_model(self.draft.config, blocks, 16, self.draft.dtype, self.device)
        for _ in self.run("Hello", max_new=8):  # warmup: kernel compilation, untimed
            pass

    def encode(self, text: str, raw: bool = False) -> list[int]:
        if raw:
            return self.tokenizer(text)["input_ids"]
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                                  tokenize=True, return_dict=False)

    def run(self, text: str, max_new: int = 64, k: int = 4, raw: bool = False):
        """Yields one result dict per method as soon as it finishes, plain decoding first."""
        ids = self.encode(text, raw)
        if len(ids) > MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt is {len(ids)} tokens; the limit is {MAX_PROMPT_TOKENS}")
        max_new, k = min(max_new, MAX_NEW), min(k, MAX_K)
        eos = self.tokenizer.eos_token_id
        target_cache, draft_cache = self.target_cache, self.draft_cache
        drafters = {"draft": lambda: ModelDraft(self.draft, draft_cache),
                    "lookup": lambda: NGramDraft(self.target.config.vocab_size)}
        reference, base = None, None
        for key, name in METHODS.items():
            self.sync()
            start = time.perf_counter()
            if key == "plain":
                tokens, stats = cached_autoregressive.generate(self.target, target_cache, ids, max_new, eos_id=eos), None
            else:
                out, stats = generate(self.target, target_cache, drafters[key](), [ids], max_new, k=k, eos_id=eos)
                tokens = out[0]
            self.sync()
            seconds = time.perf_counter() - start
            if reference is None:
                reference, base = tokens, seconds
            differs = next((i for i, (a, b) in enumerate(zip(tokens + [-1], reference + [-1])) if a != b), None)
            yield {
                "key": key, "name": name, "prompt_tokens": len(ids), "tokens": len(tokens), "seconds": seconds,
                "tokens_per_s": len(tokens) / seconds, "speedup": base / seconds,
                "accepted": stats.accepted / max(stats.drafted, 1) if stats else None,
                "same_as_plain": differs is None, "differs_at": differs,
                "segments": segments(self.tokenizer, tokens, stats),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--raw", action="store_true", help="skip the chat template")
    args = parser.parse_args()

    print(f"{DIM}Loading models...{RESET}", flush=True)
    demo = Demo()
    for text in args.prompts:
        results = list(demo.run(text, args.max_new, args.k, args.raw))
        print(f"\n{BOLD}Prompt:{RESET} {text}  {DIM}({results[0]['prompt_tokens']} tokens){RESET}")
        print(f"{BOLD}Answer{RESET} {DIM}(green: proposed by the draft model and accepted by the target){RESET}")
        print("".join(f"{GREEN}{t}{RESET}" if d else t for t, d in results[1]["segments"]))
        print()
        print(f"  {'method':20s} {'time':>8s} {'tok/s':>7s} {'speedup':>8s} {'accepted':>9s}  same tokens")
        for r in results:
            accepted = f"{r['accepted']:.0%}" if r["accepted"] is not None else "–"
            same = f"{GREEN}yes{RESET}" if r["same_as_plain"] else f"{RED}no, differs at token {r['differs_at']}{RESET}"
            print(f"  {r['name']:20s} {r['seconds']:7.2f}s {r['tokens_per_s']:7.1f} {r['speedup']:7.2f}× "
                  f"{accepted:>9s}  {same}")


if __name__ == "__main__":
    main()
