"""Greedy identity on real weights: speculative output against plain cached greedy.

Writes results/greedy_identity.json with two runs:
  exact     the identity pair in fp64 on CPU; every prompt must match.
  main      the configured target/draft pair in its own dtype on the default device. Every
            mismatch is recorded with the reference's top-2 logit gap at the first divergent
            position and with what the draft had proposed there, so a flip can be checked
            against the dtype's rounding grid and for a bias toward the draft.

Each finished prompt is appended to results/greedy_identity.<run>.raw.json, so an
interrupted run resumes where it stopped.
"""
import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from baselines import cached_autoregressive
from bench.quality import RecordingDraft, first_divergence
from cache.kv import KVCache
from lm.forward import logits as output_logits
from lm.load import check_shared_vocab, load_config, load_model, load_tokenizer, resolve_device
from spec.loop import generate


def select_prompts(counts: list[int]) -> list[tuple[str, str]]:
    prompts = json.loads(Path("prompts/prompts.json").read_text())
    return [(category, text) for category, n in zip(prompts, counts) for text in prompts[category][:n]]


def new_cache(model: torch.nn.Module, device: torch.device) -> KVCache:
    return KVCache.for_model(model.config, num_blocks=256, block_size=16, dtype=model.dtype, device=device)


def ulp(x: float, dtype: torch.dtype) -> float:
    """Spacing of the dtype's grid at |x|."""
    v = torch.tensor(abs(x), dtype=dtype)
    return float(torch.nextafter(v, torch.tensor(float("inf"), dtype=dtype)) - v)


def diagnose(target: torch.nn.Module, ids: list[int], want: list[int], at: int, device: torch.device) -> dict[str, Any]:
    """The reference's top-2 logit gap at the first divergent position, from the same output
    projection the generation used (lm.forward.logits, at least fp32), and the spacing of the
    model's compute-dtype grid at that magnitude: the scale at which a 1-token and a
    (k+1)-token pass can disagree when their hidden states are rounded to that dtype."""
    with torch.no_grad():
        hidden = target.model(torch.tensor([ids + want[:at]], device=device), use_cache=False).last_hidden_state
        top = output_logits(target, hidden[0, -1:])[0].topk(2).values.tolist()
    return {"diverge_at": at, "top2": top, "gap": top[0] - top[1], "ulp": ulp(top[0], target.dtype)}


def run(name: str, target_spec: dict, draft_spec: dict, dtype: str, device: torch.device, config: dict) -> dict[str, Any]:
    cache_dir = config["models"]["cache_dir"]
    settings = config["greedy_identity"][name]
    k = config["greedy_identity"]["k"]
    max_new = settings["max_new_tokens"]
    target_tok = load_tokenizer(target_spec, cache_dir, local_files_only=True)
    draft_tok = load_tokenizer(draft_spec, cache_dir, local_files_only=True)
    check_shared_vocab(load_config(target_spec, cache_dir), load_config(draft_spec, cache_dir), target_tok, draft_tok)
    target = load_model(target_spec, cache_dir, device, dtype, local_files_only=True)
    draft = load_model(draft_spec, cache_dir, device, dtype, local_files_only=True)
    eos = target_tok.eos_token_id

    partial = Path(f"results/greedy_identity.{name}.raw.json")
    partial.parent.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in partial.read_text().splitlines()] if partial.exists() else []
    started = time.perf_counter()
    for i, (category, text) in enumerate(select_prompts(settings["prompts_per_category"])):
        if i < len(rows):
            continue
        ids = target_tok(text)["input_ids"]
        want = cached_autoregressive.generate(target, new_cache(target, device), ids, max_new, eos_id=eos)
        drafter = RecordingDraft(draft, new_cache(draft, device))
        out, stats = generate(target, new_cache(target, device), drafter, [ids], max_new, k=k, eos_id=eos)
        row: dict[str, Any] = {"index": i, "category": category, "identical": out[0] == want,
                               "alpha": stats.accepted / max(stats.drafted, 1)}
        at = first_divergence(out[0], want)
        if at is not None:
            row.update(diagnose(target, ids, want, at, device),
                       reference_token=want[at] if at < len(want) else None,
                       chosen_token=out[0][at] if at < len(out[0]) else None,
                       draft_token=drafter.proposed_at(len(ids) + at))
        rows.append(row)
        with partial.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{name} {i:3d} {category:10s} identical={row['identical']}", flush=True)
    return {
        "target": target_spec, "draft": draft_spec, "dtype": dtype, "device": str(device),
        "logits_dtype": str(torch.promote_types(getattr(torch, dtype), torch.float32)).removeprefix("torch."),
        "k": k, "max_new_tokens": max_new, "prompts": len(rows), "identical": sum(r["identical"] for r in rows),
        "seconds_last_session": round(time.perf_counter() - started, 1), "rows": rows,
    }


def rediagnose(name: str, target_spec: dict, dtype: str, device: torch.device, config: dict) -> None:
    """Recompute the divergence fields of an existing run with the current diagnose(), after a
    change to how the gap is measured. Each reference is regenerated and must agree with the
    recorded reference token, or the run being re-read is not the one being recomputed."""
    cache_dir = config["models"]["cache_dir"]
    settings = config["greedy_identity"][name]
    tokenizer = load_tokenizer(target_spec, cache_dir, local_files_only=True)
    target = load_model(target_spec, cache_dir, device, dtype, local_files_only=True)
    partial = Path(f"results/greedy_identity.{name}.raw.json")
    rows = [json.loads(line) for line in partial.read_text().splitlines()]
    prompts = select_prompts(settings["prompts_per_category"])
    for row in rows:
        if row["identical"]:
            continue
        ids = tokenizer(prompts[row["index"]][1])["input_ids"]
        want = cached_autoregressive.generate(target, new_cache(target, device), ids, settings["max_new_tokens"],
                                              eos_id=tokenizer.eos_token_id)
        at = row["diverge_at"]
        if at < len(want) and want[at] != row["reference_token"]:
            raise RuntimeError(f"prompt {row['index']}: regenerated reference differs from the recorded one")
        row.update(diagnose(target, ids, want, at, device))
    partial.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out_path = Path("results/greedy_identity.json")
    record = json.loads(out_path.read_text())
    record[name]["rows"] = rows
    out_path.write_text(json.dumps(record, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["exact", "main"])
    parser.add_argument("--rediagnose", choices=["exact", "main"])
    args = parser.parse_args()
    config = yaml.safe_load(open("config.yaml"))
    models = config["models"]
    out_path = Path("results/greedy_identity.json")
    jobs = {
        "exact": (models["identity_target"], models["draft"], "float64", torch.device("cpu")),
        "main": (models["target"], models["draft"], models["target"]["dtype"], resolve_device(config["device"])),
    }
    if args.rediagnose:
        target_spec, _, dtype, device = jobs[args.rediagnose]
        rediagnose(args.rediagnose, target_spec, dtype, device, config)
        return
    for name, job in jobs.items():
        if args.only not in (None, name):
            continue
        result = run(name, *job, config)
        # Re-read before writing: the other run may have finished in another process meanwhile.
        record = json.loads(out_path.read_text()) if out_path.exists() else {}
        record[name] = result
        record["machine"] = platform.platform()
        out_path.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
