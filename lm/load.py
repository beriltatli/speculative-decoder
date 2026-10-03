from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from lm.forward import check_supported


class VocabMismatch(ValueError):
    pass


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "mps":
        # MPS shares RAM with the OS. The default cap is 1.7x the recommended working set
        # (9 GB on an 8 GB machine), so an oversized run swaps until the whole Mac stalls.
        # At 1.0 it raises an out-of-memory error instead.
        torch.mps.set_per_process_memory_fraction(1.0)
    return device


def load_model(
    spec: dict[str, Any],
    cache_dir: str,
    device: torch.device,
    dtype: str | None = None,
    local_files_only: bool = False,
) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"],
        revision=spec["revision"],
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        dtype=getattr(torch, dtype or spec["dtype"]),
        # Not "eager": Llama's eager attention always computes softmax in fp32
        # (modeling_llama.py:208), which caps an fp64 reference at ~1e-5. SDPA keeps the
        # requested dtype. lm.forward shares only the torch kernel with HF's path, not the
        # mask, positions or cache logic under test.
        attn_implementation="sdpa",
    )
    pack(model, device)
    model.eval()
    check_supported(model)
    return model


def pack(model: PreTrainedModel, device: torch.device) -> None:
    """Move every parameter into one contiguous buffer on `device`. Moved one tensor at a
    time, the MPS allocator rounds each up to its own heap: the 1.7B's 3.19 GiB of bf16
    weights took 4.01 GiB of driver memory, against 3.26 GiB packed, on a machine where that
    difference is the line between running and swapping. Tied weights are one parameter
    and are packed once."""
    params = list(model.parameters())
    dtypes = {p.dtype for p in params}
    if len(dtypes) != 1:
        model.to(device)
        return
    flat = torch.empty(sum(p.numel() for p in params), dtype=dtypes.pop(), device=device)
    offset = 0
    for p in params:
        view = flat[offset : offset + p.numel()].view_as(p)
        view.copy_(p.data)
        p.data = view
        offset += p.numel()
    model.to(device)  # buffers (rotary frequencies)


def load_config(spec: dict[str, Any], cache_dir: str) -> PretrainedConfig:
    return AutoConfig.from_pretrained(spec["repo"], revision=spec["revision"], cache_dir=cache_dir)


def load_tokenizer(spec: dict[str, Any], cache_dir: str, local_files_only: bool = False) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        spec["repo"], revision=spec["revision"], cache_dir=cache_dir, local_files_only=local_files_only
    )


def check_shared_vocab(
    target: PretrainedConfig,
    draft: PretrainedConfig,
    target_tok: PreTrainedTokenizerBase,
    draft_tok: PreTrainedTokenizerBase,
) -> None:
    """Accept-reject compares p_target(x) and p_draft(x) index by index. If id x means a
    different string to each model, every ratio is meaningless and nothing crashes.
    Takes configs, not models, so it runs before any weights are loaded."""
    if target.vocab_size != draft.vocab_size:
        raise VocabMismatch(f"logit width: target {target.vocab_size}, draft {draft.vocab_size}")
    if target_tok.get_vocab() != draft_tok.get_vocab():
        raise VocabMismatch("tokenizer vocabularies differ")
    if target_tok.eos_token_id != draft_tok.eos_token_id:
        raise VocabMismatch(f"eos id: target {target_tok.eos_token_id}, draft {draft_tok.eos_token_id}")


def tiny_model(
    seed: int, vocab_size: int = 256, n_layers: int = 2, hidden: int = 64, dtype: torch.dtype = torch.float32
) -> LlamaForCausalLM:
    # 4 query heads over 2 KV heads, so grouped-query attention is exercised.
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        attn_implementation="sdpa",
    )
    torch.manual_seed(seed)
    model = LlamaForCausalLM(config).to(dtype).eval()
    check_supported(model)
    return model
