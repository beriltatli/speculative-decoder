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
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    ).to(device)
    model.eval()
    check_supported(model)
    return model


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
