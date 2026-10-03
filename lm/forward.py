import sys

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from cache.kv import KVCache

SUPPORTED = {"llama", "qwen2"}


def check_supported(model: PreTrainedModel) -> None:
    config = model.config
    if config.model_type not in SUPPORTED:
        raise NotImplementedError(f"cached forward supports {sorted(SUPPORTED)}, got {config.model_type}")
    # A sliding window would need the mask below to also drop keys older than the window.
    if getattr(config, "use_sliding_window", False):
        raise NotImplementedError("sliding-window attention is not implemented")


@torch.no_grad()
def forward(
    model: PreTrainedModel,
    cache: KVCache,
    seq_ids: list[int],
    input_ids: torch.Tensor,
    n_new: list[int],
) -> torch.Tensor:
    """Run `input_ids` [B, T] (right-padded; row b has n_new[b] real tokens) through the
    model, appending to each sequence's cache. Returns logits [B, T, V]; rows past n_new[b]
    are padding and meaningless.

    The same call serves prefill (T = prompt length), decode (T = 1) and speculative
    verification (T = k + 1)."""
    rotary = sys.modules[type(model).__module__].apply_rotary_pos_emb
    slots = cache.append(seq_ids, n_new)
    b, t = input_ids.shape
    device = input_ids.device

    positions = slots.past[:, None] + torch.arange(t, device=device)
    keys = torch.arange(slots.read.shape[1], device=device)
    # Query at absolute position p may see key j iff j <= p (causal) and j < total (exists).
    # Padded queries have p >= total, so they still see at least key 0 and never produce NaN.
    allowed = (keys[None, None, :] <= positions[:, :, None]) & (keys[None, None, :] < slots.total[:, None, None])
    mask = allowed[:, None, :, :]

    core = model.model
    hidden = core.embed_tokens(input_ids)
    cos, sin = core.rotary_emb(hidden, positions)
    for i, layer in enumerate(core.layers):
        attn = layer.self_attn
        x = layer.input_layernorm(hidden)
        q = attn.q_proj(x).view(b, t, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(b, t, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(b, t, -1, attn.head_dim).transpose(1, 2)
        q, k = rotary(q, k, cos, sin)
        cache.write(i, slots, k, v)
        k_all, v_all = cache.gather(i, slots)
        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask, scale=attn.scaling, enable_gqa=True)
        hidden = hidden + attn.o_proj(out.transpose(1, 2).reshape(b, t, -1))
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return logits(model, core.norm(hidden))


def logits(model: PreTrainedModel, hidden: torch.Tensor) -> torch.Tensor:
    """The output projection in at least fp32. In bf16 the logit grid is 0.125 wide at
    |x| ~ 20, so near-tied candidates often round to the same value, and argmax then breaks
    the tie by whichever rounding a (k+1)-token verify and a 1-token decode happened to
    produce: on the 1.7B target that 7 of the 13 prompts that diverged did so at an exact tie. The
    fp32 copy of the weight is made once and kept on the model (~400 MB for the 1.7B)."""
    dtype = torch.promote_types(hidden.dtype, torch.float32)
    weight = model.lm_head.weight
    cached = getattr(model, "_lm_head_wide", None)
    if cached is None or cached.dtype != dtype or cached.device != weight.device:
        cached = weight.detach().to(dtype)
        model._lm_head_wide = cached
    return torch.nn.functional.linear(hidden.to(dtype), cached)
