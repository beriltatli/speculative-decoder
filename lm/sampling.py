import torch


def next_token(logits: torch.Tensor, temperature: float, generator: torch.Generator | None) -> int:
    """logits: [V]. Temperature 0 is greedy; torch.argmax returns the first maximum, so ties
    break toward the lower id identically in every decoder that calls this."""
    if temperature == 0.0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
