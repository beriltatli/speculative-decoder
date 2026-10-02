import torch


def next_token(logits: torch.Tensor, temperature: float, generator: torch.Generator | None) -> int:
    """logits: [V]. Temperature 0 is greedy; torch.argmax returns the first maximum, so ties
    break toward the lower id identically in every decoder that calls this."""
    if temperature == 0.0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


def probabilities(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """[..., V] logits -> float32 probabilities. Temperature 0 gives the one-hot of the argmax,
    which turns accept-reject into exact greedy verification."""
    if temperature == 0.0:
        return torch.nn.functional.one_hot(logits.argmax(dim=-1), logits.shape[-1]).float()
    return torch.softmax(logits.float() / temperature, dim=-1)
