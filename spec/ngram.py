import torch


class NGramDraft:
    """Prompt-lookup drafting (Saxena 2023): find the most recent earlier occurrence of the
    context's last n tokens and propose what followed it. No model, no cache.

    The proposal is deterministic, so q is the one-hot of the proposed token. Accept-reject
    then accepts with probability p(x) and resamples from p with x removed, which is still
    exact: the guarantee holds for any q the token was drawn from, including a point mass."""

    def __init__(self, vocab_size: int, max_n: int = 3) -> None:
        self.vocab_size = vocab_size
        self.max_n = max_n

    def prefill(self, seq_ids: list[int], prompts: list[list[int]]) -> None:
        pass

    def propose(
        self, seq_ids: list[int], contexts: list[list[int]], k: int, temperature: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        draft = torch.tensor([lookup(ctx, k, self.max_n) for ctx in contexts])
        return draft, torch.nn.functional.one_hot(draft, self.vocab_size).float()

    def rollback(self, seq_id: int, keep: int) -> None:
        pass

    def release(self, seq_id: int) -> None:
        pass


def lookup(context: list[int], k: int, max_n: int) -> list[int]:
    for n in range(min(max_n, len(context) - 1), 0, -1):
        suffix = context[-n:]
        # Most recent match first: recent text predicts the continuation better than old text.
        for start in range(len(context) - n - 1, -1, -1):
            if context[start:start + n] == suffix:
                source = context[start + n:]
                out: list[int] = []
                # Copy forward, letting the copy read its own output once it runs past the
                # end of the context, so a match near the end still yields k tokens.
                for i in range(k):
                    out.append(source[i] if i < len(source) else out[i - len(source)])
                return out
    return [context[-1]] * k
