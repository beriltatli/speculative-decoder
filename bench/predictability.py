"""Model-free predictability of text, to establish that the prompt categories differ as text
before claiming they differ in acceptance rate."""
import gzip


def gzip_ratio(text: str) -> float:
    """Compressed over raw UTF-8 bytes; lower means more redundant. gzip's ~20-byte header
    dominates short strings, so this is only meaningful on hundreds of bytes or more."""
    raw = text.encode()
    return len(gzip.compress(raw, compresslevel=9, mtime=0)) / len(raw)


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    """Fraction of whitespace-word n-grams that already occurred earlier in the text: the
    share of the text a prompt-lookup draft could in principle have copied."""
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    seen: set[tuple[str, ...]] = set()
    repeats = 0
    for gram in grams:
        repeats += gram in seen
        seen.add(gram)
    return repeats / len(grams) if grams else 0.0
