from types import SimpleNamespace

import pytest

from lm.load import VocabMismatch, check_shared_vocab, tiny_model


def tok(vocab: dict[str, int], eos: int) -> SimpleNamespace:
    return SimpleNamespace(get_vocab=lambda: vocab, eos_token_id=eos)


def test_logit_width_mismatch_raises() -> None:
    with pytest.raises(VocabMismatch, match="logit width"):
        check_shared_vocab(tiny_model(0, vocab_size=256).config, tiny_model(0, vocab_size=300).config, tok({}, 0), tok({}, 0))


def test_token_mapping_mismatch_raises() -> None:
    config = tiny_model(0).config
    with pytest.raises(VocabMismatch, match="vocabularies"):
        check_shared_vocab(config, config, tok({"a": 0, "b": 1}, 0), tok({"a": 1, "b": 0}, 0))


def test_matching_pair_passes() -> None:
    config = tiny_model(0).config
    check_shared_vocab(config, config, tok({"a": 0}, 0), tok({"a": 0}, 0))
