from spec.ngram import lookup


def test_copies_continuation_of_matching_suffix() -> None:
    assert lookup([1, 2, 3, 4, 1, 2], k=2, max_n=3) == [3, 4]


def test_extends_past_end_by_copying_its_own_output() -> None:
    # Suffix [1, 2] last occurred at 0; what followed was [3, 1, 2], then the copy continues.
    assert lookup([1, 2, 3, 1, 2], k=5, max_n=2) == [3, 1, 2, 3, 1]


def test_prefers_longest_then_most_recent_match() -> None:
    context = [7, 8, 9, 1, 8, 5, 7, 8]
    # [7, 8] occurs at 0 (followed by 9); [8] alone occurs most recently at 4 (followed by 5).
    assert lookup(context, k=1, max_n=2) == [9]
    assert lookup(context, k=1, max_n=1) == [5]
    assert lookup([1, 4, 2, 4, 3, 4], k=1, max_n=1) == [3]


def test_no_match_repeats_last_token() -> None:
    assert lookup([1, 2, 3], k=3, max_n=3) == [3, 3, 3]
