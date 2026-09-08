from premove_itn import (
    Candidate,
    SpanKind,
    build_candidate_graph,
    target_is_reachable,
)


def test_build_candidate_graph_enumerates_all_token_spans() -> None:
    candidates = build_candidate_graph("my booking id is seven eight three")

    matching = [
        candidate
        for candidate in candidates
        if candidate.token_start == 4
        and candidate.token_end == 7
        and candidate.replacement == "783"
    ]
    assert len(matching) == 1
    assert matching[0].text == "seven eight three"
    assert SpanKind.DIGIT_SEQUENCE in matching[0].kinds


def test_build_candidate_graph_merges_equivalent_kind_derivations() -> None:
    candidates = build_candidate_graph("seven three")

    equivalent = [
        candidate
        for candidate in candidates
        if candidate.token_start == 0
        and candidate.token_end == 2
        and candidate.replacement == "73"
    ]
    assert equivalent == [
        Candidate(
            token_start=0,
            token_end=2,
            char_start=0,
            char_end=11,
            text="seven three",
            replacement="73",
            kinds=(
                SpanKind.DIGIT_SEQUENCE,
                SpanKind.CARDINAL,
                SpanKind.DECIMAL,
            ),
        )
    ]


def test_build_candidate_graph_keeps_distinct_realizer_options() -> None:
    candidates = build_candidate_graph("seven eighty eight")

    cardinal_replacements = {
        candidate.replacement
        for candidate in candidates
        if candidate.token_start == 0
        and candidate.token_end == 3
        and SpanKind.CARDINAL in candidate.kinds
    }
    assert cardinal_replacements == {"95", "788"}


def test_build_candidate_graph_retains_exact_span_text() -> None:
    candidates = build_candidate_graph("seven\t  three")

    assert any(
        candidate.token_start == 0
        and candidate.token_end == 2
        and candidate.text == "seven\t  three"
        for candidate in candidates
    )


def test_build_candidate_graph_separates_boundary_punctuation() -> None:
    candidates = build_candidate_graph("call at two thirty.")

    assert any(
        candidate.text == "two thirty"
        and candidate.replacement == "02:30"
        and SpanKind.TIME in candidate.kinds
        for candidate in candidates
    )


def test_build_candidate_graph_is_empty_without_tokens() -> None:
    assert build_candidate_graph(" \t\n") == ()


def test_build_candidate_graph_excludes_exact_no_op_candidates() -> None:
    assert all(
        candidate.replacement != candidate.text
        for candidate in build_candidate_graph("123")
    )


def test_build_candidate_graph_retains_character_offsets() -> None:
    candidates = build_candidate_graph("id: seven three")

    candidate = next(
        candidate
        for candidate in candidates
        if candidate.text == "seven three" and candidate.replacement == "73"
    )
    assert (candidate.token_start, candidate.token_end) == (2, 4)
    assert (candidate.char_start, candidate.char_end) == (4, 15)


def test_build_candidate_graph_has_stable_sorted_order() -> None:
    candidates = build_candidate_graph("seven eighty eight")
    keys = [
        (candidate.token_start, candidate.token_end, candidate.replacement)
        for candidate in candidates
    ]
    assert keys == sorted(keys)


def test_target_is_reachable_combines_candidates_and_keep_regions() -> None:
    assert target_is_reachable(
        "my booking id is seven eight three",
        "my booking id is 783",
    )


def test_target_is_reachable_combines_non_overlapping_candidates() -> None:
    assert target_is_reachable(
        "meet at four thirty on march fourth",
        "meet at 04:30 on march 4th",
    )


def test_target_is_reachable_accepts_an_unchanged_sentence() -> None:
    assert target_is_reachable("give me a second", "give me a second")


def test_target_is_reachable_keeps_boundary_punctuation() -> None:
    assert target_is_reachable("call me at two thirty.", "call me at 02:30.")


def test_target_is_reachable_rejects_unavailable_output() -> None:
    assert not target_is_reachable("seven three", "74")


def test_target_is_reachable_does_not_depend_on_python_recursion_depth() -> None:
    text = "a" * 1_500

    assert target_is_reachable(text, text)
