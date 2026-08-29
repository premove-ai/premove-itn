import premove_itn.candidates as candidate_module
from premove_itn import (
    AlignmentState,
    Candidate,
    SpanKind,
    build_candidate_graph,
    build_gold_graph,
)


def test_build_gold_graph_recovers_one_complete_edit() -> None:
    graph = build_gold_graph("booking id seven three", "booking id 73")

    assert graph is not None
    assert [
        (edge.candidate.text, edge.candidate.replacement)
        for edge in graph.candidate_transitions
    ] == [("seven three", "73")]


def test_build_gold_graph_recovers_multiple_separate_edits() -> None:
    graph = build_gold_graph(
        "meet at four thirty on march fourth",
        "meet at 04:30 on march 4th",
    )

    assert graph is not None
    edits = {
        (edge.candidate.text, edge.candidate.replacement)
        for edge in graph.candidate_transitions
    }
    assert ("four thirty", "04:30") in edits
    assert ("fourth", "4th") in edits


def test_build_gold_graph_keeps_unchanged_text_implicit() -> None:
    graph = build_gold_graph("give me a second", "give me a second")

    assert graph is not None
    assert graph.candidate_transitions == ()


def test_build_gold_graph_excludes_wrong_candidates() -> None:
    graph = build_gold_graph("seven three", "73")

    assert graph is not None
    assert all(
        edge.candidate.replacement != "10"
        for edge in graph.candidate_transitions
    )


def test_build_gold_graph_removes_forward_valid_dead_end() -> None:
    candidates = build_candidate_graph("seven three")
    partial = next(
        candidate
        for candidate in candidates
        if candidate.text == "seven" and candidate.replacement == "7"
    )

    graph = build_gold_graph("seven three", "73")

    assert graph is not None
    assert partial not in {
        edge.candidate for edge in graph.candidate_transitions
    }
    assert any(
        edge.candidate.text == "seven three"
        and edge.candidate.replacement == "73"
        for edge in graph.candidate_transitions
    )


def test_build_gold_graph_retains_multiple_complete_derivations(monkeypatch) -> None:
    whole = Candidate(0, 2, 0, 2, "ab", "xy", (SpanKind.WORD,))
    first = Candidate(0, 1, 0, 1, "a", "x", (SpanKind.WORD,))
    second = Candidate(1, 2, 1, 2, "b", "y", (SpanKind.WORD,))
    monkeypatch.setattr(
        candidate_module,
        "build_candidate_graph",
        lambda text: (whole, first, second),
    )

    graph = build_gold_graph("ab", "xy")

    assert graph is not None
    assert {edge.candidate for edge in graph.candidate_transitions} == {
        whole,
        first,
        second,
    }
    assert AlignmentState(1, 1) in graph.states


def test_build_gold_graph_returns_none_for_unreachable_target() -> None:
    assert build_gold_graph("seven three", "74") is None
