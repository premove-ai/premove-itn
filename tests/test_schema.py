from premove_itn.schema import Example, SemanticClass, Span


def test_example_serializes_semantic_class_and_original_offsets() -> None:
    example = Example(
        text="meet me at four thirty",
        tokens=("meet", "me", "at", "four", "thirty"),
        labels=("O", "O", "O", "B-TIME", "I-TIME"),
        spans=(Span(start=11, end=22, kind=SemanticClass.TIME, value="04:30"),),
    )

    record = example.to_dict()

    assert example.text[record["spans"][0]["start"] : record["spans"][0]["end"]] == (
        "four thirty"
    )
    assert record["spans"][0]["kind"] == "TIME"
