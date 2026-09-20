import json
from datetime import datetime
from types import SimpleNamespace

import pytest

import premove_itn
from premove_itn import (
    Candidate,
    DateOrder,
    NormalizationContext,
    NormalizationResult,
    NormalizedSpan,
    PremoveITN,
    SpanKind,
    realize,
    realize_options,
    representations_equivalent,
)
from premove_itn.contextual import (
    DEFAULT_MODEL_ID,
    DEFAULT_RELEASE,
    DEFAULT_REVISION,
    EXPECTED_ARTIFACT_SHA256,
)


def test_package_exports_the_stable_public_api() -> None:
    assert premove_itn.__all__ == [
        "AlignmentState",
        "Candidate",
        "CandidateTransition",
        "DateOrder",
        "GoldGraph",
        "NormalizationContext",
        "NormalizationResult",
        "NormalizedSpan",
        "PremoveITN",
        "SpanKind",
        "build_candidate_graph",
        "build_gold_graph",
        "normalize_sentence",
        "realize",
        "realize_options",
        "representations_equivalent",
        "target_is_reachable",
        "tn_normalize",
    ]


def test_public_contextual_api_is_exported() -> None:
    assert premove_itn.PremoveITN is PremoveITN
    assert premove_itn.DateOrder is DateOrder
    assert premove_itn.NormalizationContext is NormalizationContext
    assert premove_itn.NormalizationResult is NormalizationResult
    assert premove_itn.NormalizedSpan is NormalizedSpan


def test_date_order_covers_every_day_month_year_permutation() -> None:
    assert tuple(DateOrder) == (
        DateOrder.DMY,
        DateOrder.DYM,
        DateOrder.MDY,
        DateOrder.MYD,
        DateOrder.YDM,
        DateOrder.YMD,
    )


def _write_release_provenance(path) -> None:
    (path / "provenance.json").write_text(
        json.dumps(
            {
                "artifact_version": DEFAULT_RELEASE,
                "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
                "base_model": "microsoft/deberta-v3-large",
                "base_model_revision": ("64a8c8eab3e352a784c658aef62be1662607476f"),
                "hub_repository": "premove-itn/premove-itn-contextual",
                "hub_revision": DEFAULT_RELEASE,
            }
        )
    )


def test_from_pretrained_loads_local_release(tmp_path, monkeypatch) -> None:
    _write_release_provenance(tmp_path)
    loaded = SimpleNamespace(model=object(), tokenizer=object())
    calls = []

    def fake_load(path, *, device):
        calls.append((path, device))
        return loaded

    monkeypatch.setattr(
        "premove_itn.inference_artifact.load_inference_artifact", fake_load
    )

    context = NormalizationContext(
        reference_datetime=datetime(2026, 9, 20, 12),
        timezone="Asia/Kolkata",
        locale="en-IN",
        date_order=DateOrder.DMY,
    )
    first = PremoveITN.from_pretrained(tmp_path, device="cpu", context=context)
    second = PremoveITN.from_pretrained(tmp_path, device="cpu")

    assert first.model is loaded.model
    assert first.tokenizer is loaded.tokenizer
    assert first.context is context
    assert second.model is loaded.model
    assert second.context is None
    assert len(calls) == 2
    assert all(call[0] == tmp_path for call in calls)
    assert all(call[1] == "cpu" for call in calls)


def test_from_pretrained_rejects_another_release(tmp_path) -> None:
    _write_release_provenance(tmp_path)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    provenance["hub_revision"] = "v9.9.9"
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))

    with pytest.raises(RuntimeError, match="Hub revision mismatch"):
        PremoveITN.from_pretrained(tmp_path, device="cpu")


def test_from_pretrained_rejects_another_model_file(tmp_path) -> None:
    _write_release_provenance(tmp_path)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    provenance["artifact_sha256"] = "changed"
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))

    with pytest.raises(RuntimeError, match="model-file digest mismatch"):
        PremoveITN.from_pretrained(tmp_path, device="cpu")


def test_hub_tag_must_resolve_to_the_frozen_commit(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshots" / DEFAULT_REVISION
    snapshot.mkdir(parents=True)
    _write_release_provenance(snapshot)
    loaded = SimpleNamespace(model=object(), tokenizer=object())
    requested = []

    def fake_snapshot_download(*, repo_id, revision):
        requested.append((repo_id, revision))
        return str(snapshot)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )
    monkeypatch.setattr(
        "premove_itn.inference_artifact.load_inference_artifact",
        lambda path, *, device: loaded,
    )

    itn = PremoveITN.from_pretrained(
        DEFAULT_MODEL_ID,
        revision=DEFAULT_RELEASE,
        device="cpu",
    )

    assert itn.model is loaded.model
    assert itn.revision == DEFAULT_REVISION
    assert requested == [(DEFAULT_MODEL_ID, DEFAULT_RELEASE)]


def test_hub_tag_rejects_a_moved_commit(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshots" / ("0" * 40)
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **options: str(snapshot),
    )

    with pytest.raises(RuntimeError, match="unexpected commit"):
        PremoveITN.from_pretrained(
            DEFAULT_MODEL_ID,
            revision=DEFAULT_RELEASE,
            device="cpu",
        )


def test_from_pretrained_rejects_another_hub_repository() -> None:
    with pytest.raises(ValueError, match="unsupported Hugging Face model"):
        PremoveITN.from_pretrained(
            "another/model",
            device="cpu",
        )


def test_from_pretrained_rejects_invalid_device(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        PremoveITN.from_pretrained(tmp_path, device="tpu")


def test_normalize_reuses_the_loaded_model(monkeypatch) -> None:
    class FakeTokenizer:
        pad_token_id = 0

    class FakeBatch:
        def to(self, device):
            assert str(device) == "cpu"
            return self

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            return batch

    model = FakeModel()
    tokenizer = FakeTokenizer()
    itn = PremoveITN(
        model=model,
        tokenizer=tokenizer,
        torch_module=__import__("torch"),
        device=__import__("torch").device("cpu"),
        model_id="local",
        revision=DEFAULT_RELEASE,
    )
    candidate = Candidate(
        token_start=1,
        token_end=8,
        char_start=6,
        char_end=31,
        text="d l t two nine eight two",
        replacement="DLT2982",
        kinds=(SpanKind.WORD,),
    )
    encoded = object()
    rendered_span = SimpleNamespace(
        candidate=candidate,
        normalized_start=6,
        normalized_end=13,
    )
    decoded = SimpleNamespace(
        text="order DLT2982",
        rendered_spans=(rendered_span,),
    )

    monkeypatch.setattr(
        "premove_itn.candidates.build_candidate_graph",
        lambda text: (candidate,),
    )
    monkeypatch.setattr(
        "premove_itn.model_inputs.encode_candidates",
        lambda text, candidates, tokenizer: encoded,
    )
    monkeypatch.setattr(
        "premove_itn.candidate_scorer.collate_candidate_batch",
        lambda examples, *, pad_token_id: FakeBatch(),
    )
    monkeypatch.setattr(
        "premove_itn.decoder.decode_candidates",
        lambda text, candidates, scores: decoded,
    )

    assert itn.normalize("order d l t two nine eight two") == "order DLT2982"
    assert itn.normalize("order d l t two nine eight two") == "order DLT2982"
    assert model.calls == 2


def test_normalize_structured_returns_all_selected_span_metadata(monkeypatch) -> None:
    class FakeTokenizer:
        pad_token_id = 0

    class FakeBatch:
        def to(self, device):
            return self

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            return batch

    candidate = Candidate(
        token_start=1,
        token_end=3,
        char_start=4,
        char_end=18,
        text="twenty dollars",
        replacement="$20",
        kinds=(SpanKind.MONEY,),
    )
    decoded = SimpleNamespace(
        text="pay $20",
        rendered_spans=(
            SimpleNamespace(
                candidate=candidate,
                normalized_start=4,
                normalized_end=7,
            ),
        ),
    )
    model = FakeModel()
    itn = PremoveITN(
        model=model,
        tokenizer=FakeTokenizer(),
        torch_module=__import__("torch"),
        device=__import__("torch").device("cpu"),
        model_id="local",
        revision=DEFAULT_RELEASE,
    )
    monkeypatch.setattr(
        "premove_itn.candidates.build_candidate_graph",
        lambda text: (candidate,),
    )
    monkeypatch.setattr(
        "premove_itn.model_inputs.encode_candidates",
        lambda text, candidates, tokenizer: object(),
    )
    monkeypatch.setattr(
        "premove_itn.candidate_scorer.collate_candidate_batch",
        lambda examples, *, pad_token_id: FakeBatch(),
    )
    monkeypatch.setattr(
        "premove_itn.decoder.decode_candidates",
        lambda text, candidates, scores: decoded,
    )

    result = itn.normalize_structured("pay twenty dollars")

    assert result == NormalizationResult(
        text="pay $20",
        resolved_text="pay $20",
        spans=(
            NormalizedSpan(
                source_start=4,
                source_end=18,
                normalized_start=4,
                normalized_end=7,
                source_text="twenty dollars",
                normalized_text="$20",
                kinds=(SpanKind.MONEY,),
                resolved_value=None,
            ),
        ),
    )
    assert model.calls == 1


def test_three_public_views_each_use_instance_context(monkeypatch) -> None:
    default_context = NormalizationContext(
        reference_datetime=datetime(2026, 9, 20, 12),
    )
    itn = object.__new__(PremoveITN)
    itn.context = default_context
    result = NormalizationResult(
        text="tomorrow at 04:30",
        resolved_text="tomorrow at 04:30",
        spans=(),
    )
    calls = []

    def normalize_internal(self, text, *, context):
        calls.append((text, context))
        return result

    monkeypatch.setattr(PremoveITN, "_normalize_internal", normalize_internal)

    assert itn.normalize("source") == result.text
    assert itn.normalize_resolved("source") == result.resolved_text
    assert itn.normalize_structured("source") is result
    assert calls == [
        ("source", default_context),
        ("source", default_context),
        ("source", default_context),
    ]


@pytest.mark.parametrize(
    "method_name",
    ("normalize", "normalize_resolved", "normalize_structured"),
)
def test_call_context_replaces_the_complete_instance_context(
    monkeypatch,
    method_name,
) -> None:
    default_context = NormalizationContext(
        reference_datetime=datetime(2026, 9, 20, 12),
        timezone="Asia/Kolkata",
        locale="en-IN",
        date_order=DateOrder.DMY,
    )
    call_context = NormalizationContext(locale="en-US")
    itn = object.__new__(PremoveITN)
    itn.context = default_context
    received = []

    def normalize_internal(self, text, *, context):
        received.append(context)
        return NormalizationResult(text=text, resolved_text=text, spans=())

    monkeypatch.setattr(PremoveITN, "_normalize_internal", normalize_internal)

    getattr(itn, method_name)("source", context=call_context)

    assert received == [call_context]
    assert received[0].reference_datetime is None
    assert received[0].timezone is None
    assert received[0].locale == "en-US"
    assert received[0].date_order is None


def test_empty_call_context_clears_instance_context(monkeypatch) -> None:
    itn = object.__new__(PremoveITN)
    itn.context = NormalizationContext(
        reference_datetime=datetime(2026, 9, 20, 12),
        timezone="Asia/Kolkata",
        locale="en-IN",
        date_order=DateOrder.DMY,
    )
    empty_context = NormalizationContext()
    received = []

    def normalize_internal(self, text, *, context):
        received.append(context)
        return NormalizationResult(text=text, resolved_text=text, spans=())

    monkeypatch.setattr(PremoveITN, "_normalize_internal", normalize_internal)

    itn.normalize_structured("source", context=empty_context)

    assert received == [empty_context]
    assert received[0] is empty_context


def test_context_rejects_invalid_fields_and_public_arguments() -> None:
    with pytest.raises(TypeError, match="reference_datetime"):
        NormalizationContext(reference_datetime="2026-09-20")
    with pytest.raises(ValueError, match="timezone"):
        NormalizationContext(timezone="")
    with pytest.raises(ValueError, match="timezone"):
        NormalizationContext(timezone="   ")
    with pytest.raises(TypeError, match="locale"):
        NormalizationContext(locale=1)
    with pytest.raises(ValueError, match="locale"):
        NormalizationContext(locale="  ")
    with pytest.raises(TypeError, match="date_order"):
        NormalizationContext(date_order="DMY")

    itn = object.__new__(PremoveITN)
    itn.context = None
    with pytest.raises(TypeError, match="NormalizationContext"):
        itn.normalize("source", context={})


def test_normalize_preserves_empty_and_whitespace_input() -> None:
    itn = object.__new__(PremoveITN)
    itn.context = None
    assert itn.normalize("") == ""
    assert itn.normalize("   \n") == "   \n"


def test_structured_views_preserve_identity_input() -> None:
    itn = object.__new__(PremoveITN)
    itn.context = None

    assert itn.normalize_structured("") == NormalizationResult("", "", ())
    assert itn.normalize_structured(" \n") == NormalizationResult(" \n", " \n", ())
    assert itn.normalize_resolved("") == ""


def test_structured_view_preserves_text_without_candidates(monkeypatch) -> None:
    itn = object.__new__(PremoveITN)
    itn.context = None
    monkeypatch.setattr(
        "premove_itn.candidates.build_candidate_graph",
        lambda text: (),
    )

    assert itn.normalize_structured("leave unchanged") == NormalizationResult(
        text="leave unchanged",
        resolved_text="leave unchanged",
        spans=(),
    )


def test_relative_date_annotation_runs_without_neural_candidates(monkeypatch) -> None:
    itn = object.__new__(PremoveITN)
    itn.context = None
    monkeypatch.setattr(
        "premove_itn.candidates.build_candidate_graph",
        lambda text: (),
    )

    result = itn.normalize_structured(
        "tomorrow",
        context=NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert result.text == "tomorrow"
    assert result.resolved_text == "2026-09-20"
    assert result.spans[0].kinds == (SpanKind.DATE,)
    assert result.spans[0].resolved_value == "2026-09-20"


def test_relative_offset_annotation_runs_without_neural_candidates(monkeypatch) -> None:
    itn = object.__new__(PremoveITN)
    itn.context = None
    monkeypatch.setattr(
        "premove_itn.candidates.build_candidate_graph",
        lambda text: (),
    )

    result = itn.normalize_structured(
        "in two days",
        context=NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert result.text == "in two days"
    assert result.resolved_text == "2026-09-21"
    assert result.spans[0].source_text == "in two days"
    assert result.spans[0].kinds == (SpanKind.DATE,)
    assert result.spans[0].resolved_value == "2026-09-21"


def test_weekday_relative_annotation_runs_without_neural_candidates(
    monkeypatch,
) -> None:
    itn = object.__new__(PremoveITN)
    itn.context = None
    monkeypatch.setattr(
        "premove_itn.candidates.build_candidate_graph",
        lambda text: (),
    )

    result = itn.normalize_structured(
        "next Monday",
        context=NormalizationContext(reference_datetime=datetime(2026, 9, 23)),
    )

    assert result.text == "next Monday"
    assert result.resolved_text == "2026-09-28"
    assert result.spans[0].source_text == "next Monday"
    assert result.spans[0].kinds == (SpanKind.DATE,)
    assert result.spans[0].resolved_value == "2026-09-28"


def test_relative_offset_composes_with_selected_number_edit(monkeypatch) -> None:
    class FakeTokenizer:
        pad_token_id = 0

    class FakeBatch:
        def to(self, device):
            return self

    class FakeModel:
        def __call__(self, batch):
            return batch

    source = "in two days"
    candidate = Candidate(
        token_start=1,
        token_end=2,
        char_start=3,
        char_end=6,
        text="two",
        replacement="2",
        kinds=(SpanKind.CARDINAL,),
    )
    decoded = SimpleNamespace(
        text="in 2 days",
        rendered_spans=(
            SimpleNamespace(
                candidate=candidate,
                normalized_start=3,
                normalized_end=4,
            ),
        ),
    )
    itn = PremoveITN(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        torch_module=__import__("torch"),
        device=__import__("torch").device("cpu"),
        model_id="local",
        revision=DEFAULT_RELEASE,
    )
    monkeypatch.setattr(
        "premove_itn.candidates.build_candidate_graph",
        lambda text: (candidate,),
    )
    monkeypatch.setattr(
        "premove_itn.model_inputs.encode_candidates",
        lambda text, candidates, tokenizer: object(),
    )
    monkeypatch.setattr(
        "premove_itn.candidate_scorer.collate_candidate_batch",
        lambda examples, *, pad_token_id: FakeBatch(),
    )
    monkeypatch.setattr(
        "premove_itn.decoder.decode_candidates",
        lambda text, candidates, scores: decoded,
    )

    result = itn.normalize_structured(
        source,
        context=NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert result.text == "in 2 days"
    assert result.resolved_text == "2026-09-21"
    assert result.spans[0].kinds == (SpanKind.DATE,)
    assert result.spans[0].resolved_value == "2026-09-21"
    assert result.spans[1].kinds == (SpanKind.CARDINAL,)
    assert result.spans[1].normalized_text == "2"


def test_missing_year_date_enriches_selected_decoder_span(monkeypatch) -> None:
    class FakeTokenizer:
        pad_token_id = 0

    class FakeBatch:
        def to(self, device):
            return self

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            return batch

    source = "september thirtieth"
    candidate = Candidate(
        token_start=0,
        token_end=2,
        char_start=0,
        char_end=len(source),
        text=source,
        replacement="september 30",
        kinds=(SpanKind.DATE,),
    )
    decoded = SimpleNamespace(
        text="september 30",
        rendered_spans=(
            SimpleNamespace(
                candidate=candidate,
                normalized_start=0,
                normalized_end=len("september 30"),
            ),
        ),
    )
    model = FakeModel()
    itn = PremoveITN(
        model=model,
        tokenizer=FakeTokenizer(),
        torch_module=__import__("torch"),
        device=__import__("torch").device("cpu"),
        model_id="local",
        revision=DEFAULT_RELEASE,
    )
    monkeypatch.setattr(
        "premove_itn.candidates.build_candidate_graph",
        lambda text: (candidate,),
    )
    monkeypatch.setattr(
        "premove_itn.model_inputs.encode_candidates",
        lambda text, candidates, tokenizer: object(),
    )
    monkeypatch.setattr(
        "premove_itn.candidate_scorer.collate_candidate_batch",
        lambda examples, *, pad_token_id: FakeBatch(),
    )
    monkeypatch.setattr(
        "premove_itn.decoder.decode_candidates",
        lambda text, candidates, scores: decoded,
    )

    result = itn.normalize_structured(
        source,
        context=NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert result.text == "september 30"
    assert result.resolved_text == "2026-09-30"
    assert result.spans[0].resolved_value == "2026-09-30"
    assert model.calls == 1


def test_numeric_date_enriches_selected_decoder_span(monkeypatch) -> None:
    class FakeTokenizer:
        pad_token_id = 0

    class FakeBatch:
        def to(self, device):
            return self

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            return batch

    source = "03 slash 04 slash 2026"
    candidate = Candidate(
        token_start=0,
        token_end=5,
        char_start=0,
        char_end=len(source),
        text=source,
        replacement="03/04/2026",
        kinds=(SpanKind.DATE,),
    )
    decoded = SimpleNamespace(
        text="03/04/2026",
        rendered_spans=(
            SimpleNamespace(
                candidate=candidate,
                normalized_start=0,
                normalized_end=len("03/04/2026"),
            ),
        ),
    )
    model = FakeModel()
    itn = PremoveITN(
        model=model,
        tokenizer=FakeTokenizer(),
        torch_module=__import__("torch"),
        device=__import__("torch").device("cpu"),
        model_id="local",
        revision=DEFAULT_RELEASE,
    )
    monkeypatch.setattr(
        "premove_itn.candidates.build_candidate_graph",
        lambda text: (candidate,),
    )
    monkeypatch.setattr(
        "premove_itn.model_inputs.encode_candidates",
        lambda text, candidates, tokenizer: object(),
    )
    monkeypatch.setattr(
        "premove_itn.candidate_scorer.collate_candidate_batch",
        lambda examples, *, pad_token_id: FakeBatch(),
    )
    monkeypatch.setattr(
        "premove_itn.decoder.decode_candidates",
        lambda text, candidates, scores: decoded,
    )

    result = itn.normalize_structured(
        source,
        context=NormalizationContext(date_order=DateOrder.DMY),
    )

    assert result.text == "03/04/2026"
    assert result.resolved_text == "2026-04-03"
    assert result.spans[0].resolved_value == "2026-04-03"
    assert model.calls == 1


def test_public_api_composes_multiple_temporal_resolutions_in_one_pass(
    monkeypatch,
) -> None:
    class FakeTokenizer:
        pad_token_id = 0

    class FakeBatch:
        def to(self, device):
            return self

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            return batch

    source = "tomorrow september thirtieth 03 slash 04 slash 2026"
    named_candidate = Candidate(
        token_start=1,
        token_end=3,
        char_start=9,
        char_end=28,
        text="september thirtieth",
        replacement="september 30",
        kinds=(SpanKind.DATE,),
    )
    numeric_candidate = Candidate(
        token_start=4,
        token_end=9,
        char_start=29,
        char_end=len(source),
        text="03 slash 04 slash 2026",
        replacement="03/04/2026",
        kinds=(SpanKind.DATE,),
    )
    candidates = (named_candidate, numeric_candidate)
    decoded = SimpleNamespace(
        text="tomorrow september 30 03/04/2026",
        rendered_spans=(
            SimpleNamespace(
                candidate=named_candidate,
                normalized_start=9,
                normalized_end=21,
            ),
            SimpleNamespace(
                candidate=numeric_candidate,
                normalized_start=22,
                normalized_end=32,
            ),
        ),
    )
    model = FakeModel()
    itn = PremoveITN(
        model=model,
        tokenizer=FakeTokenizer(),
        torch_module=__import__("torch"),
        device=__import__("torch").device("cpu"),
        model_id="local",
        revision=DEFAULT_RELEASE,
    )
    monkeypatch.setattr(
        "premove_itn.candidates.build_candidate_graph",
        lambda text: candidates,
    )
    monkeypatch.setattr(
        "premove_itn.model_inputs.encode_candidates",
        lambda text, candidates, tokenizer: object(),
    )
    monkeypatch.setattr(
        "premove_itn.candidate_scorer.collate_candidate_batch",
        lambda examples, *, pad_token_id: FakeBatch(),
    )
    monkeypatch.setattr(
        "premove_itn.decoder.decode_candidates",
        lambda text, candidates, scores: decoded,
    )

    result = itn.normalize_structured(
        source,
        context=NormalizationContext(
            reference_datetime=datetime(2026, 9, 23),
            date_order=DateOrder.DMY,
        ),
    )

    assert result.text == decoded.text
    assert result.resolved_text == "2026-09-24 2026-09-30 2026-04-03"
    assert tuple(span.resolved_value for span in result.spans) == (
        "2026-09-24",
        "2026-09-30",
        "2026-04-03",
    )
    assert model.calls == 1


def test_realize_routes_an_explicit_kind_to_rust() -> None:
    assert realize(SpanKind.TIME, "four thirty") == "04:30"
    assert realize(SpanKind.DIGIT_SEQUENCE, "zero zero seven") == "007"


def test_realize_options_returns_only_semantic_cardinal_alternatives() -> None:
    assert realize_options(SpanKind.CARDINAL, "two") == ["2"]
    assert realize_options(SpanKind.CARDINAL, "seven eighty eight") == ["95", "788"]


def test_representations_equivalent_is_separate_from_realization() -> None:
    assert representations_equivalent(SpanKind.CARDINAL, "12345", "12,345")
    assert representations_equivalent(SpanKind.DATE, "4 march 2014", "2014-03-04")
    assert representations_equivalent(SpanKind.TIME, "04:30 p.m.", "4.30 PM")
    assert representations_equivalent(SpanKind.MONEY, "$1000000", "$1M")
    assert representations_equivalent(SpanKind.MONEY, "$5", "USD 5")
    assert not representations_equivalent(SpanKind.MONEY, "$5", "CAD 5")
    assert representations_equivalent(SpanKind.DECIMAL, "1,212.3", "1212.30")
    assert representations_equivalent(SpanKind.MEASUREMENT, "90%", "90 percent")
    assert representations_equivalent(SpanKind.ORDINAL, "VIII", "the eighth")
    assert representations_equivalent(SpanKind.PHONE, "3292-3297", "329-23297")
    assert not representations_equivalent(SpanKind.CARDINAL, "12345", "12346")
