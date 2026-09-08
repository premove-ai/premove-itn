import json
from types import SimpleNamespace

import pytest

import premove_itn
from premove_itn import (
    PremoveITN,
    SpanKind,
    realize,
    realize_options,
    representations_equivalent,
)


def test_package_exports_the_stable_public_api() -> None:
    assert premove_itn.__all__ == [
        "AlignmentState",
        "Candidate",
        "CandidateTransition",
        "GoldGraph",
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


def _write_release_provenance(path) -> None:
    (path / "provenance.json").write_text(
        json.dumps(
            {
                "artifact_version": "v0.1.0",
                "base_model": "microsoft/deberta-v3-large",
                "base_model_revision": (
                    "64a8c8eab3e352a784c658aef62be1662607476f"
                ),
                "hub_repository": "premove-itn/premove-itn",
                "hub_revision": "v0.1.0",
            }
        )
    )


def test_from_pretrained_loads_one_cached_release(tmp_path, monkeypatch) -> None:
    _write_release_provenance(tmp_path)
    loaded = SimpleNamespace(model=object(), tokenizer=object())
    calls = []

    def fake_load(path, *, device):
        calls.append((path, device))
        return loaded

    monkeypatch.setattr(
        "premove_itn.inference_artifact.load_inference_artifact", fake_load
    )

    first = PremoveITN.from_pretrained(tmp_path, device="cpu")
    second = PremoveITN.from_pretrained(tmp_path, device="cpu")

    assert first.model is loaded.model
    assert first.tokenizer is loaded.tokenizer
    assert second.model is loaded.model
    assert len(calls) == 2
    assert all(call[0] == tmp_path for call in calls)
    assert all(call[1] == "cpu" for call in calls)


def test_from_pretrained_rejects_another_release(tmp_path) -> None:
    _write_release_provenance(tmp_path)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    provenance["hub_revision"] = "v0.2.0"
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))

    with pytest.raises(RuntimeError, match="Hub revision mismatch"):
        PremoveITN.from_pretrained(tmp_path, device="cpu")


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
        revision="v0.1.0",
    )
    candidate = object()
    encoded = object()
    decoded = SimpleNamespace(text="order DLT2982")

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


def test_normalize_preserves_empty_and_whitespace_input() -> None:
    itn = object.__new__(PremoveITN)
    assert itn.normalize("") == ""
    assert itn.normalize("   \n") == "   \n"


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
