import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from premove_itn import Candidate, SpanKind
from premove_itn.candidate_scorer import (
    CandidateScorer,
    collate_candidate_batch,
    kind_multihot,
    load_candidate_scorer,
    pool_candidate_spans,
    pool_replacement_tokens,
)
from premove_itn.model_inputs import EncodedCandidates


def _candidate(
    text: str,
    replacement: str,
    kinds: tuple[SpanKind, ...],
) -> Candidate:
    return Candidate(0, 1, 0, len(text), text, replacement, kinds)


class ExampleEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 2})()
        self.embedding = nn.Embedding(32, 2)
        self.calls = 0

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        self.calls += 1
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        return type("Output", (), {"last_hidden_state": hidden})()


def test_kind_multihot_uses_span_kind_order_and_preserves_all_kinds() -> None:
    candidate = Candidate(
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

    features = kind_multihot((candidate,))

    assert features.tolist() == [[1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]


def test_collate_candidate_batch_pads_sentences_and_flattens_candidates() -> None:
    first_candidates = (
        _candidate("seven three", "73", (SpanKind.DIGIT_SEQUENCE,)),
        _candidate("three", "3", (SpanKind.CARDINAL,)),
    )
    second_candidates = (_candidate("four thirty", "04:30", (SpanKind.TIME,)),)
    first = EncodedCandidates(
        input_ids=(1, 11, 12, 2),
        attention_mask=(1, 1, 1, 1),
        candidate_token_spans=((1, 3), (2, 3)),
        candidate_replacement_ids=((7, 3), (3,)),
    )
    second = EncodedCandidates(
        input_ids=(1, 21, 2),
        attention_mask=(1, 1, 1),
        candidate_token_spans=((1, 2),),
        candidate_replacement_ids=((4, 30),),
    )

    batch = collate_candidate_batch(
        ((first, first_candidates), (second, second_candidates)),
        pad_token_id=0,
    )

    assert batch.input_ids.tolist() == [[1, 11, 12, 2], [1, 21, 2, 0]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert batch.candidate_sentence_indices.tolist() == [0, 0, 1]
    assert batch.candidate_token_spans.tolist() == [[1, 3], [2, 3], [1, 2]]
    assert batch.candidate_offsets == (0, 2, 3)
    assert batch.candidate_replacement_ids.tolist() == [
        [7, 3],
        [3, 0],
        [4, 30],
    ]
    assert batch.candidate_replacement_mask.tolist() == [
        [1, 1],
        [1, 0],
        [1, 1],
    ]
    assert batch.candidate_kind_features.tolist() == [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]


def test_collate_candidate_batch_distinguishes_replacement_token_ids() -> None:
    candidates = (
        _candidate("seven eighty eight", "95", (SpanKind.CARDINAL,)),
        _candidate("seven eighty eight", "788", (SpanKind.CARDINAL,)),
    )
    encoded = EncodedCandidates(
        input_ids=(1, 11, 12, 13, 2),
        attention_mask=(1, 1, 1, 1, 1),
        candidate_token_spans=((1, 4), (1, 4)),
        candidate_replacement_ids=((9, 5), (7, 8, 8)),
    )

    batch = collate_candidate_batch(((encoded, candidates),), pad_token_id=0)

    assert batch.candidate_replacement_ids.tolist() == [
        [9, 5, 0],
        [7, 8, 8],
    ]
    assert batch.candidate_replacement_mask.tolist() == [
        [1, 1, 0],
        [1, 1, 1],
    ]


def test_candidate_batch_moves_all_tensors_and_preserves_offsets() -> None:
    candidate = _candidate("seven", "7", (SpanKind.CARDINAL,))
    encoded = EncodedCandidates(
        input_ids=(1, 11, 2),
        attention_mask=(1, 1, 1),
        candidate_token_spans=((1, 2),),
        candidate_replacement_ids=((7,),),
    )
    batch = collate_candidate_batch(((encoded, (candidate,)),), pad_token_id=0)

    moved = batch.to(torch.device("cpu"))

    assert moved.candidate_offsets == (0, 1)
    assert all(
        tensor.device.type == "cpu"
        for tensor in (
            moved.input_ids,
            moved.attention_mask,
            moved.candidate_sentence_indices,
            moved.candidate_token_spans,
            moved.candidate_kind_features,
            moved.candidate_replacement_ids,
            moved.candidate_replacement_mask,
        )
    )


def test_pool_candidate_spans_concatenates_start_end_and_mean() -> None:
    token_embeddings = torch.tensor(
        [
            [[100.0, 100.0], [1.0, 10.0], [3.0, 20.0], [999.0, 999.0]],
            [[200.0, 200.0], [5.0, 30.0], [7.0, 40.0], [9.0, 50.0]],
        ]
    )
    sentence_indices = torch.tensor([0, 1])
    token_spans = torch.tensor([[1, 3], [2, 3]])

    representations = pool_candidate_spans(
        token_embeddings, sentence_indices, token_spans
    )

    assert representations.tolist() == [
        [1.0, 10.0, 3.0, 20.0, 2.0, 15.0],
        [7.0, 40.0, 7.0, 40.0, 7.0, 40.0],
    ]


def test_candidate_scorer_encodes_batch_once_and_scores_each_candidate() -> None:
    candidates = (
        _candidate("seven", "7", (SpanKind.CARDINAL,)),
        _candidate("three", "3", (SpanKind.DIGIT_SEQUENCE,)),
    )
    encoded = EncodedCandidates(
        input_ids=(1, 11, 12, 2),
        attention_mask=(1, 1, 1, 1),
        candidate_token_spans=((1, 2), (2, 3)),
        candidate_replacement_ids=((7,), (3,)),
    )
    batch = collate_candidate_batch(((encoded, candidates),), pad_token_id=0)
    encoder = ExampleEncoder()
    scorer = CandidateScorer(
        encoder,
        kind_embedding_size=3,
        scorer_hidden_size=5,
        dropout=0.0,
    )

    scores = scorer(batch)

    assert scores.shape == (2,)
    assert encoder.calls == 1


def test_candidate_scorer_distinguishes_same_span_same_kind_replacements() -> None:
    candidates = (
        _candidate("seven eighty eight", "95", (SpanKind.CARDINAL,)),
        _candidate("seven eighty eight", "788", (SpanKind.CARDINAL,)),
    )
    encoded = EncodedCandidates(
        input_ids=(1, 11, 12, 13, 2),
        attention_mask=(1, 1, 1, 1, 1),
        candidate_token_spans=((1, 4), (1, 4)),
        candidate_replacement_ids=((9, 5), (7, 8, 8)),
    )
    batch = collate_candidate_batch(((encoded, candidates),), pad_token_id=0)
    encoder = ExampleEncoder()
    scorer = CandidateScorer(
        encoder,
        kind_embedding_size=1,
        scorer_hidden_size=1,
        dropout=0.0,
    )
    with torch.no_grad():
        encoder.embedding.weight.zero_()
        encoder.embedding.weight[9] = torch.tensor([1.0, 0.0])
        encoder.embedding.weight[5] = torch.tensor([1.0, 0.0])
        encoder.embedding.weight[7] = torch.tensor([3.0, 0.0])
        encoder.embedding.weight[8] = torch.tensor([3.0, 0.0])
        scorer.kind_projection.weight.zero_()
        scorer.scoring_head[0].weight.zero_()
        scorer.scoring_head[0].bias.zero_()
        scorer.scoring_head[0].weight[0, 7] = 1
        scorer.scoring_head[3].weight.fill_(1)
        scorer.scoring_head[3].bias.zero_()

    scores = scorer(batch)
    (scores[0] - scores[1]).backward()

    assert scores[0] != scores[1]
    assert encoder.calls == 1
    assert encoder.embedding.weight.grad[9].abs().sum() > 0
    assert encoder.embedding.weight.grad[7].abs().sum() > 0


def test_pool_replacement_tokens_preserves_token_order() -> None:
    token_embeddings = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[2.0, 20.0], [1.0, 10.0]],
        ]
    )
    token_mask = torch.ones((2, 2), dtype=torch.long)

    representations = pool_replacement_tokens(token_embeddings, token_mask)

    assert representations.tolist() == [
        [1.0, 10.0, 2.0, 20.0, 1.5, 15.0],
        [2.0, 20.0, 1.0, 10.0, 1.5, 15.0],
    ]
    assert not torch.equal(representations[0], representations[1])


def test_candidate_scorer_supports_keep_only_sentences() -> None:
    keep_only = EncodedCandidates(
        input_ids=(1, 11, 12, 2),
        attention_mask=(1, 1, 1, 1),
        candidate_token_spans=(),
        candidate_replacement_ids=(),
    )
    candidate = _candidate("seven", "7", (SpanKind.CARDINAL,))
    with_candidate = EncodedCandidates(
        input_ids=(1, 13, 2),
        attention_mask=(1, 1, 1),
        candidate_token_spans=((1, 2),),
        candidate_replacement_ids=((7,),),
    )
    scorer = CandidateScorer(ExampleEncoder())

    mixed_batch = collate_candidate_batch(
        ((keep_only, ()), (with_candidate, (candidate,))),
        pad_token_id=0,
    )
    empty_batch = collate_candidate_batch(((keep_only, ()),), pad_token_id=0)

    assert mixed_batch.candidate_offsets == (0, 0, 1)
    assert scorer(mixed_batch).shape == (1,)
    assert empty_batch.candidate_offsets == (0, 0)
    assert scorer(empty_batch).shape == (0,)


def test_collate_candidate_batch_rejects_replacement_token_collision() -> None:
    candidates = (
        _candidate("seven eighty eight", "95", (SpanKind.CARDINAL,)),
        _candidate("seven eighty eight", "788", (SpanKind.CARDINAL,)),
    )
    encoded = EncodedCandidates(
        input_ids=(1, 11, 12, 13, 2),
        attention_mask=(1, 1, 1, 1, 1),
        candidate_token_spans=((1, 4), (1, 4)),
        candidate_replacement_ids=((99,), (99,)),
    )

    with pytest.raises(ValueError, match="indistinguishable candidates"):
        collate_candidate_batch(((encoded, candidates),), pad_token_id=0)


def test_candidate_scorer_rejects_span_that_includes_padding() -> None:
    candidate = _candidate("three", "3", (SpanKind.CARDINAL,))
    encoded = EncodedCandidates(
        input_ids=(1, 11, 2, 0),
        attention_mask=(1, 1, 1, 0),
        candidate_token_spans=((1, 4),),
        candidate_replacement_ids=((3,),),
    )
    batch = collate_candidate_batch(((encoded, (candidate,)),), pad_token_id=0)
    scorer = CandidateScorer(ExampleEncoder())

    with pytest.raises(ValueError, match="outside attended tokens"):
        scorer(batch)


def test_load_candidate_scorer_uses_pinned_deberta(monkeypatch) -> None:
    encoder = ExampleEncoder()
    loaded: dict[str, object] = {}

    def load_encoder(name: str, **options: object) -> nn.Module:
        loaded.update(name=name, **options)
        return encoder

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", load_encoder)

    scorer = load_candidate_scorer()

    assert scorer.encoder is encoder
    assert loaded == {
        "name": "microsoft/deberta-v3-large",
        "revision": "64a8c8eab3e352a784c658aef62be1662607476f",
        "dtype": torch.float32,
    }
