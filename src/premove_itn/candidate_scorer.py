"""Contextual scoring over deterministic Rust candidates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from premove_itn.candidates import Candidate
from premove_itn.labels import SPAN_KINDS, SpanKind
from premove_itn.model_inputs import MODEL_NAME, MODEL_REVISION, EncodedCandidates


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    """One padded sentence batch with flattened candidate metadata."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    candidate_sentence_indices: torch.Tensor
    candidate_token_spans: torch.Tensor
    candidate_kind_features: torch.Tensor
    candidate_replacement_ids: torch.Tensor
    candidate_replacement_mask: torch.Tensor
    candidate_offsets: tuple[int, ...]

    def to(self, device: torch.device | str) -> CandidateBatch:
        """Move every tensor to one training device."""
        return CandidateBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            candidate_sentence_indices=self.candidate_sentence_indices.to(device),
            candidate_token_spans=self.candidate_token_spans.to(device),
            candidate_kind_features=self.candidate_kind_features.to(device),
            candidate_replacement_ids=self.candidate_replacement_ids.to(device),
            candidate_replacement_mask=self.candidate_replacement_mask.to(device),
            candidate_offsets=self.candidate_offsets,
        )


class CandidateScorer(nn.Module):
    """Score all deterministic candidates after one encoder pass."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        kind_embedding_size: int = 32,
        scorer_hidden_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        encoder_hidden_size = int(encoder.config.hidden_size)
        self.kind_projection = nn.Linear(
            len(SPAN_KINDS), kind_embedding_size, bias=False
        )
        self.scoring_head = nn.Sequential(
            nn.Linear(
                encoder_hidden_size * 4 + kind_embedding_size,
                scorer_hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(scorer_hidden_size, 1),
        )

    def forward(self, batch: CandidateBatch) -> torch.Tensor:
        """Return one scalar per candidate in flattened batch order."""
        sentence_indices = batch.candidate_sentence_indices
        token_spans = batch.candidate_token_spans
        if sentence_indices.numel() and (
            torch.any(sentence_indices < 0)
            or torch.any(sentence_indices >= batch.input_ids.shape[0])
        ):
            raise ValueError("candidate sentence index is outside the batch")
        if token_spans.numel():
            attended_lengths = batch.attention_mask.sum(dim=1)
            starts = token_spans[:, 0]
            ends = token_spans[:, 1]
            if torch.any(starts < 0) or torch.any(
                ends > attended_lengths[sentence_indices]
            ):
                raise ValueError("candidate token span is outside attended tokens")

        encoder_output = self.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        span_features = pool_candidate_spans(
            encoder_output.last_hidden_state,
            batch.candidate_sentence_indices,
            batch.candidate_token_spans,
        )
        kind_features = self.kind_projection(batch.candidate_kind_features)
        replacement_mask = batch.candidate_replacement_mask
        if replacement_mask.numel() and torch.any(replacement_mask.sum(dim=1) == 0):
            raise ValueError("candidate replacement must contain a token")
        replacement_embeddings = self.encoder.get_input_embeddings()(
            batch.candidate_replacement_ids
        )
        replacement_mask = replacement_mask.to(replacement_embeddings.dtype).unsqueeze(
            2
        )
        replacement_features = (replacement_embeddings * replacement_mask).sum(
            dim=1
        ) / replacement_mask.sum(dim=1)
        features = torch.cat(
            (span_features, kind_features, replacement_features), dim=1
        )
        return self.scoring_head(features).squeeze(1)


def load_candidate_scorer() -> CandidateScorer:
    """Load the first experiment scorer with its pinned pretrained encoder."""
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    return CandidateScorer(encoder)


def kind_multihot(candidates: Sequence[Candidate]) -> torch.Tensor:
    """Encode candidate kinds in the stable ``SPAN_KINDS`` order."""
    kind_indices = {kind: index for index, kind in enumerate(SPAN_KINDS)}
    features = torch.zeros((len(candidates), len(SPAN_KINDS)), dtype=torch.float32)
    for row, candidate in enumerate(candidates):
        for kind in candidate.kinds:
            features[row, kind_indices[kind]] = 1
    return features


def collate_candidate_batch(
    examples: Sequence[tuple[EncodedCandidates, Sequence[Candidate]]],
    *,
    pad_token_id: int,
) -> CandidateBatch:
    """Pad sentences and flatten candidates in stable input order."""
    if not examples:
        raise ValueError("candidate batch must contain at least one sentence")

    max_tokens = max(len(encoded.input_ids) for encoded, _ in examples)
    input_ids = torch.full((len(examples), max_tokens), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_tokens), dtype=torch.long)
    sentence_indices: list[int] = []
    token_spans: list[tuple[int, int]] = []
    replacement_ids: list[tuple[int, ...]] = []
    candidates: list[Candidate] = []
    candidate_offsets = [0]

    for sentence_index, (encoded, sentence_candidates) in enumerate(examples):
        if len(encoded.input_ids) != len(encoded.attention_mask):
            raise ValueError("input IDs and attention mask must have equal length")
        if len(encoded.candidate_token_spans) != len(sentence_candidates):
            raise ValueError("candidate spans and candidates must have equal length")
        if len(encoded.candidate_replacement_ids) != len(sentence_candidates):
            raise ValueError(
                "replacement token IDs and candidates must have equal length"
            )

        replacements_by_features: dict[
            tuple[tuple[int, int], frozenset[SpanKind], tuple[int, ...]], str
        ] = {}
        for token_span, token_ids, candidate in zip(
            encoded.candidate_token_spans,
            encoded.candidate_replacement_ids,
            sentence_candidates,
            strict=True,
        ):
            if not token_ids:
                raise ValueError("candidate replacement must contain a token")
            feature_key = (token_span, frozenset(candidate.kinds), token_ids)
            previous = replacements_by_features.setdefault(
                feature_key, candidate.replacement
            )
            if previous != candidate.replacement:
                raise ValueError(
                    "indistinguishable candidates have different replacements: "
                    f"{previous!r} and {candidate.replacement!r}"
                )

        token_count = len(encoded.input_ids)
        input_ids[sentence_index, :token_count] = torch.tensor(encoded.input_ids)
        attention_mask[sentence_index, :token_count] = torch.tensor(
            encoded.attention_mask
        )
        sentence_indices.extend([sentence_index] * len(sentence_candidates))
        token_spans.extend(encoded.candidate_token_spans)
        replacement_ids.extend(encoded.candidate_replacement_ids)
        candidates.extend(sentence_candidates)
        candidate_offsets.append(len(candidates))

    candidate_token_spans = (
        torch.tensor(token_spans, dtype=torch.long)
        if token_spans
        else torch.empty((0, 2), dtype=torch.long)
    )
    max_replacement_tokens = max(map(len, replacement_ids), default=0)
    candidate_replacement_ids = torch.full(
        (len(replacement_ids), max_replacement_tokens),
        pad_token_id,
        dtype=torch.long,
    )
    candidate_replacement_mask = torch.zeros_like(candidate_replacement_ids)
    for row, token_ids in enumerate(replacement_ids):
        token_count = len(token_ids)
        candidate_replacement_ids[row, :token_count] = torch.tensor(token_ids)
        candidate_replacement_mask[row, :token_count] = 1
    return CandidateBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        candidate_sentence_indices=torch.tensor(sentence_indices, dtype=torch.long),
        candidate_token_spans=candidate_token_spans,
        candidate_kind_features=kind_multihot(candidates),
        candidate_replacement_ids=candidate_replacement_ids,
        candidate_replacement_mask=candidate_replacement_mask,
        candidate_offsets=tuple(candidate_offsets),
    )


def pool_candidate_spans(
    token_embeddings: torch.Tensor,
    candidate_sentence_indices: torch.Tensor,
    candidate_token_spans: torch.Tensor,
) -> torch.Tensor:
    """Return ``[start; end; mean]`` for each half-open candidate span."""
    hidden_size = token_embeddings.shape[-1]
    if candidate_token_spans.shape[0] == 0:
        return token_embeddings.new_empty((0, hidden_size * 3))

    starts = candidate_token_spans[:, 0]
    ends = candidate_token_spans[:, 1]
    if torch.any(ends <= starts):
        raise ValueError("candidate token spans must be non-empty")

    start_embeddings = token_embeddings[candidate_sentence_indices, starts]
    end_embeddings = token_embeddings[candidate_sentence_indices, ends - 1]
    prefix_sums = torch.cat(
        (
            token_embeddings.new_zeros((token_embeddings.shape[0], 1, hidden_size)),
            token_embeddings.cumsum(dim=1),
        ),
        dim=1,
    )
    span_sums = (
        prefix_sums[candidate_sentence_indices, ends]
        - prefix_sums[candidate_sentence_indices, starts]
    )
    span_lengths = (ends - starts).to(token_embeddings.dtype).unsqueeze(1)
    mean_embeddings = span_sums / span_lengths
    return torch.cat((start_embeddings, end_embeddings, mean_embeddings), dim=1)
