---
title: Premove ITN structured prediction
sidebarTitle: Structured prediction
description: How exact source and gold graphs train contextual candidate scores and decode compatible edits.
---

Premove ITN scores candidate edits, but the unit of training and decoding is a
**complete path through the transcript**. This matters because candidate spans
can overlap. A correct decision about one span can exclude another decision,
and several edit sequences can produce the same final sentence.

## A concrete ambiguity

For the source `the room code is one oh five`, the candidate builder produces
these two edits over the same source character interval, `[17, 28)`:

| Source span | Replacement | Candidate kinds |
| --- | --- | --- |
| `one oh five` | `105` | `DIGIT_SEQUENCE`, `CARDINAL`, `DATE`, `DECIMAL`, `PHONE` |
| `one oh five` | `01:05` | `TIME` |

The kinds describe which Rust realizers produced an output. They are features
for scoring, not separate copies of an identical replacement. If the expected
sentence ends in `105`, the time edit is a valid candidate but an incorrect
interpretation in this context. The graph must also allow the unchanged source
and edits to other, non-overlapping parts of a sentence.

## The source graph

Let the source string have `n` characters. Graph positions are the boundaries
`0` through `n`; they are **source character positions**, not tokenizer IDs.
Each candidate is a directed edge from its half-open `char_start` to
`char_end`, weighted by the scorer's scalar output. A one-character `KEEP` edge
from `i` to `i + 1` has weight zero at every source position.

A complete path starts at `0`, ends at `n`, and advances only forward. It
covers every source character once, either with an edit or with `KEEP`.
Non-overlapping edits can coexist; overlapping edits cannot occur on the same
path. Untouched spaces and punctuation are covered by `KEEP` even though the
candidate builder enumerates token spans. For example,
`meet at four thirty on march fourth` can select both `four thirty` →
`04:30` and `fourth` → `4th` in one path.

For path `p`, the score is the sum of its selected candidate scores:

$$
S(p) = \sum_{c \in p} s(c)
$$

The scores are learned compatibility values, not independent probabilities.
The all-`KEEP` path always scores zero. This gives the model a meaningful
alternative to every proposed edit without training a separate `KEEP` score.

## The exact gold graph

Training receives a source `x` and an expected output `y`.
`build_gold_graph()` tracks alignment states `(i, j)`, where `i` is a source
character position and `j` is an output character position. It starts at
`(0, 0)` and accepts only a path that reaches `(|x|, |y|)`.

There are three permitted forward transitions:

1. **Exact `KEEP`:** when `x[i] == y[j]`, advance to `(i + 1, j + 1)` with
   score zero.
2. **Candidate edit:** if candidate `c` starts at `i` and `y` begins with
   `c.replacement` at `j`, advance to
   `(c.char_end, j + |c.replacement|)` with score `s(c)`.
3. **Restricted attachment:** one source whitespace character may advance
   `i` without advancing `j` when a matching candidate immediately after it
   writes a symbol such as `%` that may attach to the previous output. This is how
   `the rate five percent` can align to `the rate 5%`. It is not a general
   whitespace deletion rule.

The builder explores reachable states, then walks backward from the final
state and discards transitions that cannot finish the target. If the final
state is unreachable, training preparation rejects the example. For an
unchanged source and target, the builder takes an explicit keep-only fast path.

The two coordinates prevent a subtle error: marking candidates as “gold” by
source position alone can recombine individually valid edges into a sequence
that writes the wrong target. The gold graph retains the exact target position
at every choice. It also retains **multiple complete derivations** when, for
example, two smaller edits and one whole-span edit both write the same target.
No arbitrary alignment needs to be selected for a changed target.

## Train by marginalizing complete paths

Let `P(x)` contain every complete path through the source graph. Let
`G(x, y)` contain the complete paths retained in the gold graph. The loss is:

$$
\mathcal{L}(x,y) =
\log \sum_{p \in P(x)} e^{S(p)}
- \log \sum_{p \in G(x,y)} e^{S(p)}
$$

This is the negative log probability assigned to *all* exact gold paths under
the source graph. A wrong but valid interpretation contributes to the first
term, not the second. For the room-code example, the `01:05` edge competes
with the `105` edge without a manually authored negative label.

`all_paths_log_partition()` computes the first log-sum-exp with a forward
dynamic program over source positions. From each position it propagates the
current value through `KEEP` and each candidate beginning there. The gold
partition uses the same operation over the pruned `(source, target)` states.
`torch.logaddexp` combines alternatives in log space without listing every
complete path. Because the gold term sums *all* exact derivations, a second
valid derivation is rewarded rather than treated as an error.

The source recurrence is compact. Initialize `F[0] = 0` and every other
position to negative infinity. At each source position `i`, update
`F[i + 1]` with `F[i]` for `KEEP`; for every candidate `c` starting at `i`,
update `F[c.char_end]` with `F[i] + s(c)`. Each update uses `logaddexp`.
The final `F[n]` is the all-path log partition. The gold recurrence uses the
same update on retained alignment states and their exact transitions.

The gradient for one candidate score is its marginal selection frequency
over all paths minus its marginal frequency over gold paths. This is the
mechanism behind contextual hard negatives: a wrong candidate with high
all-path mass receives pressure to lose to exact target paths.

`prepare_training_example()` builds the candidates, exact gold graph, and
tokenized inputs. For a batch, the scorer emits one flattened score vector;
`candidate_offsets` divides it back into sentences. `structured_batch_loss()`
computes one structured loss per sentence and averages them. The graph
topology stays on the host while scores remain PyTorch tensors, so gradients
flow through candidate scores into the scorer.

## Decode with max-sum instead of log-sum-exp

At inference there is no target or gold graph. `max_path_indices()` runs over
the same source intervals and replaces log-sum-exp with a maximum. It stores a
predecessor and optional candidate index for each best position, then
backtracks from `n`. The result is the **exact** highest-scoring compatible
path among the generated candidates, with no beam search.

`KEEP` makes the choice conservative: any negative-score candidate can be
replaced by zero-score `KEEP` edges over the same interval, leaving the other
selected edits compatible and increasing the path score. The decoder rejects
non-finite scores and invalid candidate spans. Equal totals retain the path
already recorded because updates use a strict `>` comparison; callers should
not interpret ties as a learned preference.

`apply_candidate_replacements()` sorts the selected edits, checks that their
spans do not overlap and still match the source, then copies untouched source
gaps verbatim. It applies the same restricted attachment rule used by gold
alignment to remove one separator before eligible symbols. The final string
therefore comes from validated replacements plus preserved source text, not
from unconstrained text generation.

Once candidates and scores exist, the source-graph dynamic programs take
`O(n + C)` time for `n` source characters and `C` candidate edges. Candidate
generation and model scoring have separate costs; this bound is for path
selection only. A form absent from the Rust candidate set cannot be recovered
by training or decoding.

For tokenization and scorer features, see the
[architecture](/itn/docs/internals/architecture). The implementation lives in
[`candidates.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/candidates.py),
[`structured_loss.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/structured_loss.py),
[`training_batch.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/training_batch.py),
and [`decoder.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/decoder.py).
