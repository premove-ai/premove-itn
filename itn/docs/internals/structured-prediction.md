---
title: Structured prediction
description: How Premove ITN trains and decodes over complete compatible edit paths.
---

Premove ITN trains and decodes over complete compatible edit paths. It does not
classify candidate spans independently.

## Candidate graph

Each candidate is a weighted edge over a half-open source interval. A
one-character `KEEP` edge with score zero is available at every source
position. A complete path covers the entire source text without overlapping
candidate intervals.

## Gold graph

Training aligns source and target positions. Gold states retain both positions
so that different candidate and `KEEP` transitions can represent the same
target. A target is trainable only when the candidate graph can reach it
exactly.

## Objective

The structured loss compares the log-partition over all complete paths with the
log-partition over gold-compatible paths. This trains candidate scores in the
context of competing and overlapping edits.

## Exact decoding

At inference time, dynamic programming finds the maximum-score complete path.
Predecessor pointers recover the selected non-overlapping candidates. A
negative-scoring candidate loses to `KEEP` unless another compatible path has a
higher total score.

See the [architecture](/itn/docs/internals/architecture) for the complete inference pipeline and
[`src/premove_itn/structured_loss.py`](../../../src/premove_itn/structured_loss.py)
for the implementation.
