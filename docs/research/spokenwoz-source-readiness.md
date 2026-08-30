# SpokenWOZ source readiness

## Conclusion

SpokenWOZ is usable as a **research-only** contextual ITN source. The official
site releases it under CC BY-NC 4.0. Do not mix its records into a commercial
training artifact. The source does not supply Google-TN-style spoken/written
pairs. The compiler must therefore retain only target edits that both originate
in an official `span_info` annotation and are exactly constructible by a current
Rust realizer.

## Primary sources

- The [official project site](https://spokenwoz.github.io/SpokenWOZ-github.io/)
  identifies 5,700 dialogues, 203,000 turns, 249 audio hours, train/dev/test
  splits, the CC BY-NC 4.0 license, and the official text downloads.
- The [NeurIPS 2023 paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7b16688a2b053a1b01474ab5c78ce662-Paper-Datasets_and_Benchmarks.pdf)
  reports 4,200/500/1,000 train/dev/test dialogues and 149,126/18,384/35,564
  turns. Appendix D documents the JSON schema: each dialogue has a goal and a
  log; turns carry text, dialogue state, dialogue acts, span annotations, and
  word-level ASR timing.
- The authors' [official code directory](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/spokenwoz)
  contains the baseline and evaluation implementation linked from the project
  site.
- The official author-owned Hugging Face releases are revision-pinned below:
  [train/dev](https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Text/tree/d3aad10f2e5a37e7e1e84375f0db368b5872a044)
  and [fixed test](https://huggingface.co/datasets/ssz1111/SpokenWOZ-Test-Text-Fixed/tree/d6c2d9e53b1005e327db582d327b69311092eb65).
- [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode)
  permits sharing and adaptation with attribution, notices, and no additional
  restrictions, but only for non-commercial purposes.

## Downloaded sources

Raw files are intentionally ignored under `data/external/spokenwoz/`.

| Source | Revision | SHA-256 | Size |
|---|---|---|---:|
| train/dev `data.json` | `d3aad10f2e5a37e7e1e84375f0db368b5872a044` | `017db5b19f6b6e7c1173aadce2ea6ab204c19f6c8f4359d91d0b9fb3d007b58d` | 246,209,221 bytes |
| train/dev `valListFile.json` | same | `f79f5a80afe428a667cfa4a0aed41fe235507c41071fd44647ae85b862089515` | official line-delimited ID list |
| fixed-test `data.json` | `d6c2d9e53b1005e327db582d327b69311092eb65` | `96b6ff493a8294ffaf03021c024ef4aa2bfe92f980b84c029c81f2aa0f07f728` | 52,232,323 bytes |
| fixed-test `testListFile.json` | same | `b1aaf07ebb76bf8e0dbca273ec6ec89739fb70b4e397e9db4f581a595020d0a5` | official line-delimited ID list |

## Compiler policy

The source `text` is a transcript, not a written ITN target. `span_info` gives
an annotated canonical slot value plus inclusive transcript-token indices. The
compiler searches only inside that annotated range. It applies a value only if
`realize_options` produces that exact value. It normalizes whitespace around
punctuation, retains official dialogue-level partitions, keeps only user turns,
deduplicates full `(text, expected_text)` pairs, records bounded quarantine
provenance, and requires `build_gold_graph` success before publication.

This policy deliberately rejects useful-looking slot corrections when they are
not deterministic ITN. For example, ASR repairs, entity-name corrections, and
reasoning-derived times are not treated as gold normalization. Without this
invariant, the classifier would learn ASR correction and dialogue-state
reasoning as if they were Rust realization behavior.

The compiler emits the same five training fields as Google Dataset 1 and keeps
source dialogue/turn IDs in a parallel `provenance.jsonl`. The audit checks the
schema, partitions, uniqueness, kind labels, manifest distribution, provenance
alignment, and every record through `GoldGraph`.

## Complete build result

The revision-pinned build processed 202,950 turns and emitted 74,979 unique
user-turn records: 49,920 train KEEP records, 6,081 validation KEEP records,
10,976 test KEEP records, and 8,002 records with one or more constructive ITN
kinds. Contained-kind counts include 7,199 CARDINAL, 7,190 DIGIT_SEQUENCE,
3,670 DECIMAL, 703 TIME, 384 PHONE, 82 WORD, 33 ELECTRONIC, 59 DATE, and 3
MEASUREMENT occurrences. A record can count toward multiple kinds when Rust
exposes more than one valid derivation.

The build quarantined 34 turns with overlapping constructible annotations and
3 turns whose final target was not reachable. It excluded 101,475 system turns
by policy. These exclusions are recorded in the manifest; bounded actionable
examples are stored in `quarantine.jsonl`.

## Reproduction

```bash
uv run python -m scripts.build_spokenwoz_dataset \
  data/external/spokenwoz/d3aad10f2e5a37e7e1e84375f0db368b5872a044/data.json \
  data/external/spokenwoz/d3aad10f2e5a37e7e1e84375f0db368b5872a044/valListFile.json \
  data/external/spokenwoz/d6c2d9e53b1005e327db582d327b69311092eb65/data.json \
  data/generated/spokenwoz
uv run python -m scripts.audit_spokenwoz_dataset data/generated/spokenwoz
```
