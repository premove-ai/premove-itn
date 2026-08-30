# SLURP dataset source note

## Decision

Use the real textual SLURP annotations from the official repository at commit
`8eb16545762be97ace75334109d73824217311f1`. Do not download audio or include
`train_synthetic.jsonl` in this spoken-context dataset.

## Primary-source findings

- The authors identify the official dataset repository as
  [`pswietojanski/slurp`](https://github.com/pswietojanski/slurp/tree/8eb16545762be97ace75334109d73824217311f1)
  and state that textual annotations are under `dataset/slurp/`. They provide
  `train.jsonl`, `devel.jsonl`, and `test.jsonl`, plus a separate synthetic
  training file. The repository README also says that audio is a separate
  download of about 6 GB from Zenodo. See the
  [revision-pinned README](https://github.com/pswietojanski/slurp/blob/8eb16545762be97ace75334109d73824217311f1/README.md).
- A record contains `slurp_id`, `sentence`, `sentence_annotation`, `intent`,
  `action`, `tokens`, `scenario`, `recordings`, and `entities`. An entity has a
  token-index `span` and a `type`. The README defines recording-level `wer`,
  `ent_wer`, and `status`; these do not supply a written ITN target. See the
  [official schema example](https://github.com/pswietojanski/slurp/blob/8eb16545762be97ace75334109d73824217311f1/README.md#brief-overview).
- The official license says repository-distributed textual data is CC BY 4.0,
  while separately hosted audio is CC BY-NC 4.0. See
  [`LICENSE.txt`](https://github.com/pswietojanski/slurp/blob/8eb16545762be97ace75334109d73824217311f1/LICENSE.txt).
- The paper describes SLURP as an English spoken-language-understanding dataset
  spanning 18 domains. The authoritative publication is
  [Bastianelli et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.588/).

## ITN limitation and compilation policy

SLURP does not provide written ITN references. Therefore, it cannot be treated
as equivalent to Google TN supervision. The compiler uses SLURP only as a
source of natural spoken contexts. It canonicalizes complete, aligned entity
spans only when a current Rust realizer supplies an unambiguous replacement for
the entity's semantic type. It otherwise keeps the source text unchanged. Each
accepted target must be constructively reachable through `GoldGraph`.

This policy avoids turning title numbers (for example, “movie three”) into
unsupported pseudo-gold labels. It also quarantines records whose token/entity
annotations cannot be aligned to the sentence. The output preserves the
official real-data splits (`devel` becomes `validation`) and records source IDs
in a separate provenance artifact.
