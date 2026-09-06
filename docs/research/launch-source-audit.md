# Launch source audit

## Decision

For a public package and model release in 48 hours, freeze the data mixture.
Make provenance, reproducible evaluation, and release artifacts the critical
path. Do not make a new dataset compiler or adaptation run a launch dependency.
This is a recommendation, not a measured claim about the current checkpoint.

Sources were checked on 2026-09-06. Dataset cards establish source facts. They
do not establish that a particular checkpoint can be distributed. This note
does not verify the pasted plan's model scores or training lineage.

## PolyNorm: useful diagnostic, not a native ITN gold standard

The [Apple release](https://github.com/apple/ml-speech-polynorm-bench) describes
a TN benchmark made from synthetic examples followed by expert editing. I
read its [en-US JSONL](https://github.com/apple/ml-speech-polynorm-bench/blob/main/polynorm_bench/en-US/en-US_groundtruth.jsonl)
and counted 540 rows, 27 categories, and 20 rows per category. Fields are
`index`, `category`, `original_text`, and `normalized_text`.

Use the file's category names. They differ from the README: the data uses
`Phone Number`, `URL or Email`, `Hashtag or Mention`, `Unit`, and
`License Plate or Serial Numbers`, among others.

Reversing `normalized_text -> original_text` is mechanically possible. It does
not make the original string uniquely recoverable. Row 383 loses the hyphen
from a serial number during TN. Several version rows expand a written `v` to
the word `version`. Date formats also encode a written policy. A correct ITN
system can return a different valid representation. This is an inference from
the inspected pairs, not a claimed defect in Apple's TN benchmark.

Recommended report:

1. All 540 examples, strict sentence exact match, with the full denominator.
2. Coverage by the frozen Rust candidate generator.
3. Selection accuracy on reachable examples, with its denominator.
4. A separate, predeclared formatting-equivalence score where appropriate.
5. Error labels for candidate absence, selection, formatting, unsupported
   transformation, and ambiguous reversal.

Do not silently remove unreachable rows or change equivalence rules after
seeing model outputs. Twenty rows per category means each error changes a
category score by 5 percentage points. Treat rare-category results as a
diagnostic. Repeated checkpoint selection makes this development data, even
if it is never used for gradient updates.

The [license](https://github.com/apple/ml-speech-polynorm-bench/blob/main/LICENSE)
is CC BY-NC-ND 4.0. Its grant permits noncommercial reproduction and private
adaptation; sharing adapted material is excluded. Calling a run
“evaluation-only” does not establish that a commercial product evaluation is
permitted. Do not bundle transformed rows in the public package without a
resolved basis for that use. Defer this benchmark if its terms would delay
the release.

## Candidate training sources

| Source | Verified facts | Recommendation for 48 hours |
| --- | --- | --- |
| [Taskmaster-2](https://github.com/google-research-datasets/Taskmaster/tree/master/TM-2-2020) | README reports 17,289 dialogs and CC BY 4.0. USER turns are transcribed speech; ASSISTANT turns are written and played through TTS. Semantic spans have two annotators. There are no API-call annotations. Transcription repair can insert, remove, or replace language. | Defer compilation. Spans are weak semantic evidence, not canonical ITN targets. No source evidence promises useful ELECTRONIC or PHONE yield. |
| [Taskmaster-3](https://github.com/google-research-datasets/Taskmaster/tree/master/TM-3-2020) | 23,789 movie-ticket dialogs; CC BY 4.0; written self-dialogue. Contains semantic spans, API arguments, repairs, and clarifications. API values can include defaults or information inferred from dialogue. | Defer. API arguments do not automatically align to a spoken span. Preserve complete dialogue context when auditing. |
| [LJSpeech](https://www.tensorflow.org/datasets/catalog/ljspeech) | 13,100 clips, about 24 hours, one reader, seven nonfiction books; public-domain text/audio according to the release description. Exposes `text` and `text_normalized`. | Defer. Reversal can add realization variants but is not evidence of conversational ambiguity coverage. Only audit metadata if already available. |
| [LibriTTS](https://www.openslr.org/60/) | About 585 hours of read English speech; CC BY 4.0; original and normalized text. [TFDS](https://www.tensorflow.org/datasets/catalog/libritts) lists 33,236 + 116,500 + 205,044 = 354,780 training utterances. | Defer. The full TFDS download is 78.42 GiB. No need to download audio for text-pair research. |

The existing [Taskmaster source note](taskmaster-2-and-additional-itn-sources.md)
already records a pinned TM2 audit with 17,304 JSON dialogs, rather than the
README's 17,289. That count was not recomputed here. Preserve the discrepancy
and use pinned file counts in a compiler manifest. The repository now also
contains TM4 despite the top-level README describing three datasets.

For later training, group splits by dialogue/instruction family for
Taskmaster and by the official speaker/chapter partitions for audiobook
sources. Dedupe against every evaluation source. Reachability establishes
that Rust can produce a label; it does not establish that the label is
semantically correct. Independently review weak labels. Avoid testing on
near-identical templates from synthetic training families.

## Audio benchmarks

### Audio2Tool

The [first-party card](https://huggingface.co/datasets/RVtech/Audio2Tool)
reports 16,843 queries, 36,421 audio files, 152 tools, and eight tiers.
The speech is synthesized from reference voices. It is not a corpus of
spontaneous human requests.

The proposed tiers are useful but large: tier2 has 6,320 files/5.38 hours;
tier6 has 4,292 files/9.18 hours. That is 10,612 files and 14.56 audio hours
before retries or multiple pipeline variants. Query counts and audio counts
are different denominators. Voice variants of one query are correlated.

Recommendation: use a pinned, stratified sample of 100–200 unique queries
only if the harness and permitted-use basis already exist. Keep the same ASR
outputs and extractor settings for raw, Rust, and model branches. Score
parameter values, not just whole-agent success. A correction task also tests
dialogue interpretation; it cannot isolate ITN by itself.

The card specifies CC BY-NC 4.0 and says commercial use requires separate
permission. It also retains underlying speech-corpus terms. Evaluation-only
status does not remove those conditions.

### Audio Agent Bench

The [suite card](https://huggingface.co/datasets/arcada-labs/audio-agent-bench-suite)
has six domains and 221 scripted turns, recorded by two human voice actors.
Appointment, product, and assistant contain 25 + 31 + 31 = 87 turns. It is
small enough for a smoke test if an agent harness exists. Run complete
dialogs: later turns depend on earlier turns. The card specifies CC BY 4.0
and explicitly asks that it be used for evaluation, not training, despite
the loader naming its split `train`.

It tests limited speakers and scripted input. It cannot establish broad
accent/noise robustness or streaming stability. This is a scope limitation,
not a reason to discard the smoke test.

### VoiceAgentBench

The [dataset card](https://huggingface.co/datasets/krutrim-ai-labs/VoiceAgentBench)
describes multilingual tool use, dialogue, and refusal tasks. Its README
does not establish the pasted plan's exact 5,394 count. Do not repeat that
count without a pinned file audit.

The [actual license](https://huggingface.co/datasets/krutrim-ai-labs/VoiceAgentBench/blob/main/LICENSE.md)
contains restrictions on competition, commercial use/distribution, and
commercial derivative models. It is not simply a generic benchmark license.
Recommendation: defer it from this public release's critical path.

## Public model release: unresolved source terms are a gate

The [original author's Google TN repository](https://github.com/rwsproat/text-normalization-data)
points to a [Kaggle dataset](https://www.kaggle.com/datasets/richardwilliamsproat/text-normalization-for-english-russian-and-polish).
The dataset and English competition rules pages returned no readable terms
in this audit. The source's exact permissions remain unverified. Do not
substitute the Russian competition's rules, a mirror's license label, a
notebook license, or Wikipedia's license for the downloaded artifact's terms.

Before publishing weights, record the exact base model, tokenizer, training
data versions, source terms, and checkpoint lineage. Map each obligation to
the intended public artifact and uses. Include required notices. Dataset
licenses do not automatically become a model's license; nor does clean
additional training establish rights to an earlier checkpoint. The pasted
plan's “research/private beta” label also does not establish permission.

If terms cannot be resolved by the deadline, mark public weights blocked.
A separately cleared package can still ship, but that is not completion of
the user's package-and-model release. Do not hide the distinction by changing
the label to beta.

## Evidence limits

This was a source audit. It did not download audio, run ASR, compile training
data, or evaluate model accuracy. No code changed. No runtime validation is
needed for this note; the release itself still needs all repository checks.
