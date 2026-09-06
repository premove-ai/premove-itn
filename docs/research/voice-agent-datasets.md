# Voice-agent dataset landscape

## Conclusion

The highest-value additions for the current Premove ITN scorer are LibriTTS
and Taskmaster-2. LibriTTS has direct original/normalized text pairs that can
be reversed and filtered through `GoldGraph`. Taskmaster-2 has natural spoken
dialogue context and annotated spans for dates, times, money, counts, and
measurements. See the existing
[source audit](taskmaster-2-and-additional-itn-sources.md) for the detailed
decision and constraints.

For a future end-to-end voice-agent model, use task-oriented corpora for
training, broad speech corpora for acoustic robustness, and reserve current
audio-agent benchmarks for evaluation.

For that future voice-agent model, the first candidates are SLURP, SpokenWOZ,
Taskmaster, MASSIVE, Common Voice, FLEURS, and CoVoST 2. STOP is useful only
in a separate restricted research workflow because its license limits
derivative and incorporation uses. VoiceBench,
AudioBench, Dynamic-SUPERB, AIR-Bench, and MMAU are better held out as
evaluation suites. Training on their questions, answers, audio, or source test
sets would make later benchmark scores unreliable.

For Premove ITN specifically, this is a landscape note, not an implementation
proposal. The current architecture decision permits contextual candidate
scoring, but not an acoustic or end-to-end voice-agent model. `rust/` remains
the sole owner of deterministic realization. Most sources below lack native
spoken-to-written ITN references and cannot be treated as direct ITN
supervision.

## Recommended sources

Sizes below are the figures published by the dataset owners. Mutable datasets
must be pinned to an exact release or commit before use.

| Source | Task and languages | Published size | License or access | Best use |
| --- | --- | ---: | --- | --- |
| [STOP](https://github.com/facebookresearch/spoken_task_oriented_parsing) | End-to-end spoken semantic parsing; English; 8 assistant domains | More than 200,000 recordings, more than 800 speakers, from 125,000 unique TOPv2 utterance/parse pairs | The [official license](https://github.com/facebookresearch/spoken_task_oriented_parsing/blob/main/LICENSE) is restrictive and limits derivative works and incorporation into another dataset; public repository is archived | Restricted research evaluation or training only after legal review. Do not compile it into this repository's general-use training pool. |
| [SLURP](https://github.com/pswietojanski/slurp) | Spoken language understanding; English; 18 domains | 16,521 real text rows plus separate synthetic training data; audio is about 6 GB | Text is CC BY 4.0; audio is CC BY-NC 4.0, per the [official license](https://github.com/pswietojanski/slurp/blob/8eb16545762be97ace75334109d73824217311f1/LICENSE.txt) | Already prepared in this repository. Use real train data for intent, scenario, action, and entity learning. Treat synthetic and real recordings as separate domains. |
| [SpokenWOZ](https://spokenwoz.github.io/) | Multi-domain spoken dialogue state tracking; English | 5,700 dialogues, 203,000 turns, 249 hours | CC BY-NC 4.0, as stated by the [official dataset distribution](https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Text) | Train multi-turn state tracking and robustness to spoken disfluency. It is already prepared here and is noncommercial. |
| [Taskmaster](https://github.com/google-research-datasets/Taskmaster) | Spoken and written task-oriented dialogue; English; more than 12 domains | More than 55,000 dialogues across TM-1, TM-2, and TM-3 | Per-subcorpus terms must be checked and pinned; the official repository has no single top-level license | Train dialogue flow, slot carry-over, repair, and natural user phrasing. This repository already prepared a constrained TM-1 textual source. Do not assume one license covers all releases. |
| [Fluent Speech Commands](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) | Audio-to-action, object, and location slots; English | 30,043 utterances, 97 speakers, about 19 hours | Custom Fluent Speech Commands Public License; inspect the bundled terms before commercial use | Small, clean training/evaluation set for compositional intent and slot prediction. It is narrower than STOP but useful for fast ablations. |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) | Text intent classification and slot filling; 51 languages, 60 intents, 55 slot types | More than 1 million localized utterances; 11,514 train, 2,033 dev, and 2,974 test per language | CC BY 4.0 | Train multilingual NLU after ASR, especially Hindi and other Indian languages. It contains text, not audio, and localizes SLURP, so English/semantic overlap with SLURP must be grouped during splitting. |
| [Speech-MASSIVE](https://github.com/hlt-mt/speech-massive) | Spoken SLU; 12 languages; 18 domains, 60 intents, and 55 slots | Full French and German training sets; 115-example few-shot training split for each supported language | CC BY-NC-SA 4.0 | Train multilingual audio-to-intent/slot models. It is closer to the requested use than text-only MASSIVE, but it is noncommercial and ShareAlike. Group localized MASSIVE prompts across languages and modalities. |
| [Common Voice](https://commonvoice.mozilla.org/en/datasets) | Crowdsourced ASR and language/accent coverage; multilingual | Release-dependent; Common Voice 21 reported nearly 33,500 hours, 134 languages, and more than 350,000 speakers in the [official release notice](https://discourse.mozilla.org/t/common-voice-21-dataset-now-available/141629) | CC0; download requires acceptance of dataset terms and an email address | Train or adapt the acoustic front end, accent robustness, and language identification. It does not provide agent actions or ITN targets. Pin release, locale, validated clips, and speaker-disjoint splits. |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | ASR, language identification, and retrieval; 102 languages | 2,009 parallel sentences per language, about 10 training hours per language; 768,120 rows in the hosted representation | CC BY 4.0 | Controlled multilingual evaluation and limited fine-tuning. Its parallel prompts make cross-language comparisons clean but create extensive repeated semantic content. |
| [CoVoST 2](https://github.com/facebookresearch/covost) | Multilingual speech-to-text translation | 2,880 hours, 78,000 speakers; 21 languages into English and English into 15 languages | Core CoVoST data is CC0; Tatoeba evaluation audio has per-item licenses and other repository material is CC BY-NC 4.0 | Train speech translation and multilingual acoustic-text alignment. It derives audio from Common Voice v4, so deduplicate by clip path or audio hash across both sources. |
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | Multi-party meeting speech; English; close and far microphones, overlap, timing, dialogue acts, and summaries | About 100 hours | Many layers are CC BY 4.0; confirm the terms for each downloaded annotation and media component | Train diarization, addressee/turn detection, interruption handling, and far-field ASR. It is conversational but is not a task-oriented assistant corpus. Split by meeting and participant. |
| [VoxPopuli](https://github.com/facebookresearch/voxpopuli) | ASR, speech translation, and accent recognition; European parliamentary speech | 400,000 hours unlabeled in 23 languages; 1,800 transcribed hours in 16; 17,300 interpretation hours; 29 hours of accented English | Data is CC0 subject to the European Parliament notice; repository code and models are CC BY-NC 4.0 | Acoustic pretraining and speech translation. Keep data rights distinct from the noncommercial code/model license. Formal parliamentary speech is not voice-agent dialogue. |

## Evaluation-only suites

| Suite | Coverage and size | License or access | Recommended policy |
| --- | --- | --- | --- |
| [VoiceBench](https://github.com/MatthewCYM/VoiceBench) | 20,554 English rows covering instruction following, knowledge, reasoning, safety, multi-turn, and speech robustness; mixes TTS and human speech | Dataset card states Apache-2.0, but it derives from AlpacaEval, OpenBookQA, MMLU-Pro, MT-Bench, IFEval, AdvBench, and other sources whose terms also need review | Evaluation only. Keep every prompt, recording, answer, and source benchmark out of training and development prompt tuning. |
| [MultiTalkBench](https://huggingface.co/datasets/MultiTalk/MultiTalkBench) | Long, multi-party, full-duplex dialogue in English and Chinese; addressee selection, turn taking, and entity tracking | 104 test sessions, 56.52 hours; sessions exceed 10 minutes | CC BY-SA 4.0 | Evaluation only. It fills a gap left by short, single-speaker benchmarks. Do not train on its test-only sessions. |
| [AudioBench](https://github.com/audiollms/audiobench) | More than 26 component datasets spanning speech understanding, audio scenes, and paralinguistics; includes ASR, translation, spoken QA, and long-form tests | Mixed upstream licenses; no single repository license grants rights to every component | Evaluation only. Build a per-component license manifest. Do not infer data rights from the benchmark code license. |
| [Dynamic-SUPERB Phase 2](https://github.com/dynamic-superb/dynamic-superb) | 180 instruction-guided tasks across speech, music, and environmental audio; classification, regression, and generation | Mixed source licenses, documented per dataset in the official repository | Evaluation only. Select a stable task release and keep its source datasets out of training where possible. |
| [AIR-Bench](https://github.com/OFA-Sys/AIR-Bench) | 19 foundation tasks with about 19,000 multiple-choice questions and 2,000 open-ended chat instances over speech, sounds, music, and mixed audio | Repository states Apache-2.0, but it is assembled from multiple source datasets; upstream restrictions still need review | Evaluation only. Its generated questions and GPT-based chat judge make it useful as a broad diagnostic, not clean supervised truth. |
| [MMAU](https://github.com/Sakshi113/MMAU) | Expert audio understanding and reasoning over speech, sound, and music; 27 tasks | 10,000 clips/questions; 1,000-answer `test-mini`, 9,000-answer hidden test; project states CC BY 4.0 | Evaluation only. The 9,000 answers are withheld. Keep `test-mini` out of training and prompt selection. The repository also warns that redistributed source media can be removed after copyright reports. |

These suites overlap conceptually and sometimes physically with public source
datasets. A single aggregate score should not be used as a training target.
Report per-capability results and the exact suite revision.

## Leakage and licensing controls

1. Create three registries: `train`, `development`, and `sealed evaluation`.
   Put VoiceBench, AudioBench, Dynamic-SUPERB, AIR-Bench, MMAU, and the supplied
   VoiceAgentBench/audio-agent-bench-suite splits in the sealed registry.
2. Record dataset revision, upstream source, license, original split, speaker or
   conversation ID, and a normalized transcript hash for every example. For
   audio, also record a robust audio fingerprint. URL or filename matching is
   insufficient after resampling or re-encoding.
3. Split connected components, not rows. Group the same speaker, dialogue,
   source prompt, translated/localized prompt, transcript, and near-duplicate
   audio in one split. This is essential for SLURP/MASSIVE, Common
   Voice/CoVoST, TOPv2/STOP, and any TTS version of a public text benchmark.
4. Do not train on benchmark answers, judge rubrics, test transcripts, or TTS
   renditions of test questions. TTS changes the waveform but does not remove
   semantic leakage.
5. Keep commercial and noncommercial pools separate. STOP, SpokenWOZ, and
   SLURP audio are noncommercial. Taskmaster and aggregate suites need
   per-release or per-component review. A permissive code license does not
   relicense included data.
6. Treat public benchmark performance as potentially contaminated when the
   base model's pretraining data is undisclosed. Add a private, newly recorded,
   speaker-disjoint test set with paraphrased tasks and held-out tool schemas.

## Practical order

1. For current Premove ITN work, audit LibriTTS text pairs and Taskmaster-2
   annotated user turns first.
2. For a separate end-to-end agent, add MASSIVE for multilingual NLU. Use STOP
   only in a legally separate restricted research workflow.
3. Use Common Voice plus FLEURS for acoustic, accent, and language robustness.
4. Add CoVoST 2 only if translation is a product requirement.
5. Use SpokenWOZ and Taskmaster for multi-turn dialogue behavior; preserve
   their conversation-level splits and license boundaries.
6. Freeze a compact evaluation matrix from VoiceBench, AudioBench,
   Dynamic-SUPERB, AIR-Bench, and MMAU. Never mix that matrix into instruction
   tuning.

This order strengthens separate failure modes. It avoids confusing more hours
of generic ASR audio with better tool use, dialogue state, or spoken reasoning.
