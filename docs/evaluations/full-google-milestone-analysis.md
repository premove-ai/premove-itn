# Full-Google checkpoint analysis

This report compares every preserved full-Google milestone. Golden and NVIDIA
Numb3rs are evaluation-only. Google validation is evaluated only on the final
checkpoint. No evaluation input was used for training or model selection.

## Overall curve

| Exposure | Golden exact | Golden positive | Golden KEEP | Golden MULTI | Numb3rs exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50,000 | 64.71% | 57.29% | 82.50% | 25.00% | 34.45% |
| 98,000 | 64.71% | 57.29% | 82.50% | 25.00% | 34.45% |
| 146,000 | 59.56% | 51.04% | 80.00% | 37.50% | 29.04% |
| 202,000 | 68.38% | 64.58% | 77.50% | 37.50% | 29.96% |
| 250,000 | 70.59% | 77.08% | 55.00% | 62.50% | 38.24% |
| 298,000 | **74.26%** | **79.17%** | 62.50% | 62.50% | 37.95% |
| 346,000 | 72.79% | 77.08% | 62.50% | **75.00%** | 38.68% |
| 402,000 | 73.53% | 73.96% | 72.50% | 62.50% | 38.78% |
| 450,000 | 73.53% | 75.00% | 70.00% | 50.00% | **38.95%** |
| 498,000 | 69.85% | 69.79% | 70.00% | 25.00% | 38.63% |
| 546,000 | 69.12% | 72.92% | 60.00% | 37.50% | 37.14% |
| 602,000 | 67.65% | 66.67% | 70.00% | 62.50% | 32.92% |
| 650,000 | 70.59% | 72.92% | 65.00% | 25.00% | 37.08% |
| 698,000 | 62.50% | 59.38% | 70.00% | 50.00% | 34.04% |
| 711,135 | 68.38% | 68.75% | 67.50% | 37.50% | 36.34% |

Golden peaks at 298k with 101/136 correct. The separately retained 378k
checkpoint ties this exact score, with a different positive/KEEP balance.
Numb3rs peaks at 450k with 3,946/10,131 correct. Its deterministic baseline
remains higher at 39.43%. The curve is non-monotonic, so the final checkpoint
is not the best generalization checkpoint.

## Golden per-kind accuracy

| Exposure | CARD | DATE | DEC | DIGIT | ELEC | KEEP | MEAS | MONEY | MULTI | ORD | PHONE | TIME |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k | 66.7% | 100% | 100% | 88.2% | 0% | 82.5% | 33.3% | 87.5% | 25.0% | 42.9% | 40.0% | 0% |
| 98k | 66.7% | 100% | 100% | 88.2% | 0% | 82.5% | 33.3% | 87.5% | 25.0% | 42.9% | 40.0% | 0% |
| 146k | 66.7% | 100% | 100% | 64.7% | 0% | 80.0% | 50.0% | 25.0% | 37.5% | 57.1% | 40.0% | 0% |
| 202k | 88.9% | 100% | 100% | 64.7% | 20.0% | 77.5% | 66.7% | 100% | 37.5% | 57.1% | 40.0% | 23.1% |
| 250k | 88.9% | 100% | 90.0% | 82.4% | 40.0% | 55.0% | 66.7% | 75.0% | 62.5% | 100% | 40.0% | 69.2% |
| 298k | 77.8% | 87.5% | 90.0% | 82.4% | 0% | 62.5% | 66.7% | 100% | 62.5% | 100% | 80.0% | 84.6% |
| 346k | 77.8% | 100% | 90.0% | 70.6% | 20.0% | 62.5% | 66.7% | 87.5% | 75.0% | 100% | 60.0% | 76.9% |
| 402k | 88.9% | 100% | 70.0% | 64.7% | 20.0% | 72.5% | 66.7% | 75.0% | 62.5% | 100% | 100% | 69.2% |
| 450k | 88.9% | 100% | 70.0% | 76.5% | 20.0% | 70.0% | 66.7% | 100% | 50.0% | 100% | 60.0% | 69.2% |
| 498k | 88.9% | 100% | 70.0% | 64.7% | 20.0% | 70.0% | 66.7% | 100% | 25.0% | 100% | 40.0% | 69.2% |
| 546k | 66.7% | 100% | 80.0% | 76.5% | 20.0% | 60.0% | 66.7% | 100% | 37.5% | 100% | 40.0% | 76.9% |
| 602k | 77.8% | 100% | 30.0% | 58.8% | 0% | 70.0% | 66.7% | 100% | 62.5% | 100% | 60.0% | 69.2% |
| 650k | 77.8% | 100% | 80.0% | 82.4% | 20.0% | 65.0% | 66.7% | 100% | 25.0% | 100% | 40.0% | 69.2% |
| 698k | 33.3% | 87.5% | 30.0% | 58.8% | 0% | 70.0% | 50.0% | 100% | 50.0% | 100% | 60.0% | 69.2% |
| Final | 77.8% | 100% | 70.0% | 64.7% | 0% | 67.5% | 66.7% | 100% | 37.5% | 100% | 40.0% | 69.2% |

Golden has only 5–40 examples per displayed kind. Treat large percentage
changes, especially ELECTRONIC, MULTI, and PHONE, as directional rather than
precise. The main persistent weaknesses are ELECTRONIC, KEEP over-normalization,
and unstable MULTI selection. The final checkpoint recovers from the 698k
collapse but remains six exact matches below the 298k peak.

## Numb3rs per-category accuracy

| Exposure | ADDRESS | CARD | DATE | DEC | DIGIT | FRACTION | MEAS | MONEY | ORD | PLAIN | PHONE | TIME |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k | 0% | 81.9% | 49.2% | 62.5% | 84.3% | 0% | 23.3% | 12.1% | 75.8% | 0% | 6.1% | 5.4% |
| 98k | 0% | 81.9% | 49.2% | 62.5% | 84.3% | 0% | 23.3% | 12.1% | 75.8% | 0% | 6.1% | 5.4% |
| 146k | 0% | 71.7% | 49.0% | 62.4% | 79.4% | 0% | 22.5% | 11.0% | 34.3% | 0% | 6.2% | 3.8% |
| 202k | 0% | 69.7% | 47.8% | 62.4% | 84.0% | 0% | 24.6% | 11.7% | 39.6% | 0% | 6.0% | 4.9% |
| 250k | 0% | 87.6% | 48.7% | 62.8% | 81.5% | 0% | 27.7% | 9.5% | 99.1% | 0% | 6.0% | 18.3% |
| 298k | 0% | 87.8% | 41.4% | 62.6% | 78.2% | 0% | 30.6% | 12.4% | 95.2% | 0% | 4.6% | 25.6% |
| 346k | 0% | 89.9% | 48.3% | 62.7% | 79.9% | 0% | 31.5% | 10.5% | 99.1% | 0% | 1.6% | 22.8% |
| 402k | 0% | 90.0% | 48.1% | 62.8% | 80.2% | 0% | 31.3% | 11.5% | 99.1% | 0% | 1.7% | 22.9% |
| 450k | 0% | 90.0% | 47.0% | 62.8% | 83.1% | 0% | 30.5% | 12.4% | 99.1% | 0% | 1.0% | 24.2% |
| 498k | 0% | 86.3% | 48.5% | 62.7% | 83.7% | 0% | 31.1% | 11.4% | 99.1% | 0% | 1.1% | 22.2% |
| 546k | 0% | 84.5% | 35.9% | 62.7% | 77.8% | 0% | 30.3% | 12.4% | 98.7% | 0% | 1.6% | 25.1% |
| 602k | 0% | 46.9% | 48.0% | 55.0% | 77.3% | 0% | 27.1% | 11.4% | 90.1% | 0% | 1.4% | 19.3% |
| 650k | 0% | 78.8% | 44.4% | 62.2% | 81.7% | 0% | 28.1% | 12.4% | 94.7% | 0% | 1.4% | 24.2% |
| 698k | 0% | 67.6% | 46.9% | 59.2% | 79.5% | 0% | 29.9% | 12.3% | 75.4% | 0% | 1.7% | 20.7% |
| Final | 0% | 75.0% | 48.4% | 61.5% | 81.5% | 0% | 31.2% | 12.3% | 88.5% | 0% | 1.2% | 19.7% |

Numb3rs gains are concentrated in CARDINAL, DIGIT, MEASURE, ORDINAL, and TIME.
ADDRESS, FRACTION, and PLAIN remain unreachable under the current candidate
system. MONEY and TELEPHONE remain low because many references use rendering
policies not represented by the current realizers. The 602k drop is driven
mainly by CARDINAL, DECIMAL, ORDINAL, and TIME and partially recovers by final.

## Cross-benchmark checkpoint selection

The comparable validation suites select **378k** as the best overall base for
the next experiment. The 346k checkpoint is the strongest alternative when
conversational balanced accuracy is the sole priority. Conversational exact
accuracy is dominated by its 9,469
KEEP rows, so the balanced column is the unweighted mean of conversational
positive accuracy and candidate-bearing KEEP accuracy. It prevents 5,067
candidate-free KEEP rows from hiding selection errors.

| System | Golden exact | Numb3rs exact | Google exact | Conversational exact | Conversational positive | Candidate-bearing KEEP | Conversational balanced |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deterministic Rust | 65.44% | **39.43%** | 61.19% | 89.11% | 32.98% | 84.98% | 58.98% |
| 1k | 69.85% | 22.19% | 80.81% | **92.57%** | 42.64% | **91.41%** | 67.03% |
| 10k | 70.59% | 25.33% | 88.35% | 88.91% | 27.76% | 85.21% | 56.49% |
| 250k | 70.59% | 38.24% | 93.67% | 86.52% | **75.46%** | 72.65% | 74.05% |
| 298k | **74.26%** | 37.95% | 95.04% | 88.81% | 66.10% | 79.28% | 72.69% |
| 346k | 72.79% | 38.68% | 95.09% | 88.84% | 74.08% | 78.17% | **76.12%** |
| **378k** | **74.26%** | 38.91% | 95.24% | 89.21% | 69.48% | 79.71% | 74.60% |
| 402k | 73.53% | 38.78% | 95.27% | 87.86% | 72.09% | 76.22% | 74.15% |
| 450k | 73.53% | 38.95% | **95.43%** | 82.28% | 64.42% | 64.54% | 64.48% |
| 498k | 69.85% | 38.63% | 94.87% | 87.62% | 66.56% | 76.49% | 71.53% |
| 711,135 | 68.38% | 36.34% | 95.11% | 88.26% | 63.19% | 78.46% | 70.83% |

The 346k checkpoint leads conversational balanced accuracy by 1.53 points over
378k, mainly through stronger positive recall. The 378k checkpoint retains
1.55 points more candidate-bearing KEEP accuracy, ties the best Golden result,
and also scores higher on Numb3rs, Google, and conversational exact accuracy.
It is therefore the safer base before conversational adaptation. The 450k
checkpoint wins Google by 0.19 points and Numb3rs by 0.04 points over 378k, but
loses one Golden record and 10.12 points of conversational balanced accuracy.
The final checkpoint is worse than 378k on all four headline exact metrics. The
1k checkpoint's high conversational exact score comes from KEEP; its positive
accuracy is only 42.64%.

### Conversational validation by source

| Checkpoint | SGD positive | SLURP positive | SpokenWOZ positive | Taskmaster-1 positive |
| ---: | ---: | ---: | ---: | ---: |
| 1k | 7.64% | 42.11% | 64.53% | 17.86% |
| 10k | 3.47% | 36.84% | 44.13% | 3.57% |
| 298k | 80.56% | **71.05%** | 65.36% | **48.21%** |
| **378k** | **81.25%** | 63.16% | **75.42%** | 37.50% |
| 450k | 72.22% | 60.53% | 70.67% | 35.71% |
| 711,135 | 72.92% | 52.63% | 68.72% | 36.61% |

### Conversational validation by kind

| Checkpoint | CARD | DATE | DEC | DIGIT | ELEC | KEEP | MEAS | MULTI | TIME | WORD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 298k | 72.3% | 66.7% | 95.2% | 92.3% | 50.0% | 90.4% | 0% | 61.8% | **68.3%** | 0% |
| **378k** | **72.3%** | **66.7%** | 95.2% | **92.3%** | **50.0%** | **90.6%** | 0% | **72.8%** | 60.9% | 0% |
| 450k | 70.8% | 66.7% | **100%** | 84.6% | 50.0% | 83.5% | 0% | 68.0% | 53.0% | 0% |
| 711,135 | 72.3% | 66.7% | **100%** | 15.4% | 0% | 90.0% | **33.3%** | 65.7% | 56.9% | 0% |

DATE, ELECTRONIC, MEASUREMENT, and WORD contain only 2–6 validation records,
so their percentages are directional. The larger MULTI, TIME, CARDINAL, and
KEEP groups provide more useful checkpoint evidence.

The next phase should use the combined conversational training pool rather than
one source in sequence. SGD and SpokenWOZ provide the strongest validation
signal at 378k. SLURP and Taskmaster-1 remain weaker and would be forgotten more
easily by sequential fine-tuning. Candidate-bearing conversational KEEP must be
sampled alongside all conversational positives, with kind-balanced Google
positive replay to retain Google performance. Candidate-free KEEP should remain
in the corpus and validation set but does not need model training exposure.

The conversational validation set contains 10,121 records: 652 positive, 9,469
KEEP, and 4,402 candidate-bearing KEEP. An exhaustive identity audit found zero
exact pair, source-text, target-text, or cross-value overlap with the 711,352
Google training rows. No conversational test record was evaluated.

## Selected Google validation

| System | Exact | Positive | KEEP | MULTI |
| --- | ---: | ---: | ---: | ---: |
| Deterministic Rust | 61.19% | 56.83% | 87.87% | 41.53% |
| 10k model | 88.35% | 86.63% | **98.92%** | 81.80% |
| 298k model | 95.04% | 95.27% | 93.63% | 93.20% |
| 378k model | 95.24% | 95.87% | 91.40% | 94.35% |
| 450k model | **95.43%** | **96.09%** | 91.38% | **94.58%** |
| Final 711,135 model | 95.11% | 95.66% | 91.76% | 94.35% |

| Checkpoint | CARD | DATE | DEC | DIGIT | KEEP | MEAS | MONEY | MULTI | ORD | PHONE | PUNCT | TIME |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 298k | 94.9% | 96.6% | 95.2% | 95.9% | **93.6%** | 94.7% | 98.9% | 93.2% | 97.4% | 94.4% | 93.3% | 96.7% |
| 378k | 94.6% | 96.1% | **96.7%** | 97.5% | 91.4% | 94.6% | **99.1%** | 94.4% | **98.5%** | 97.1% | **96.2%** | 96.7% |
| **450k** | **95.1%** | **97.2%** | 95.4% | 97.9% | 91.4% | **95.2%** | 99.0% | **94.6%** | 98.2% | **97.5%** | 95.2% | **97.3%** |
| 711,135 | 93.5% | 96.1% | 95.9% | **98.3%** | 91.8% | 94.9% | 98.9% | 94.3% | 98.2% | 96.2% | 95.2% | 96.4% |

The final checkpoint improves Google validation exact accuracy by 6.76 points
over the 10k model. Positive accuracy improves by 9.03 points and MULTI by
12.55 points. KEEP falls by 7.16 points, which confirms the expected
over-normalization cost of training the remaining data with no additional KEEP
examples.

## Checkpoint choice

- Use **378k** as the base checkpoint for combined conversational adaptation.
- Keep 450k as the Google/Numb3rs reference and 298k as the SLURP/Taskmaster
  reference.
- Do not use the final checkpoint as the adaptation base; it is dominated by
  378k on Golden, Numb3rs, Google validation, and conversational validation.
- Do not select a production checkpoint until the conversational adaptation
  checkpoints pass all four validation suites.

All raw files are under
`data/generated/full_google_train/evaluations/{benchmark}_{exposure}/metrics.json`.
