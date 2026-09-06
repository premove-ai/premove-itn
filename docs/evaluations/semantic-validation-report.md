# Semantic evaluation report

## Scope

This report adds kind-aware semantic accuracy to the two new external
validation sets: Apple PolyNorm-Bench en-US and VoiceCodeBench. Exact accuracy
is retained for formatting audit. Semantic equality is accepted only when a
Rust value parser proves equality for the declared kind.

Supported semantic comparators are `CARDINAL`, `DATE`, `DECIMAL`,
`DIGIT_SEQUENCE`, `MEASUREMENT`, `MONEY`, `ORDINAL`, `PHONE`, and `TIME`.
Categories without one of these mappings still require exact equality. Their
scores are conservative and must not be described as full semantic accuracy.

## Results

### Apple PolyNorm-Bench en-US

| System | Exact, all 540 | Semantic, all 540 | Semantic, mapped 140 |
| --- | ---: | ---: | ---: |
| Deterministic Rust | 92/540 = 17.04% | 139/540 = 25.74% | 108/140 = 77.14% |
| 378k baseline | 113/540 = 20.93% | **153/540 = 28.33%** | **116/140 = 82.86%** |
| 378k + 20k | **115/540 = 21.30%** | 152/540 = 28.15% | **116/140 = 82.86%** |

The mapped subset contains Cardinal, Currency, Decimal, Ordinal, Phone Number,
Time, and Unit. The remaining PolyNorm categories use exact fallback because
Premove does not have safe semantic comparators for them. The 378k model's
one-row overall lead is in an exact-fallback category; both checkpoints tie on
the mapped semantic subset.

### VoiceCodeBench entity component evaluation

| System | Exact, all 1,482 | Semantic, all 1,482 | Semantic, mapped 545 |
| --- | ---: | ---: | ---: |
| Deterministic Rust | **333/1,482 = 22.47%** | **521/1,482 = 35.16%** | **378/545 = 69.36%** |
| 378k baseline | 298/1,482 = 20.11% | 407/1,482 = 27.46% | 280/545 = 51.38% |
| 378k + 20k | **378/1,482 = 25.51%** | **476/1,482 = 32.12%** | **319/545 = 58.53%** |

The mapped subset contains currency, date, IP address, measurement, percentage,
phone extension, phone number, plain number, port number, and time. The test
normalizes each published acoustic entity in isolation. It is a Premove
component test, not VoiceCodeBench's official raw-audio ASR protocol. The
model's lack of sentence context depresses selection accuracy; deterministic
Rust therefore leads on mapped semantic values.

The PHONE semantic comparator now treats `ext4821`, `x4821`, and
`extension 4821` as equal. This accounts for valid extension formatting without
adding runtime candidates.

## Valid VoiceCodeBench Rust gaps

The following inventory evaluates the full entity input with its intended
existing Rust kind. It counts unchanged input when it is already semantically
equal. It excludes formatting differences that a semantic comparator proves.

| Area | Valid uncovered rows | What is missing |
| --- | ---: | --- |
| Phone numbers | 37/60 | `area code` prefixes and spoken `dash` inside phone input. |
| Phone extensions | 9/30 | Spoken `x` prefix; the other 21 are semantically covered. |
| Time | 60/60 | Time ranges and bare `Eastern`/`Central`/`Mountain`/`Pacific` timezone aliases. |
| Measurements | 8/61 | `ng/mL`, `mmol/L`, measurement ranges, and a small number of uncovered compound units. |
| Versions | 30/30 | Attached prefixes such as `v3 dot 1 dot 9`; the new version grammar currently requires a numeric first token. |
| Account or record numbers | 13/65 | Mask symbols and leading zero preservation inside separated groups. |
| Product codes | 13/90 | Leading zero preservation inside code groups. |
| Reference IDs | 31/150 | Leading zero preservation and a few structured numeric group forms. |
| Email addresses | 45/65 after case-only matches | Spelled `O` is often interpreted as zero, plus seven forms with no electronic candidate. |
| URLs | 1/62 after case-fold diagnostic | One genuine token realization miss. Forty other rows depend on path casing that the speech does not always specify, so they are not safe grammar targets. |
| CLI flags | 43/44 | `dash` and `double dash`; one row contains an explicit correction and is excluded. |
| Environment variables | 35/35 | `all caps` plus spoken underscores. |
| File paths | 51/51 | General spoken path punctuation and filename extensions. |
| Code symbols | 34/35 | Spoken underscore/case conventions; one correction row is excluded. |

These are not Rust gaps:

- all 75 currency values, 89 dates, 25 IP addresses, 50 percentages, and 65
  plain numbers are already semantically covered;
- all 30 port entities include the contextual label `port` in the acoustic
  span but omit it from the canonical entity; removing that label belongs to
  entity extraction or argument binding;
- 49 composite shell-command rows require command parsing rather than one ITN
  realization;
- person or team names, postal addresses, domain terms, and ordinary spelled
  sequences are outside the numeric and structured-value Rust scope;
- unspoken URL path capitalization is underdetermined and must not become a
  hard-coded normalization rule.

## Priority

For the launch target, implement the gaps in this order:

1. preserve zero-padded groups in identifiers and prefixed versions;
2. accept `area code`, spoken phone separators, and `x` extensions;
3. add CLI flags, environment variables, and common file-path punctuation;
4. resolve spelled `O` versus zero in ELECTRONIC without corrupting valid
   addresses;
5. add single time timezone aliases, then decide whether time ranges belong in
   the core realizer;
6. add the eight measurement forms.

Do not add fractions, ISBN-specific rules, route identifiers, chemical formulas,
or port-label deletion for this launch stage.
