# Realizer coverage

`premove-itn` exposes deterministic English inverse text normalization (ITN).
Each explicit kind consumes the complete input and returns one semantic value,
or `None` when the input is not valid for that kind. Rust owns the realization
rules. The implementation reuses the upstream
[`text-processing-rs`](https://github.com/FluidInference/text-processing-rs)
v0.3.0 parsers. Local Rust code adds grammar, complete-span validation, and
canonicalization only where the upstream parser is not sufficient; it is not a
second Python realizer or a fork of the upstream library. A local parser may
run before the upstream parser for a kind with additional supported forms,
while delegated kinds retain the upstream parser after local guards pass.

## Supported kinds

| Kind | Forms handled |
| --- | --- |
| `DIGIT_SEQUENCE` | Spoken or numeric digit tokens, leading zeroes, `oh`/`o`/`nought`/`naught`/`nil`, `single`/`double`/`triple`/`quadruple` repetition, and unambiguous multi-digit grouped ID readings such as `twenty-three forty-five`. |
| `CARDINAL` | Signed cardinal words (including explicit positive signs), local scales through undecillion, digit-sequence fallback, and a distinct aviation reading in `realize_options`. |
| `DATE` | Month-first and day-first dates, weekdays, short or split years, plural years and centuries, eras, and calendar validity checks. |
| `TIME` | 12- and 24-hour clocks, zero aliases, military/hundred forms, `o'clock`, AM/PM phrases, relative times, `midnight`/`noon`, durations with milliseconds, and recognized timezone suffixes and offsets. |
| `MONEY` | Major and minor currency names, `buck`/`bucks` and `quid`, singular/plural and hyphenated spoken forms, signs, currency placement, grouping, decimal and mixed-fraction values, and thousand/lakh/crore/million/billion/trillion scales across common ISO and legacy currency names. |
| `DECIMAL` | Signed integers and fractions, spoken digit fractions, named scales through undecillion, and scientific notation. |
| `MEASUREMENT` | Signed decimal quantities, spoken `point`/`dot`, long-number scales, fractions and mixed fractions, bare degrees, feet-and-inches heights, metric/data/power units, compound units (`per`, square/cubic units), and strict complete-span validation. Every spoken alias passes through one case-sensitive canonical unit renderer before runtime output. |
| `ORDINAL` | Optional `the`, hyphenated or conjunctive words, ordinal scales through undecillionth, numeric suffixes, and canonical Roman numerals, with complete-span validation. |
| `PHONE` | Spoken phone, `area code` prefixes, explicit `dash`/`hyphen` separators, extensions, single-letter serial, IP, and SSN forms, `double`/`triple` digits, country-code prefixes, and corpus `sil` separators, with complete-span validation. Multi-word text beside a separator is rejected. |
| `PUNCTUATION` | Common spoken punctuation aliases, paired delimiters, open/close quote symbol candidates, ASCII symbols, ellipses, and distinct en/em dashes, with complete-span validation. |
| `WHITELIST` | Sentence-level approved abbreviation and phrase replacements with case preservation, word-boundary checks, and fail-closed handling for non-ASCII or ambiguous input. |
| `WORD` | Mixed spoken ASCII letters and numbers such as `B two B`, leading-zero digit groups inside identifiers, grouped numeric chunks such as `twenty twenty three A`, explicitly spoken `dash`/`hyphen`/`slash` separators, multi-part versions such as `one point zero point seven`, canonical `v`-prefixed versions such as `v three dot one dot nine`, numbers with one attached ASCII or common Unicode punctuation mark, and complete-span validation. |
| `ELECTRONIC` | Email, domain, protocol URL, and path forms delegated to the corresponding upstream English parser, with local handling for spoken `underscore` and `plus`, uppercase `O` inside spelled-letter runs, and kind-specific guards where documented below. Spoken `oh` and lowercase `o` retain digit-zero semantics. |

The explicit ELECTRONIC and PHONE entry points add complete-span guards before
delegating. ELECTRONIC also validates protocol/domain structure and the
delegated output. Unknown trailing words are rejected instead of being
silently ignored or appended to the normalized value.

`open quote` and `close quote` each realize to a quote-symbol candidate. The
realizer does not define sentence-level opening and closing quote spacing.
Structural whitespace commands such as `new line` and `new paragraph` are not
runtime candidates because the scorer and path renderer do not yet represent
whitespace-only replacements.

## Representation comparison

`representations_equivalent` is evaluation-only. It compares semantic values
without adding display aliases to runtime candidates. It currently supports:

- `CARDINAL`: grouping, zero padding, Roman numerals, and signed zero.
- `DATE`: unambiguous field order and separators, month or weekday
  abbreviations, ordinal suffixes, and era punctuation, including eras on full
  dates.
- `DECIMAL`: grouping, decimal padding, decimal comma, named scales, attached
  scale suffixes, scientific notation, preserved negative zero, and bounded
  canonical expansion.
- `DIGIT_SEQUENCE`: harmless grouping punctuation and separators are ignored
  while the digit order and value are preserved.
- `TIME`: clock padding and separators, compact clocks, AM/PM case and
  punctuation, 12/24-hour notation, timezone case and offset padding, and
  duration or fraction padding.
- `MONEY`: currency symbols, ISO codes, currency placement, grouping, decimal
  zero padding, major/minor units, and named or abbreviated thousand/lakh/crore/
  million/billion/trillion scales. Currency identity is preserved: `$` maps to
  USD, while explicit CAD/AUD and other currency codes remain distinct.
- `MEASUREMENT`: numeric grouping, decimal padding, Unicode minus/spacing,
  unit symbols and names, square/cubic aliases, compound rate units, and
  metric/data/power unit aliases. Numeric value and canonical unit must both
  match. Case-sensitive unit identities remain distinct, including meter `m`
  versus minute `min`, mega versus milli prefixes, and byte versus bit units.
- `ORDINAL`: numeric suffixes, ordinal words and scales, optional articles,
  Roman numerals, and harmless terminal punctuation.
- `PHONE`: separator/grouping punctuation, country-code spacing, and SSN or
  IP structure are normalized while the digit sequence and structure remain
  significant.

The seven-shard Google TN TIME audit covers 51,569 rows. The current result is
99.994183% exact or representation-equivalent; the three remaining rows are
inconsistent dataset annotations.

The identity-aware seven-shard Google TN MONEY audit covers 214,728 rows. The
current result is 21,239 exact plus 191,379 representation-equivalent rows.
The 2,047 mismatches are underdetermined annotations where a generic spoken
`dollar`/`cent` is paired with an explicit non-USD target such as `A$` or `€`;
the source does not contain enough information to infer that currency. The 63
`None` rows are malformed `sil`/`pa` annotations. The remaining 212,618
determinate, non-malformed rows have 100% exact-or-equivalent coverage.

The seven-shard audits also cover 4,761,909 CARDINAL rows (99.913% exact or
equivalent), 12,029,272 DATE rows (99.995%), and 350,825 DECIMAL rows
(99.982%). Their remaining rows are malformed, contradictory, or carry target
annotation suffixes that are not part of the semantic value.

These corpus audits are regression evidence, not the source of the grammar.
Rules must be expressed in terms of generic numeric, date, time, unit, and
span structure. A sentence, URL, annotation token, or other dataset-specific
string must not receive a special case.

Synthetic checks cover signed and zero forms, malformed numeric separators,
balanced digit grouping, bounded scientific expansion, overflow and leap-day
rejection, grouping and locale decimal separators, repeated-dot grouping,
Unicode spacing, invalid timezone offsets, repeated scales, and unrelated
suffixes. These checks are kept alongside the focused Rust tests so future
kinds receive the same complete-span and formatting review.

Update this document and the README when a kind gains or loses behavior. Add a
focused Rust test for each new form.
