# Realizer coverage

`premove-itn` exposes deterministic English inverse text normalization (ITN).
Each explicit kind consumes the complete input and returns one semantic value,
or `None` when the input is not valid for that kind. Rust owns the realization
rules. The upstream [`text-processing-rs`](https://github.com/FluidInference/text-processing-rs)
v0.3.0 parser handles kinds without a local extension.

## Supported kinds

| Kind | Forms handled |
| --- | --- |
| `DIGIT_SEQUENCE` | Spoken or numeric digit tokens, leading zeroes, `oh`/`o`/`nought`/`naught`/`nil`, and `single`/`double`/`triple`/`quadruple` repetition. |
| `CARDINAL` | Signed cardinal words, local scales through undecillion, digit-sequence fallback, and a distinct aviation reading in `realize_options`. |
| `DATE` | Month-first and day-first dates, weekdays, short or split years, plural years and centuries, eras, and calendar validity checks. |
| `TIME` | 12- and 24-hour clocks, zero aliases, military/hundred forms, `o'clock`, AM/PM phrases, relative times, `midnight`/`noon`, durations with milliseconds, and generic timezone suffixes and offsets. |
| `MONEY` | Major and minor currency names, singular/plural and hyphenated spoken forms, signs, currency placement, grouping, decimal values, and thousand/lakh/crore/million/billion/trillion scales across common ISO and legacy currency names. |
| `DECIMAL`, `ELECTRONIC`, `MEASUREMENT`, `ORDINAL`, `PHONE`, `PUNCTUATION`, `WHITELIST`, `WORD` | Delegated to the corresponding upstream English parser. |

## Representation comparison

`representations_equivalent` is evaluation-only. It compares semantic values
without adding display aliases to runtime candidates. It currently supports:

- `CARDINAL`: grouping, zero padding, Roman numerals, and signed zero.
- `DATE`: unambiguous field order and separators, month or weekday
  abbreviations, ordinal suffixes, and era punctuation.
- `TIME`: clock padding and separators, compact clocks, AM/PM case and
  punctuation, 12/24-hour notation, timezone case and offset padding, and
  duration or fraction padding.
- `MONEY`: currency symbols, ISO codes, currency placement, grouping, decimal
  zero padding, major/minor units, and named or abbreviated thousand/lakh/crore/
  million/billion/trillion scales. Currency identity is preserved: `$` maps to
  USD, while explicit CAD/AUD and other currency codes remain distinct.

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

Update this document and the README when a kind gains or loses behavior. Add a
focused Rust test for each new form.
