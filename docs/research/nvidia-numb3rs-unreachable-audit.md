# NVIDIA Numb3rs unreachable-row audit

## Conclusion

The 46.29% exact oracle is mostly an exact-rendering ceiling, not a semantic
normalization ceiling. Of the 5,441 exact-unreachable rows, 3,240 rows in
already-supported categories have a full-span candidate that the mapped
kind's existing `representations_equivalent` function accepts. Including the
4,690 exact-reachable rows gives a category-aware semantic oracle of
**7,930 / 10,131 = 78.27%** on the complete benchmark. Restricted to the nine
mapped categories, it is **7,930 / 7,985 = 99.31%**. Exact comparison rejects
differences such as `12:30 p.m.` versus `12:30 pm`, `$21999` versus `$21,999`,
and `0.1` versus `.1`.

A broader, diagnostic-only check across every candidate kind finds 3,800
semantically equivalent exact-unreachable rows, but that result must not be
used as the oracle: it admits incidental cross-kind matches, such as ADDRESS
rows interpreted by PHONE rules.

Do not add thousands of reference-specific spelling and punctuation variants.
Add fraction support first. Then add an explicit semantic evaluation metric.
Consider address and chemical-formula support only through architecture
decisions. Fix the small set of genuine parser gaps after these changes.

## Scope and sources

This audit uses the 10,131 rows in the locally pinned
`data/external/nvidia-numb3rs/192908075e1bd293914cc7b508f4a183ba6ef2b8/metadata.jsonl`.
Its SHA-256 is
`d6c6020fc8d3ccf8404395814307dbf0aca415305ed6964543e63fb3aa5df377`.
The source revision and field policy are recorded in
[`nvidia-numb3rs-source.md`](nvidia-numb3rs-source.md). NVIDIA's pinned
[dataset card](https://huggingface.co/datasets/nvidia/Numb3rs/blob/192908075e1bd293914cc7b508f4a183ba6ef2b8/README.md)
states that Numb3rs was derived from Google Text Normalization pairs and that
human reviewers retained the samples. The audit treats `text` as spoken input
and `original_text` as the exact written reference, as NVIDIA defines them.

The candidate graph and exact oracle come from
[`candidates.py`](../../src/premove_itn/candidates.py). The graph enumerates
every Rust realization over every token span. `build_gold_graph` accepts a row
only when candidate replacements and unchanged characters reproduce the target
character for character. Semantic equivalence comes from the Rust-backed
`representations_equivalent` API.

No production code or dataset row was changed.

## Method

For every row, the audit did the following:

1. Built all candidates with `build_candidate_graph(text)`.
2. marked the row exact-reachable only when
   `build_gold_graph(text, original_text, candidates)` returned a graph;
3. for exact-unreachable rows in a mapped category, selected full-input
   candidates of the mapped Premove kind and called
   `representations_equivalent(kind, candidate, original_text)`;
4. inspected the remaining 55 mapped rows against their complete full-span
   candidate sets and their source/reference pair;
5. grouped repeated `(category, text)` keys to find multiple exact references;
6. grouped reference shapes and manually reviewed every residual parser or
   annotation case.

The four primary classifications below are exclusive. The ambiguity analysis
is also reported separately because ambiguity overlaps unsupported categories
and otherwise-valid rendering variants.

## Primary classification of all 5,441 rows

| Primary cause | Rows | Share of unreachable | Meaning |
| --- | ---: | ---: | --- |
| Unsupported category | 2,146 | 39.44% | No explicit Premove mapping exists for `ADDRESS`, `FRACTION`, or `PLAIN`. |
| Missing exact realization form | 3,240 | 59.55% | A mapped-kind candidate is semantically equal, but its exact rendering differs. |
| Candidate parser/graph gap | 12 | 0.22% | The input/reference is usable, but the relevant parser does not preserve the required numeric structure. |
| Invalid, contradictory, or category-mismatched annotation | 43 | 0.79% | The written target cannot be inferred from the spoken input, contradicts it, or is not the declared semiotic class. |
| **Total** | **5,441** | **100.00%** | |

No defect was found in the dynamic-programming alignment itself. Every
confirmed graph-gap row lacks the necessary candidate before alignment. Twenty
additional rows become reachable after trimming the reference (`12` `DIGIT`
and `8` `CARDINAL` rows). Those rows are counted under exact-form or annotation
issues, not as an aligner defect, because the source reference contains the
extra whitespace.

## Category inventory

| Category | Unreachable | Mapped-kind semantic match | Residual |
| --- | ---: | ---: | ---: |
| `ADDRESS` | 885 | not mapped | 885 |
| `CARDINAL` | 78 | 76 | 2 |
| `DATE` | 494 | 494 | 0 |
| `DECIMAL` | 262 | 262 | 0 |
| `DIGIT` | 82 | 82 | 0 |
| `FRACTION` | 884 | not mapped | 884 |
| `MEASURE` | 225 | 216 | 9 |
| `MONEY` | 670 | 662 | 8 |
| `ORDINAL` | 9 | 9 | 0 |
| `PLAIN` | 377 | not mapped | 377 |
| `TELEPHONE` | 843 | 809 | 34 |
| `TIME` | 632 | 630 | 2 |
| **Total** | **5,441** | **3,240** | **2,201** |

Checking a full-span candidate under any of its kinds, rather than only the
declared category mapping, raises the semantic-match count to **3,800**. The
remaining 1,641 rows are `ADDRESS` 329, `CARDINAL` 2, `FRACTION` 884,
`MEASURE` 9, `MONEY` 8, `PLAIN` 377, `TELEPHONE` 30, and `TIME` 2. The 556
additional `ADDRESS` matches are incidental numeric interpretations. They do
not establish address support and are excluded from the category-aware
semantic oracle.

The category-aware semantic oracle is therefore:

| Scope | Reachable | Total | Coverage |
| --- | ---: | ---: | ---: |
| Complete Numb3rs benchmark | 7,930 | 10,131 | 78.27% |
| Nine mapped Premove categories | 7,930 | 7,985 | 99.31% |

## Unsupported categories

### `FRACTION`: add

All 884 fraction rows are unsupported. Most are regular and useful:

| Reference family | Rows | Example |
| --- | ---: | --- |
| Slash fraction | 756 | `fourteen one thousand nine hundred eighty eighths` -> `14/1988` |
| Unicode half | 51 | `seven and a half` -> `7½` |
| Mixed slash fraction | 33 | `two and a half` -> `2 1/2` |
| Negative fraction | 16 | `minus twelve one thousand nine hundred ninety thirds` -> `-12/1993` |
| Spaced slash | 9 | `two thousand sevenths` -> `2000 / 7` |
| Other Unicode or grouping forms | 19 | `one and an eighth` -> `1⅛` |

Add a `FRACTION` kind with one semantic numerator/denominator source of truth.
Generate a conservative canonical slash form first. Unicode glyphs and spacing
are presentation choices and should be semantic equivalents, not separate
learned meanings. Mixed fractions and negative fractions need explicit tests.

### `ADDRESS`: decide through an ADR

All 885 address rows are unsupported. They are mainly route and road
identifiers, not postal addresses: 531 have a compact letter-number shape such
as `c six` -> `C6`, 150 contain trailing whitespace, 123 have a spaced shape
such as `s r twenty three` -> `SR 23`, and 64 use a hyphen such as `i fifty
five` -> `I-55`.

A dedicated `ADDRESS` kind is reasonable only if road identifiers are in the
product scope. It needs explicit prefix vocabulary (`I`, `SR`, `US`) and a
canonical separator policy. Do not infer trailing spaces or punctuation. Do
not silently map these rows to `CARDINAL` or `DIGIT_SEQUENCE`.

### `PLAIN`: do not add as `PLAIN`

The 377 `PLAIN` rows are chemical-formula transformations, for example
`aluminium oxide` -> `Al2O3` and `aluminium sulphate` -> `Al2(SO4)3`. Many
spoken inputs map to several incompatible references. `aluminium chloride`
maps to `AlCl3`, `AlCl4`, `Al2Cl3`, and `Al2Cl6`. Several references also have
unbalanced parentheses, such as `CO)12`.

These are not identity/plain normalization. Do not add them to a `PLAIN`
realizer. If chemical formulas are required, define a separate `CHEMICAL` kind
and obtain data that speaks every stoichiometric index. Otherwise exclude this
category from the Premove exact oracle.

## Missing exact realization forms

The 3,240 mapped rows in this group already have the correct semantic value.
Their dominant differences are display policy:

- `MONEY`: digit grouping, scale words versus `M`/`m`/`K`, currency prefix,
  and capitalization. Example: `$21999` versus `$21,999`.
- `TIME`: colon versus dot, omitted `:00`, case, and dotted meridiem. Example:
  `12:30 p.m.` versus `12:30 pm`.
- `DATE`: month order, abbreviation, commas, and ISO rendering. Example:
  `July 25, 2015` versus `25 July 2015`.
- `DECIMAL`: leading zero, trailing zero, and thousands separators. Example:
  `0.1` versus `.1`.
- `TELEPHONE`: arbitrary hyphen and parenthesis grouping. Example: a plain
  digit candidate versus `0-88192-558-6`.
- `MEASURE`: unit spacing, superscript `²` versus `2`, abbreviations, and
  attached percent signs.
- `CARDINAL`, `DIGIT`, and `ORDINAL`: grouping commas, trailing source
  whitespace/punctuation, ordinal spacing, and `º` versus `th`.

Adding every observed format would inflate the candidate graph and still not
solve ambiguity. Report both exact and semantic oracle coverage. Use exact
match only for an explicitly chosen rendering policy. If Numb3rs exact match
must remain, add only deterministic, broadly useful variants such as grouped
cardinals and leading-dot decimals, with tests and candidate-count limits.

## Confirmed parser or candidate-graph gaps

Twelve usable rows lack a semantic candidate:

- One fractional measurement: `one and a half teaspoons` -> `1½ tsp`.
- Eleven ISBN/code-like `TELEPHONE` rows whose spoken grouping includes forms
  such as `hundred` or `thousand`, including `0-7007-0600-3`,
  `0-300-06906-5`, `1-4000-4005-1`, `978-85-200-0193-6`,
  `0-500-28795-3`, `0-500-05141-0`, `134-2000`, `08000-23999`, and
  `978-0-300-11465-2`.

Add the fraction parser before the fractional-measure composition. For phone
and identifier data, first decide whether ISBNs and numeric ranges belong to
`PHONE` or need an `IDENTIFIER` kind. Preserve digit groups; do not treat words
such as `hundred` as arithmetic unless the source convention requires it.

## Invalid, contradictory, or mislabeled residuals

The remaining 43 mapped rows should not drive new realization rules:

- Eight `MONEY` rows request `A$`, `HK$`, or `NT$` although the input says only
  “dollars.” The currency jurisdiction is absent from the input.
- Twenty-three `TELEPHONE` rows are years, eras, sports text, time zones, or
  opaque identifiers rather than telephone numbers. Examples include
  `one nine eight five eight six ahl` -> `1985-86 AHL`,
  `o nine hundred u t c` -> `0900 UTC`, and a full sports headline labeled
  `TELEPHONE`.
- Seven `MEASURE` rows write inches as two quote characters, for example
  `four inches` -> `4""`. One more maps `thirty seven minutes` to `37m`, which
  conflicts with the normal meter abbreviation.
- Two `CARDINAL` rows contradict or append absent content:
  `four` -> `40` and `three hundred eighty three` -> `383 U.S. `.
- Two `TIME` rows combine 24-hour values with `pm`:
  `eleven fifty five p m` -> `23:55 pm` and
  `five p m` -> `17.00 pm`.

These counts use a conservative rule. Opaque identifier rows that retain all
spoken symbols are classified as parser gaps; rows that inject a domain or
belong to another clear semiotic class are classified as annotation problems.

## Ambiguity and benchmark validity

There are 1,038 unreachable rows whose identical `(category, text)` key has
more than one exact reference elsewhere in the pinned dataset. This is an
overlapping quality flag, not an extra primary category.

| Category | Ambiguous unreachable rows |
| --- | ---: |
| `ADDRESS` | 312 |
| `TIME` | 283 |
| `PLAIN` | 144 |
| `MONEY` | 136 |
| `FRACTION` | 75 |
| `DIGIT` | 33 |
| `MEASURE` | 23 |
| `CARDINAL` | 18 |
| `ORDINAL` | 7 |
| `DATE` | 6 |
| `TELEPHONE` | 1 |

Examples include `one million dollars` with `$1 million`, `$1,000,000`, `$1M`,
and `$1m`; `eight a m` with several dotted, spaced, and `:00` forms; and
`three thousand` with `3000` and `3,000`. A deterministic system cannot infer
which duplicate reference the evaluator selected. These rows are valid as
semantic pairs but invalid as a single-reference test of exact rendering.

## Prioritized recommendations

1. Add semantic oracle and semantic accuracy beside exact accuracy. Keep exact
   metrics for continuity, but label them as reference-rendering metrics.
2. Add `FRACTION` with canonical slash output, mixed and negative fractions,
   and semantic equivalence across slash and Unicode presentations.
3. Compose fractions into `MEASUREMENT`, starting with `one and a half
   teaspoons`.
4. Exclude the 43 contradictory or mislabeled mapped rows from actionable
   error counts. Keep the frozen raw benchmark unchanged.
5. Decide `ADDRESS` scope in an ADR. If accepted, implement road identifiers
   with canonical prefixes and separators. Ignore source trailing whitespace.
6. Do not implement the current `PLAIN` rows as plain normalization. Define a
   `CHEMICAL` architecture only if the product needs it and the spoken form
   contains enough information.
7. Decide whether ISBNs and opaque numeric identifiers belong to `PHONE` or a
   new `IDENTIFIER` kind. Then add the eleven defensible grouping cases.
8. Do not add all Numb3rs punctuation, casing, grouping, or currency-prefix
   variants. They increase search space without adding recoverable semantics.

The immediate achievable improvement is therefore not “train longer.” It is
to measure semantic correctness, add fraction semantics, and make explicit
scope decisions for address, chemical, and identifier classes.
