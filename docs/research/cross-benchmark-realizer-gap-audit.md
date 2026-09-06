# Cross-benchmark Rust realizer gap audit

## Conclusion

The evaluations expose three types of coverage result. Golden has complete
exact candidate coverage. Google and conversational validation cannot measure
missing Rust coverage because their rows were admitted through the same
candidate graph. Numb3rs and PolyNorm are the independent gap sources.

The safe first change is structured `WORD` support for versions and identifiers.
It does not change the public kind inventory or either frozen checkpoint's
feature shape. Fraction, address, and chemical support require explicit scope
and model-compatibility decisions.

## Benchmark findings

| Evaluation | Rust coverage finding | Action |
| --- | --- | --- |
| Golden | 136/136 exact targets are reachable. | Keep as a regression set. |
| Google validation | Rows are constructed with `GoldGraph`; unreachable targets are excluded by construction. | Do not claim coverage from this set. Audit the unfiltered source when available. |
| Conversational validation | Rows are also built through `GoldGraph`. | Use it for scorer selection, not Rust coverage discovery. |
| NVIDIA Numb3rs | 4,690/10,131 exact. In mapped kinds, 7,930/7,985 are exact or semantically equivalent. There are 12 usable parser gaps, 884 unsupported fractions, 885 unsupported addresses, and 377 unsafe chemical `PLAIN` rows. | Add semantic metrics. Decide new kinds before changing the model schema. |
| Apple PolyNorm en-US | 150/540 exact targets were reachable before this change. The first Rust stage raises this to 161/540. Many remaining misses are reference formatting. | Continue auditing genuine semantic misses without inventing unspoken formatting. |

## Implemented in this stage

The `WORD` realizer now handles:

- multi-part spoken versions, such as `one point zero point seven` to `1.0.7`;
- grouped numeric chunks inside identifiers, such as `twenty twenty three A`
  to `2023A`;
- explicitly spoken `dash`, `hyphen`, and `slash` separators inside mixed
  alphanumeric identifiers.

The grammar rejects missing version components and leading, trailing, or
consecutive identifier separators. It does not infer punctuation that the
spoken input omits.

On PolyNorm, all 10 exactly recoverable version rows became reachable. The
378k baseline rose from 106 to 113 exact rows. The 378k + 20k checkpoint rose
from 108 to 115. Both models selected all 10 new version candidates correctly.

## Remaining gaps

1. `FRACTION` is the largest clear semantic gap. Adding it to `SPAN_KINDS`
   changes the scorer input shape and makes the frozen 378k and 378k + 20k
   checkpoints incompatible. Define checkpoint migration in an ADR first.
2. Fractional measurements depend on the same fraction representation. The
   confirmed Numb3rs example is `one and a half teaspoons`.
3. Numeric ISBN and code readings that use `hundred` or `thousand` need an
   `IDENTIFIER` versus `PHONE` scope decision. Arithmetic expansion can corrupt
   phone semantics if added globally.
4. Numb3rs route identifiers need an `ADDRESS` scope decision. They are mainly
   forms such as `I-55`, `SR 23`, and `C6`.
5. Numb3rs chemical formulas must not be implemented as `PLAIN`. Several
   identical spoken names have contradictory formulas, so the input does not
   determine the target.
6. PolyNorm phone, time, currency, and measurement misses often differ only in
   parentheses, separators, case, abbreviation, or punctuation. Add semantic
   comparison where possible instead of multiplying runtime candidates.

The detailed Numb3rs counts and examples are in
[`nvidia-numb3rs-unreachable-audit.md`](nvidia-numb3rs-unreachable-audit.md).
