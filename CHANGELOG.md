# Changelog

All notable changes to Premove ITN are documented here. The project follows
semantic versioning for the Python package and Rust extension.

## [Unreleased]

### Documentation

- No unreleased changes.

## [0.3.0] - 2026-09-20

### Added

- Added `normalize_structured()` for immutable normalized text, resolved text,
  and exact source/normalized spans.
- Added `normalize_resolved()` as the resolved-text view of the same inference
  result.
- Added caller-supplied `NormalizationContext` with timezone, locale, and all
  six `DateOrder` permutations.
- Added deterministic contextual resolution for relative dates, bounded day
  and week offsets, calendar-week weekdays, named dates, weekday-qualified
  dates, and numeric dates.

### Changed

- Temporal enrichment runs after the decoder and never performs a second model
  call, candidate build, decode, or alignment pass.
- Existing readable normalization remains the authoritative `text` view;
  semantic date values are exposed through `resolved_value` and
  `resolved_text`.
- The package release is v0.3.0. It continues to use the frozen v0.2.0 model
  artifact because this release changes the runtime API and deterministic
  temporal layer, not the trained scorer.

## [0.2.0] - 2026-09-14

### Changed

- Improved the contextual scorer's disambiguation of spoken times and numeric
  identifiers. On the targeted 10-row comparison, v0.2.0 matched 10/10
  intended outputs; v0.1.0 matched 2/10.
- Kept the deterministic Rust candidate layer and exact decoder as the source
  of valid output forms.

Representative corrections include:

- `desk six forty` → `desk 640` instead of `desk 06:40`.
- `unit two ten` in a technician-dispatch sentence → `unit 210` instead of
  `unit 02:10`.
- `twelve oh six` in `try locker twelve oh six, I'll be there at twelve oh six`
  → `1206` for the locker and `12:06` for the arrival time.

See the [v0.2.0 targeted release comparison](docs/evaluations/v0.2.0-targeted-release-comparison.md)
for the complete 10-row table, methodology, and limitations.

[Unreleased]: https://github.com/premove-ai/premove-itn/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/premove-ai/premove-itn/releases/tag/v0.3.0
[0.2.0]: https://github.com/premove-ai/premove-itn/releases/tag/v0.2.0
