# Changelog

All notable changes to Premove ITN are documented here. The project follows
semantic versioning for the Python package and Rust extension.

## [Unreleased]

### Documentation

- Moved the complete v0.2.0 targeted comparison out of the README into a
  dedicated evaluation document, with methodology, reproduction steps, and
  limitations.
- Added a concise release summary and representative examples to the README
  and this changelog.

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

[Unreleased]: https://github.com/premove-ai/premove-itn/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/premove-ai/premove-itn/releases/tag/v0.2.0
