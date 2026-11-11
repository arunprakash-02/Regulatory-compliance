# Changelog

All notable changes to this project.

## [1.2.0] — 2026-11-11

### Added
- **Definition boosting** — when queries ask for definitions, chunks receive a +0.30 boost.
- **Retry-on-refusal** — automated retry with expanded context.
- **Comprehensive evaluation harness v2** — 5 modules covering ingestion, retrieval, generation, extraction, and pipeline.
- **Wilson confidence intervals** on all rate metrics.

### Fixed
- **Cross-reference bug** — chunker splits only on line-start Article headers.
- **Mid-word cuts** — chunker snaps to word boundaries.
- **UTF-8 BOM** handling.
