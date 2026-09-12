---
this_file: WORK.md
---

# Issue 301 implementation

The workspace PLAN/TODO/WORK and research/301 contain the full audit trail.
This package owns the shared model-aware engine, UB1 format and in-file writes.
Tests preceded each feature and the cache, tokenizer and packaging regressions.
The latest full suite passes (287 tests, 5 legacy skips); final workspace runner
also checks the sdist-to-wheel build and isolated installed-package behavior.

Local Jina, Gemma, Granite, Nomic, MiniLM and Voyage nano were executed on M4 Max.
Live Jina/Voyage vendor calls and four exact self-hosted API parity checks passed.
Cloudflare/HF account failures remain documented, not counted as live successes.
A bilingual-fixture selection bug invalidated early quality reports; the corrected
reader has synthetic regression tests and all current reports identify its version.

Final post-review workspace runner passed; isolated rebuilt wheel passed.

## 2026-09-11 — Shared native workspace and translation memory

Consolidated native libraries into uubed-core, uubed-tm and uubed-python. Added
portable SQLite exact/pair storage, signed int8 search, streaming English TMX
import, atomic builds, model-space checks and `uubed tm` commands. Both book
alignment and Abersetz consume the shared Python model runtime. Removed duplicate
release runners, placeholder native tests and generated site output. Built-module
imports now use the actual uubed_native extension.

Verification is recorded in ../../research/consolidation: Rust/SIMD, built native
wheel, Python and consumer suites, real TMX retrieval, copied-book alignment,
recording-LLM integration and C API execution. No packages were published.

## Release and privacy cleanup

Implemented standalone tag-derived release commands, source/archive privacy gates, fresh artifact manifests, dry runs and same-tag retries. Verification is recorded in the workspace release report. Existing local data remains on disk.

### Release verification — 2026-09-12

300 tests passed, 5 skipped; v1.0.6 wheel/sdist preview verified. Shared release-flow suite: 11 passed. Source HEAD and latest tag remained unchanged. No live publication was performed.

## 2026-09-12 — License-file release validation

The shared archive gate now checks every License-File declaration in metadata
2.4+ against a regular file at the required sdist or wheel location. Missing,
misplaced and directory-only licenses fail before pushing or uploading. Synced
the helper across all six repositories. Verification: 14 release-tool tests
passed; nine existing sibling preview artifacts passed the stronger checks.
