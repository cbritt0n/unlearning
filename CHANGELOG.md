# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once 1.0.0 is reached. While in **0.x**, APIs may still change.

## [0.3.2] — 2026-07-16

### Added

* Residual-first benchmark suite: residual scan, recall retention, usable flag,
  delete wall-clock, scenarios **A–E**, `--backend hnswlib|native`
* Profiles: `gdpr_light`, `gdpr_batch`, `publish` (hnswlib-oriented)
* Adaptive delete strategy defaults (`HEALER_ADAPTIVE_COMPACT`, `HEALER_ALLOW_HEAL`)
* Qdrant / Weaviate rebuild-based adapters (in-memory clients for tests)
* Queue transports (file / Redis / SQS), workflow hooks, outbox, metrics,
  append-only receipt log, multi-tenant helpers
* Community docs: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue/PR templates

### Changed

* Product narrative: **wipe + compact/rebuild** recommended; MN-RU heal experimental
* README / BENCHMARKS emphasize residual risk over pure recall vs soft-delete

### Evaluation notes

* hnswlib `quick` (N=2k): soft residual=YES; rebuild residual=no at ~equal recall (~0.95)
* hnswlib `standard` (N=50k): at 10% delete, soft R@10≈0.349 vs rebuild ≈0.351 (residual no);
  at 50% delete soft≈0.470 residual YES vs rebuild≈0.447 residual no — see
  [docs/benchmarks/standard_hnswlib.md](docs/benchmarks/standard_hnswlib.md)
* Native synthetic graphs can show MN-RU heal collapse — stress only, not marketing baseline

## [0.3.1] — 2026-07-16

### Added

* Multi-engine adapters (Qdrant, Weaviate), delete strategy module, queue transports
* Benchmark scenario D rebuild; ENGINES.md

## [0.3.0] — 2026-07-16

### Added

* Receipt log, metrics, ingest API, workflow hooks, auth guards, coalesced compact

## [0.2.0] — 2026-07-16

### Added

* Receipt schema v2, auto-compact, residual proof fail-closed, golden path, workflow API

## [0.1.0] — prior

### Added

* Native wipe + MN-RU heal, WAL, FastAPI, hnswlib/Chroma adapters, residual proofs
