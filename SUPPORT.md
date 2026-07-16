# Support

This is alpha open source. We’ll help when we can; there’s no SLA unless you
have a separate commercial arrangement.

## Where to go

| Question | Place |
|----------|--------|
| How do I install / run a first delete? | [docs/INSTALL.md](docs/INSTALL.md), [docs/GOLDEN_PATH.md](docs/GOLDEN_PATH.md) |
| Benchmarks / hnswlib on Windows | [docs/BENCHMARKS.md](docs/BENCHMARKS.md), [docs/HNSWLIB_AND_BENCHMARKS.md](docs/HNSWLIB_AND_BENCHMARKS.md) |
| Something’s broken | GitHub Issues (use the bug template) |
| Feature idea | Feature-request issue |
| Possible security issue | [SECURITY.md](SECURITY.md) — keep it private |
| Design chat | Discussions, if we’ve turned them on |

## Before you open an issue

- Search existing issues so we don’t duplicate threads.
- Mention OS, Python version, and whether `import hnsw_healer` / `import hnswlib` works.
- A minimal local repro beats a giant app dump.

## Out of scope here

- Legal advice (“are we GDPR compliant?”)
- Fixing hosted SaaS vector DBs we can’t plug into
- Promising to wipe swap, core dumps, or every cloud snapshot
  (see [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md))

We don’t promise response times. Security reports get priority when we can.
