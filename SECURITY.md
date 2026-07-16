# Security

## Supported versions

| Version | Support |
|---------|---------|
| 0.3.x | Best effort (alpha) |
| older | Best effort only |

Alpha means: useful for careful pilots, not a guarantee. Validate against your
own threat model before you put real user data behind it.

## What we want to hear about

Please report:

- Receipts that say success while deleted embeddings are still recoverable
- Hard erase failing but soft-delete still running (fail-open)
- Auth bypass on delete / workflow routes when an API key is configured
- Receipt signature issues with a real `HEALER_SIGNING_KEY`
- Tenant isolation problems when multi-tenant mode is on

Usually **not** treated as security bugs:

- Soft-delete leaving residual floats — that’s the whole problem we document
- Not wiping swap, core dumps, or offline volume snapshots
  ([docs/THREAT_MODEL.md](docs/THREAT_MODEL.md))
- “Is this enough for GDPR?” as a legal question

## How to report

1. Prefer GitHub **Security Advisories** on the repo (private):  
   *Security → Report a vulnerability* (once the repo exists on GitHub).
2. Otherwise contact maintainers privately. Don’t post full exploit write-ups
   against systems you don’t own.

Include version/commit, which package area broke, steps on a **local** fixture
index, and whether residual floats remain after a “successful” receipt.

We’ll try to acknowledge within a week and sketch a fix plan within a month for
confirmed issues. Alpha resources are limited.

## Operators (short checklist)

- Don’t ship the default signing key; set `HEALER_SIGNING_KEY` and `HEALER_API_KEY`
- Use `HEALER_ENV=production` so the process refuses weak defaults
- Prefer wipe + compact/rebuild; treat heal as experimental
- Fan out deletes to replicas; plan backup retention or crypto-shred
- Keep receipts and residual samples in your own audit log
