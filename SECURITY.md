# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes (alpha — best effort) |
| &lt; 0.3 | Best effort only |

This project is **alpha**. Treat it as middleware you must validate in your
threat model before production use.

## What we care about

Priority issues:

* Hard-delete paths that leave **recoverable residual embeddings** when a
  receipt claims success
* Fail-open soft-delete when hard-erase fails
* Auth bypass on delete / workflow endpoints when `HEALER_API_KEY` is set
* Signature / receipt forgery when signing keys are configured correctly
* Path traversal or tenant isolation breaks under `HEALER_MULTI_TENANT`

Lower priority / out of scope for “vulnerability” reports:

* Soft-delete residual risk **by design** (that is the problem we document)
* Incomplete wipe of OS swap, core dumps, or offline volume snapshots
  (documented non-claims — see [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md))
* Legal determination of GDPR completeness

## Reporting a vulnerability

1. **Prefer** GitHub **Security Advisories** (private) on the repository once
   published: *Security → Report a vulnerability*.
2. If that is unavailable, open a **private** contact with maintainers without
   attaching full exploit chains against third-party systems.
3. Please include:
   * Affected version / commit
   * Component (`api`, `integrations`, native module, adapter)
   * Steps to reproduce on a **local** fixture index
   * Whether residual floats remain after a “successful” receipt

We aim to acknowledge within **7 days** and provide a remediation plan for
confirmed issues within **30 days** (alpha best effort).

## Safe harbor

Good-faith research against **your own** deployments or local fixtures is
welcome. Do not test against systems you do not own.

## Hardening checklist (operators)

* Set a strong `HEALER_SIGNING_KEY` (never the default in production)
* Set `HEALER_API_KEY` and `HEALER_ENV=production`
* Prefer wipe + compact/rebuild; treat MN-RU heal as experimental
* Fan-out deletes to all replicas; manage backup retention / crypto-shred
* Store erasure receipts and residual proof samples in your audit log
