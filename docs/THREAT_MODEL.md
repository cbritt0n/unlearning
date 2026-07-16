# Threat model — residual vectors & Vec2Text

## Assets

| Asset | Sensitivity |
|-------|-------------|
| Live embedding \(\mathbf{v} \in \mathbb{R}^d\) in HNSW RAM | High — invertible to text via Vec2Text-class models |
| On-disk index (`index.bin`, hnswlib file, DB segments) | High |
| WAL / logs containing ids + reasons | Medium (PII of *who* was deleted) |
| Backups / snapshots / replicas | High if they hold pre-delete floats |
| Application document store (raw text) | Out of scope of this library unless also deleted |

## Adversaries

1. **Honest-but-curious operator** with disk read on the vector node.
2. **Cloud storage reader** with access to volume snapshots.
3. **Insider** with API access but not full DB admin.
4. **Remote attacker** who only sees search API (weaker residual threat).

## Threat: soft-delete residual inversion

```
delete(user) → metadata tombstone only
             → v still in HNSW binary
             → attacker dumps floats
             → Vec2Text / inversion model → plaintext
```

### Mitigations in this project

| Control | Mechanism |
|---------|-----------|
| Physical wipe | `overwrite_vector` / `erase_node` zeros live floats |
| Graph heal | MN-RU so hard delete does not force “never hard-delete” |
| Durable commit | WAL + atomic `index.bin` so crash ≠ resurrection from old checkpoint |
| hnswlib compact | Rebuild index without deleted rows (adapter) |
| Crypto-shred | Destroy per-entity DEK so backup ciphertext is inert |
| Residual proofs | Automated zero + pattern-absence checks |

## Out of scope (explicit non-claims)

- Guaranteeing wipe of **swap**, **core dumps**, or **CPU caches**
- Erasing **third-party SaaS** vector products without an adapter
- Legal determination of GDPR “erasure” completeness
- Preventing inversion of **non-deleted** neighbors (inherent to embeddings)

## Residual matrix (honest)

| Surface | Wiped by hard delete? |
|---------|------------------------|
| Native `HNSWIndexProxy` RAM | Yes |
| Project `index.bin` after COMMIT | Yes (rewritten) |
| hnswlib after `mark_deleted` only | **No** — run `compact()` |
| hnswlib after `compact()` | Yes (rebuilt without row) |
| Chroma metadata delete only | **No** — use `ChromaHardDeleteCollection` |
| S3/EBS snapshots | **No** — retention policy + crypto-shred |
| Replica not receiving delete | **No** — fan-out deletes to all replicas |

See [BACKUPS_AND_REPLICAS.md](BACKUPS_AND_REPLICAS.md).
