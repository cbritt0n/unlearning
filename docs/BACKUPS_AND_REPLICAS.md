# Backups, replicas, and crypto-shred

Physical wipe of the **live** HNSW node is necessary but **not sufficient**
if an attacker can restore yesterday’s snapshot.

## Replica fan-out

Every hard delete must reach **all** serving replicas:

```
App → ErasureService (region A) → backend.hard_delete
                ↓
         async job / message bus
                ↓
     region B & C ErasureService workers
```

Until all replicas ack, treat the delete as **in progress**. Search may still
hit a lagging replica that holds \(\mathbf{v}\).

### Suggested ack policy

1. WAL BEGIN on primary  
2. Local wipe + heal  
3. Replicate intent (request_id, collection, ids)  
4. Each replica wipe + local COMMIT  
5. Quorum → mark GDPR request complete  

## Backup lifecycle

| Backup type | Action on user erase |
|-------------|----------------------|
| Continuous volume snapshots | Encryption + key shred, or expire snapshots under retention that matches policy |
| Nightly index exports | Re-export without deleted ids; destroy old export objects |
| WAL / binlog | Ensure no full embedding payloads, or encrypt payloads under entity DEK |
| Object store documents | Separate document-store delete (out of band) |

**Never** claim erasure complete while a restorable backup still contains
plaintext floats for that user **unless** those backups are crypto-shredded.

## Crypto-shred pattern

```python
from compliance.crypto_shred import CryptoShredVault
from integrations import ErasureService

vault = CryptoShredVault()
# On write:
#   ct = vault.encrypt_vector(user_id, embedding)
#   store ct in cold storage; put plaintext embedding only in volatile index

# On delete:
receipt = erase_service.delete("users", [user_id], reason="gdpr")
shred = vault.shred(user_id)
# Cold ciphertext remains but is useless without DEK
```

Use a real **KMS** in production (`ScheduleKeyDeletion`, key rotation, dual control).

## Operational runbook (short)

1. Receive verified erasure request (`request_id`)  
2. `ErasureService.delete` (physical + heal)  
3. `compact()` / segment rewrite on ANN engine  
4. `CryptoShredVault.shred` / KMS delete  
5. Enqueue replica + backup-index jobs  
6. Store `ErasureReceipt` + shred receipt in audit log  
7. Run residual proof sample  
8. Close ticket when quorum + backup policy satisfied  
