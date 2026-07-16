# Workflow hooks

Optional steps after live hard-delete. Configure via env or pass callables into
`ErasureWorkflowRunner`.

## Environment

| Variable | Effect |
|----------|--------|
| `HEALER_WEBHOOK_DOCUMENT_STORE` | POST document-store delete webhook |
| `HEALER_WEBHOOK_BACKUP_ACK` | POST backup policy ack webhook |
| `HEALER_WEBHOOK_API_KEY` | Optional `X-API-Key` on webhooks |
| `HEALER_LOCAL_BACKUP_ACK=1` | Dev-only auto backup ack (no HTTP) |
| `HEALER_CRYPTO_SHRED=1` | In-process `CryptoShredVault` shred hook |
| `HEALER_OUTBOX_REPLICA=1` | Enqueue replica intents to file outbox |

## Webhook body

```json
{
  "event": "document_store_delete",
  "collection": "docs",
  "ids": ["user-42"],
  "request_id": "ticket-1",
  "reason": "gdpr_art_17",
  "timestamp_unix_ns": 0
}
```

Events: `document_store_delete`, `backup_ack`.

## Code wiring

```python
from integrations.hooks import HttpWebhookHook, make_crypto_shred_hook
from compliance.crypto_shred import CryptoShredVault
from integrations.workflow import ErasureWorkflowRunner, ErasureWorkflowStore

doc = HttpWebhookHook(
    "https://app.example/hooks/erase-docs",
    event="document_store_delete",
    api_key="…",
)
vault = CryptoShredVault()
runner = ErasureWorkflowRunner(
    store,
    erase_service,
    document_store=doc.as_document_store_hook(),
    crypto_shred=make_crypto_shred_hook(vault),
)
wf = runner.create(
    "docs", ["user-42"],
    require_document_store=True,
    require_crypto_shred=True,
)
runner.advance(wf.request_id)
```

## Outbox replica path

```bash
export HEALER_OUTBOX_REPLICA=1
# after workflow with require_replica=true:
curl -X POST http://127.0.0.1:8000/v1/outbox/dispatch
curl http://127.0.0.1:8000/v1/outbox/pending
```

Production: replace `OutboxDispatcher` apply function with
`ReplicaFanoutCoordinator` / SQS consumer.
