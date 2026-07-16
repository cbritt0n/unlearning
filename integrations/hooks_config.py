"""
Build workflow hooks from environment variables.

| Variable | Role |
|----------|------|
| HEALER_WEBHOOK_DOCUMENT_STORE | URL for document-store delete |
| HEALER_WEBHOOK_BACKUP_ACK | URL for backup policy ack |
| HEALER_WEBHOOK_API_KEY | Optional X-API-Key for webhooks |
| HEALER_CRYPTO_SHRED | ``1`` enable in-process CryptoShredVault hook |
| HEALER_OUTBOX_REPLICA | ``1`` use file outbox for replica step |
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from integrations.hooks import (
    HttpWebhookHook,
    LocalBackupAckHook,
    make_crypto_shred_hook,
)
from integrations.outbox import FileOutbox, make_outbox_replica_hook

logger = logging.getLogger(__name__)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def build_hooks_from_env(
    data_dir: str | Path,
) -> dict[str, Any]:
    """
    Return kwargs for ``ErasureWorkflowRunner``: document_store, crypto_shred,
    replica_fanout, backup_ack (any may be None).
    """
    hooks: dict[str, Any] = {
        "document_store": None,
        "crypto_shred": None,
        "replica_fanout": None,
        "backup_ack": None,
    }
    api_key = os.environ.get("HEALER_WEBHOOK_API_KEY") or None

    doc_url = os.environ.get("HEALER_WEBHOOK_DOCUMENT_STORE", "").strip()
    if doc_url:
        wh = HttpWebhookHook(
            doc_url, event="document_store_delete", api_key=api_key
        )
        hooks["document_store"] = wh.as_document_store_hook()
        logger.info("document_store webhook → %s", doc_url)

    bak_url = os.environ.get("HEALER_WEBHOOK_BACKUP_ACK", "").strip()
    if bak_url:
        wh = HttpWebhookHook(
            bak_url, event="backup_ack", api_key=api_key
        )
        hooks["backup_ack"] = wh.as_backup_ack_hook()
        logger.info("backup_ack webhook → %s", bak_url)
    elif env_flag("HEALER_LOCAL_BACKUP_ACK"):
        hooks["backup_ack"] = LocalBackupAckHook(auto_ack=True)
        logger.info("backup_ack using LocalBackupAckHook (dev)")

    if env_flag("HEALER_CRYPTO_SHRED"):
        from compliance.crypto_shred import CryptoShredVault

        vault = CryptoShredVault()
        hooks["crypto_shred"] = make_crypto_shred_hook(vault)
        logger.info("crypto_shred vault hook enabled")

    if env_flag("HEALER_OUTBOX_REPLICA"):
        outbox = FileOutbox(data_dir)
        hooks["replica_fanout"] = make_outbox_replica_hook(outbox)
        logger.info("replica_fanout → file outbox under %s", data_dir)

    return hooks
