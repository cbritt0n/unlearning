"""
KMS-backed crypto-shred backends
================================

``CryptoShredVault`` can store DEKs in-process for demos. Production should
envelope-encrypt DEKs under a cloud KMS or HashiCorp Vault and **destroy**
(or disable) the envelope on shred.

This module defines:

* ``KeyManagementService`` protocol
* ``LocalFileKMS`` — file-backed envelope keys for tests / air-gapped labs
* ``AwsKmsBackend`` — AWS KMS (optional ``boto3``)
* ``GcpKmsBackend`` — GCP Cloud KMS (optional ``google-cloud-kms``)
* ``VaultTransitBackend`` — Vault transit secrets engine (optional ``hvac``)
* ``KmsCryptoShredVault`` — CryptoShredVault drop-in using a KMS backend
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from compliance.crypto_shred import CryptoShredVault, ShredReceipt

logger = logging.getLogger(__name__)


@runtime_checkable
class KeyManagementService(Protocol):
    """Minimal envelope KMS surface for DEK wrap/unwrap/destroy."""

    def wrap_dek(self, entity_id: str, dek: bytes) -> bytes:
        """Encrypt DEK under KMS; return opaque blob to store beside ciphertext."""
        ...

    def unwrap_dek(self, entity_id: str, wrapped: bytes) -> bytes:
        """Decrypt DEK; must fail after destroy."""
        ...

    def destroy_entity_key(self, entity_id: str) -> None:
        """
        Irreversibly prevent future unwrap for this entity.

        Cloud KMS often schedules CMK deletion; here we destroy per-entity
        wrapping material (data key or alias binding).
        """
        ...


# ---------------------------------------------------------------------------
# Local file KMS (tests / offline)
# ---------------------------------------------------------------------------


class LocalFileKMS:
    """
    AES-GCM master key stored on disk (or generated ephemerally).

    Per-entity wrap: master encrypts DEK with AAD=entity_id.
    Destroy: delete wrapped DEK record so unwrap fails (master remains).
    """

    def __init__(self, path: str | Path | None = None, *, master_key: bytes | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        if master_key is not None:
            if len(master_key) != 32:
                raise ValueError("master_key must be 32 bytes")
            self._master = master_key
        elif self.path and self.path.is_file():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._master = base64.b64decode(raw["master"])
            self._wrapped: dict[str, str] = dict(raw.get("wrapped", {}))
            self._destroyed: set[str] = set(raw.get("destroyed", []))
            return
        else:
            self._master = secrets.token_bytes(32)
        self._wrapped = {}
        self._destroyed: set[str] = set()
        self._persist()

    def wrap_dek(self, entity_id: str, dek: bytes) -> bytes:
        eid = str(entity_id)
        with self._lock:
            if eid in self._destroyed:
                raise RuntimeError(f"entity {eid!r} shredded")
            nonce = secrets.token_bytes(12)
            ct = AESGCM(self._master).encrypt(nonce, dek, eid.encode())
            blob = nonce + ct
            self._wrapped[eid] = base64.b64encode(blob).decode("ascii")
            self._persist()
            return blob

    def unwrap_dek(self, entity_id: str, wrapped: bytes) -> bytes:
        eid = str(entity_id)
        with self._lock:
            if eid in self._destroyed:
                raise RuntimeError(f"entity {eid!r} shredded")
            # Prefer stored wrap if present
            if eid in self._wrapped:
                wrapped = base64.b64decode(self._wrapped[eid])
            if len(wrapped) < 13:
                raise ValueError("wrapped DEK too short")
            nonce, ct = wrapped[:12], wrapped[12:]
            return AESGCM(self._master).decrypt(nonce, ct, eid.encode())

    def destroy_entity_key(self, entity_id: str) -> None:
        eid = str(entity_id)
        with self._lock:
            self._wrapped.pop(eid, None)
            self._destroyed.add(eid)
            self._persist()

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "master": base64.b64encode(self._master).decode("ascii"),
            "wrapped": dict(self._wrapped),
            "destroyed": sorted(self._destroyed),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Cloud backends (optional SDKs)
# ---------------------------------------------------------------------------


class AwsKmsBackend:
    """
    AWS KMS envelope encryption.

    Requires ``boto3`` and IAM permission for Encrypt/Decrypt on ``key_id``.
    Per-entity destroy stores a local denylist (CMK deletion is account-wide;
    production should use grants or per-tenant CMKs).
    """

    def __init__(self, key_id: str, *, region: str | None = None) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "boto3 required for AwsKmsBackend: pip install boto3"
            ) from exc
        self.key_id = key_id
        self._client = boto3.client("kms", region_name=region or os.environ.get("AWS_REGION"))
        self._destroyed: set[str] = set()
        self._lock = threading.Lock()
        self._cache: dict[str, bytes] = {}

    def wrap_dek(self, entity_id: str, dek: bytes) -> bytes:
        with self._lock:
            if entity_id in self._destroyed:
                raise RuntimeError(f"entity {entity_id!r} shredded")
        resp = self._client.encrypt(
            KeyId=self.key_id,
            Plaintext=dek,
            EncryptionContext={"entity_id": str(entity_id)},
        )
        blob = resp["CiphertextBlob"]
        with self._lock:
            self._cache[str(entity_id)] = blob
        return blob

    def unwrap_dek(self, entity_id: str, wrapped: bytes) -> bytes:
        with self._lock:
            if entity_id in self._destroyed:
                raise RuntimeError(f"entity {entity_id!r} shredded")
        resp = self._client.decrypt(
            CiphertextBlob=wrapped,
            EncryptionContext={"entity_id": str(entity_id)},
        )
        return resp["Plaintext"]

    def destroy_entity_key(self, entity_id: str) -> None:
        with self._lock:
            self._destroyed.add(str(entity_id))
            self._cache.pop(str(entity_id), None)
        logger.info("AWS KMS: entity %s marked shredded (local denylist)", entity_id)


class GcpKmsBackend:
    """GCP Cloud KMS CryptoKey encrypt/decrypt (optional google-cloud-kms)."""

    def __init__(self, key_name: str) -> None:
        """
        Parameters
        ----------
        key_name:
            Full resource name, e.g.
            ``projects/P/locations/L/keyRings/R/cryptoKeys/K``.
        """
        try:
            from google.cloud import kms

            self._client = kms.KeyManagementServiceClient()
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "google-cloud-kms required: pip install google-cloud-kms"
            ) from exc
        self.key_name = key_name
        self._destroyed: set[str] = set()
        self._lock = threading.Lock()

    def wrap_dek(self, entity_id: str, dek: bytes) -> bytes:
        with self._lock:
            if entity_id in self._destroyed:
                raise RuntimeError(f"entity {entity_id!r} shredded")
        # AAD via additional_authenticated_data
        resp = self._client.encrypt(
            request={
                "name": self.key_name,
                "plaintext": dek,
                "additional_authenticated_data": str(entity_id).encode(),
            }
        )
        return resp.ciphertext

    def unwrap_dek(self, entity_id: str, wrapped: bytes) -> bytes:
        with self._lock:
            if entity_id in self._destroyed:
                raise RuntimeError(f"entity {entity_id!r} shredded")
        resp = self._client.decrypt(
            request={
                "name": self.key_name,
                "ciphertext": wrapped,
                "additional_authenticated_data": str(entity_id).encode(),
            }
        )
        return resp.plaintext

    def destroy_entity_key(self, entity_id: str) -> None:
        with self._lock:
            self._destroyed.add(str(entity_id))


class VaultTransitBackend:
    """
    HashiCorp Vault Transit secrets engine.

    Uses ``transit/encrypt/{key}`` and ``transit/decrypt/{key}`` with
    context = entity_id. Destroy adds entity to a denylist (or you can
    rotate/rewrite policy in Vault).
    """

    def __init__(
        self,
        key_name: str,
        *,
        url: str | None = None,
        token: str | None = None,
    ) -> None:
        try:
            import hvac
        except ImportError as exc:  # pragma: no cover
            raise ImportError("hvac required: pip install hvac") from exc
        self.key_name = key_name
        self._client = hvac.Client(
            url=url or os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200"),
            token=token or os.environ.get("VAULT_TOKEN"),
        )
        self._destroyed: set[str] = set()
        self._lock = threading.Lock()

    def wrap_dek(self, entity_id: str, dek: bytes) -> bytes:
        with self._lock:
            if entity_id in self._destroyed:
                raise RuntimeError(f"entity {entity_id!r} shredded")
        plaintext = base64.b64encode(dek).decode("ascii")
        resp = self._client.secrets.transit.encrypt_data(
            name=self.key_name,
            plaintext=plaintext,
            context=base64.b64encode(str(entity_id).encode()).decode("ascii"),
        )
        # Store ciphertext string as bytes
        return resp["data"]["ciphertext"].encode("utf-8")

    def unwrap_dek(self, entity_id: str, wrapped: bytes) -> bytes:
        with self._lock:
            if entity_id in self._destroyed:
                raise RuntimeError(f"entity {entity_id!r} shredded")
        ct = wrapped.decode("utf-8") if isinstance(wrapped, (bytes, bytearray)) else str(wrapped)
        resp = self._client.secrets.transit.decrypt_data(
            name=self.key_name,
            ciphertext=ct,
            context=base64.b64encode(str(entity_id).encode()).decode("ascii"),
        )
        return base64.b64decode(resp["data"]["plaintext"])

    def destroy_entity_key(self, entity_id: str) -> None:
        with self._lock:
            self._destroyed.add(str(entity_id))


# ---------------------------------------------------------------------------
# Vault that uses KMS for DEK storage
# ---------------------------------------------------------------------------


@dataclass
class _EntityRecord:
    wrapped_dek: bytes


class KmsCryptoShredVault(CryptoShredVault):
    """
    Crypto-shred vault that never keeps raw DEKs at rest.

    - provision: generate DEK, wrap with KMS, keep only wrapped blob in memory
    - encrypt/decrypt: unwrap DEK ephemerally
    - shred: KMS destroy_entity_key + drop local record
    """

    def __init__(self, kms: KeyManagementService) -> None:
        # Bypass parent key storage; we reimplement with KMS.
        self._kms = kms
        self._lock = threading.RLock()
        self._records: dict[str, _EntityRecord] = {}
        self._shredded: set[str] = set()
        self._signing_placeholder = b""

    def provision(self, entity_id: str) -> bytes:
        eid = str(entity_id).strip()
        if not eid:
            raise ValueError("entity_id required")
        with self._lock:
            if eid in self._shredded:
                raise RuntimeError(
                    f"entity {eid!r} was crypto-shredded; cannot re-provision"
                )
            if eid in self._records:
                return self._kms.unwrap_dek(eid, self._records[eid].wrapped_dek)
            dek = secrets.token_bytes(32)
            wrapped = self._kms.wrap_dek(eid, dek)
            self._records[eid] = _EntityRecord(wrapped_dek=wrapped)
            return dek

    def encrypt(
        self, entity_id: str, plaintext: bytes, *, aad: bytes = b""
    ) -> bytes:
        dek = self.provision(entity_id)
        try:
            nonce = secrets.token_bytes(12)
            ct = AESGCM(dek).encrypt(nonce, plaintext, aad or None)
            return nonce + ct
        finally:
            # Best-effort: drop local reference to raw DEK
            del dek

    def decrypt(
        self, entity_id: str, blob: bytes, *, aad: bytes = b""
    ) -> bytes:
        eid = str(entity_id).strip()
        with self._lock:
            if eid in self._shredded or eid not in self._records:
                raise RuntimeError(
                    f"crypto-shredded or unknown entity {eid!r}: cannot decrypt"
                )
            wrapped = self._records[eid].wrapped_dek
        dek = self._kms.unwrap_dek(eid, wrapped)
        try:
            if len(blob) < 13:
                raise ValueError("ciphertext too short")
            nonce, ct = blob[:12], blob[12:]
            return AESGCM(dek).decrypt(nonce, ct, aad or None)
        finally:
            del dek

    def shred(self, entity_id: str) -> ShredReceipt:
        eid = str(entity_id).strip()
        ts = time.time_ns()
        with self._lock:
            self._records.pop(eid, None)
            self._shredded.add(eid)
        try:
            self._kms.destroy_entity_key(eid)
            msg = "KMS entity key destroyed; local DEK record dropped"
        except Exception as exc:  # noqa: BLE001
            msg = f"local drop ok; KMS destroy error: {exc}"
        return ShredReceipt(
            entity_id=eid,
            shredded=True,
            timestamp_unix_ns=ts,
            message=msg,
        )

    def is_shredded(self, entity_id: str) -> bool:
        return str(entity_id).strip() in self._shredded
