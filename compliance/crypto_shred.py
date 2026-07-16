"""
Per-entity crypto-shredding
==========================

Physical zeroing of HNSW slots does not erase **backups**, object-store
replicas, or snapshots. A complementary control is to encrypt embeddings
under a **per-entity data key** and destroy that key on delete
("crypto-shred"). Without the key, residual ciphertext is useless to
Vec2Text.

This module provides a small envelope-encryption vault suitable for
application-level shredding. Wire it **in addition to** ``ErasureService``
hard delete, not instead of it.

Uses the ``cryptography`` package (already a core dependency).
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass
class ShredReceipt:
    entity_id: str
    shredded: bool
    timestamp_unix_ns: int
    message: str


class CryptoShredVault:
    """
    In-memory per-entity AES-GCM keys (demo / single-process).

    Production: replace ``_keys`` with KMS/HSM (AWS KMS, GCP KMS, Vault)
    and never persist raw DEKs alongside ciphertext.
    """

    def __init__(self, *, master_key: bytes | None = None) -> None:
        """
        Parameters
        ----------
        master_key:
            Optional 32-byte key used only as an app secret marker in this
            simplified vault. Entity DEKs are random 256-bit keys stored
            until shredded.
        """
        self._lock = threading.RLock()
        self._master = master_key or os.environ.get(
            "HEALER_CRYPTO_MASTER_KEY", ""
        )
        # entity_id -> 32-byte DEK
        self._keys: dict[str, bytes] = {}
        self._shredded: set[str] = set()

    def provision(self, entity_id: str) -> bytes:
        """Create (or return) a data-encryption key for ``entity_id``."""
        eid = str(entity_id).strip()
        if not eid:
            raise ValueError("entity_id required")
        with self._lock:
            if eid in self._shredded:
                raise RuntimeError(
                    f"entity {eid!r} was crypto-shredded; cannot re-provision "
                    "without explicit admin recovery policy"
                )
            if eid not in self._keys:
                self._keys[eid] = secrets.token_bytes(32)
            return self._keys[eid]

    def encrypt(
        self, entity_id: str, plaintext: bytes, *, aad: bytes = b""
    ) -> bytes:
        """
        Encrypt ``plaintext`` under the entity DEK.

        Wire format: ``nonce (12 bytes) || ciphertext+tag``.
        """
        key = self.provision(entity_id)
        nonce = secrets.token_bytes(12)
        aes = AESGCM(key)
        ct = aes.encrypt(nonce, plaintext, aad or None)
        return nonce + ct

    def decrypt(
        self, entity_id: str, blob: bytes, *, aad: bytes = b""
    ) -> bytes:
        eid = str(entity_id).strip()
        with self._lock:
            if eid in self._shredded or eid not in self._keys:
                raise RuntimeError(
                    f"crypto-shredded or unknown entity {eid!r}: cannot decrypt"
                )
            key = self._keys[eid]
        if len(blob) < 13:
            raise ValueError("ciphertext too short")
        nonce, ct = blob[:12], blob[12:]
        return AESGCM(key).decrypt(nonce, ct, aad or None)

    def encrypt_vector(
        self, entity_id: str, vector: list[float] | bytes
    ) -> bytes:
        if isinstance(vector, bytes):
            raw = vector
        else:
            import numpy as np

            raw = np.asarray(vector, dtype=np.float32).tobytes()
        return self.encrypt(entity_id, raw, aad=b"embedding/f32")

    def decrypt_vector(self, entity_id: str, blob: bytes):
        import numpy as np

        raw = self.decrypt(entity_id, blob, aad=b"embedding/f32")
        return np.frombuffer(raw, dtype=np.float32).copy()

    def shred(self, entity_id: str) -> ShredReceipt:
        """
        Destroy the entity DEK. Ciphertext becomes unrecoverable here.

        Call **after** or **with** physical HNSW wipe for defense in depth.
        """
        eid = str(entity_id).strip()
        ts = time.time_ns()
        with self._lock:
            had = eid in self._keys
            self._keys.pop(eid, None)
            self._shredded.add(eid)
        return ShredReceipt(
            entity_id=eid,
            shredded=True,
            timestamp_unix_ns=ts,
            message=(
                "DEK destroyed"
                if had
                else "no DEK present; marked shredded for fail-closed decrypt"
            ),
        )

    def is_shredded(self, entity_id: str) -> bool:
        return str(entity_id).strip() in self._shredded
