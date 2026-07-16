"""Crypto-shred vault tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance.crypto_shred import CryptoShredVault


def test_encrypt_decrypt_vector() -> None:
    vault = CryptoShredVault()
    v = np.arange(8, dtype=np.float32)
    blob = vault.encrypt_vector("user-1", v)
    out = vault.decrypt_vector("user-1", blob)
    assert np.allclose(out, v)


def test_shred_blocks_decrypt() -> None:
    vault = CryptoShredVault()
    blob = vault.encrypt_vector("user-2", [1.0, 2.0, 3.0, 4.0])
    receipt = vault.shred("user-2")
    assert receipt.shredded
    with pytest.raises(RuntimeError, match="shredded"):
        vault.decrypt_vector("user-2", blob)
