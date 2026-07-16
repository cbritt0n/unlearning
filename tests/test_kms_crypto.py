"""KMS-backed crypto-shred tests (LocalFileKMS — no cloud SDKs required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compliance.kms_backends import KmsCryptoShredVault, LocalFileKMS


def test_local_kms_wrap_unwrap_shred(tmp_path: Path) -> None:
    kms = LocalFileKMS(tmp_path / "kms.json")
    vault = KmsCryptoShredVault(kms)

    v = np.arange(8, dtype=np.float32)
    blob = vault.encrypt_vector("user-9", v)
    out = vault.decrypt_vector("user-9", blob)
    assert np.allclose(out, v)

    receipt = vault.shred("user-9")
    assert receipt.shredded
    with pytest.raises(RuntimeError, match="shredded"):
        vault.decrypt_vector("user-9", blob)


def test_local_kms_persists_destroy(tmp_path: Path) -> None:
    path = tmp_path / "kms.json"
    kms = LocalFileKMS(path)
    vault = KmsCryptoShredVault(kms)
    blob = vault.encrypt_vector("e1", [1.0, 2.0, 3.0, 4.0])
    vault.shred("e1")

    kms2 = LocalFileKMS(path)
    vault2 = KmsCryptoShredVault(kms2)
    # Re-provision should fail for shredded entity on same KMS destroy set
    with pytest.raises(RuntimeError):
        kms2.unwrap_dek("e1", b"\x00" * 32)
